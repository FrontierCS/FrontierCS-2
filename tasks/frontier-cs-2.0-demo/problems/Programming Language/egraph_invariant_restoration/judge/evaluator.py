"""Paired, judge-clock evaluator for the excised egg rebuilding pipeline (#757).

evaluate(tree) on the shared verifier: `tree` is pristine/ with only the editable files overlaid.
check_tree refuses the content rules that keep the measurement honest (content_policy). The
reference is pristine/ + reference.patch. Every build and every test process of candidate and
reference runs under the shared isolate launcher (isolate.run: no judge files, clean environment,
no network, private disk): a build instance returns the three test binaries, a guard instance
runs one guard, and each timed sample is one instance whose judge-written driver repeats the
workload process and times each invocation with the judge's wall clock, as before. Both pass
three invariant guards and simple_tests, then are timed in nine paired samples per workload
(after one warm-up).

Score: G_C = geomean over the active workloads of the candidate's median wall-clock seconds;
raw_score = score_unbounded = 100 * S / (S + G_C) with the fixed anchor S = 0.020 s.
Any build, guard, correctness, deadline or sandbox failure is a scored zero.

Metrics (one beat_spec_workload_passes row per workload; development: 3, final: 6):
  wall_clock_seconds  median process wall clock; lower; passes if C <= 0.95 * R
  congruence_seconds  median apply+rebuild seconds from the Runner report; lower;
                      passes if C <= 0.95 * R
beats_reference = 1 iff every workload passes the guards and both metrics.
The metric definitions before #757 are archived at docs/task-authoring/egraph_invariant_restoration/provenance/beat_spec.json.
"""
from __future__ import annotations

import difflib, json, math, os, random, re, shutil, statistics, subprocess, sys, tempfile, time
from pathlib import Path
from types import SimpleNamespace
TASK_DIR = Path(__file__).resolve().parent
if str(TASK_DIR) not in sys.path:
    sys.path.insert(0, str(TASK_DIR))
import isolate  # the shared launcher, beside edit_scope in the Verifier base

AUTHENTIC_ENTRYPOINT = "tests/simple.rs"
PRISTINE = TASK_DIR / "pristine"

REPETITIONS, WARMUPS = 9, 1
BUILD_TIMEOUT, RUN_TIMEOUT, EVALUATION_DEADLINE_SECONDS = 240, 30, 1700
MIN_SAMPLE_SECONDS, MIN_SAMPLE_RUNS = 4.0, 3
MATERIAL_SPEEDUP_RATIO = 0.95
DEV = [("dev_ematching", "math", "math_ematching_bench", {"EGG_ITER_LIMIT": "6", "EGG_SAMPLES": "20"}),
       ("dev_diff_explanations", "math", "diff_power_harder", {}),
       ("dev_associative", "math", "math_associate_adds", {})]
FINAL = [("final_integration", "math", "integ_part2", {}),
         ("final_radical", "math", "math_simplify_root", {}),
         ("final_factorization", "math", "math_simplify_factor", {})]

TARGETS = ("simple", "math", "frontier_correctness")
# The Rust toolchain lives under /usr/local (readable in every instance); cargo stays offline on
# the vendored dependencies and writes only inside the instance.
PATH = "/usr/local/cargo/bin:/usr/local/bin:/usr/bin:/bin"
ENV = {"CARGO_NET_OFFLINE": "true", "CARGO_BUILD_JOBS": "8",
       "CARGO_HOME": "/usr/local/cargo", "RUSTUP_HOME": "/usr/local/rustup",
       "CARGO_TARGET_DIR": "{work}/target", "RUST_TEST_THREADS": "1",
       "RAYON_NUM_THREADS": "1", "RUSTUP_TOOLCHAIN": "1.94.1-x86_64-unknown-linux-gnu"}
# Build instance: compile, then hand back the uniquely identified test binaries.
BUILD_SCRIPT = r'''
cargo test --release --locked --offline --no-run --test simple --test math --test frontier_correctness >&2 || exit 3
for target in simple math frontier_correctness; do
  found=""; count=0
  for path in target/release/deps/"$target"-*; do
    if [ -f "$path" ] && [ -x "$path" ]; then found=$path; count=$((count + 1)); fi
  done
  [ "$count" = 1 ] || exit 4
  cp "$found" "$1/$target" || exit 5
done
'''
# Sample driver (judge code, runs inside the instance): repeats the workload process until at
# least min_runs have run and min_seconds have elapsed, timing each invocation with
# perf_counter around the whole process exactly as the judge did before; prints one JSON object.
DRIVER = r'''
import json, subprocess, sys, time
spec = json.loads(sys.argv[1])
command = [spec["binary"], *spec["args"]]
runs, count = [], 0
start = time.perf_counter()
while count < spec["min_runs"] or time.perf_counter() - start < spec["min_seconds"]:
    try:
        proc = subprocess.run(command, text=True, capture_output=True, timeout=spec["timeout"])
    except subprocess.TimeoutExpired:
        print(json.dumps({"error": "timeout"}))
        raise SystemExit(0)
    count += 1
    runs.append({"returncode": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr})
    if proc.returncode:
        break
elapsed = (time.perf_counter() - start) / count
print(json.dumps({"elapsed": elapsed, "count": count, "runs": runs}))
'''

GUARD_SOURCE = r'''use egg::*;
define_language! { enum GuardLang { Symbol(Symbol), "f" = F(Id), "+" = Add([Id; 2]), Num(i32), } }

#[derive(Default)]
struct Reachability;
impl Analysis<GuardLang> for Reachability {
    type Data = u64;
    fn make(egraph: &mut EGraph<GuardLang, Self>, node: &GuardLang, _id: Id) -> u64 {
        match node {
            GuardLang::Num(n) => 1u64 << (*n as u32),
            GuardLang::Symbol(_) => 0,
            _ => node.fold(0, |bits, child| bits | egraph[child].data),
        }
    }
    fn merge(&mut self, to: &mut u64, from: u64) -> DidMerge {
        let before = *to;
        *to |= from;
        DidMerge(*to != before, from != *to)
    }
}

#[test]
fn frontier_invariants_and_extraction() {
    let mut g: EGraph<GuardLang, ()> = EGraph::default();
    let a = g.add(GuardLang::Symbol("a".into()));
    let b = g.add(GuardLang::Symbol("b".into()));
    let fa = g.add(GuardLang::F(a)); let fb = g.add(GuardLang::F(b));
    g.union(a, b); g.rebuild();
    assert!(g.clean); assert_eq!(g.find(fa), g.find(fb));
    assert_eq!(g.add(GuardLang::F(a)), g.find(fa));
    for class in g.classes() {
        assert_eq!(class.id, g.find(class.id));
        for node in &class.nodes { for child in node.children() { assert_eq!(*child, g.find(*child)); } }
    }
    let expr: RecExpr<GuardLang> = "(+ 0 (+ 0 x))".parse().unwrap();
    let rules: Vec<Rewrite<GuardLang, ()>> = vec![rewrite!("zero"; "(+ 0 ?x)" => "?x")];
    let runner = Runner::default().with_expr(&expr).run(&rules);
    let (_, best) = Extractor::new(&runner.egraph, AstSize).find_best(runner.roots[0]);
    assert_eq!(best.to_string(), "x");
}

#[test]
fn frontier_analysis_propagation() {
    let mut g: EGraph<GuardLang, Reachability> = EGraph::default();
    let a = g.add(GuardLang::Num(0));
    let b = g.add(GuardLang::Num(1));
    let fa = g.add(GuardLang::F(a));
    let sum = g.add(GuardLang::Add([fa, b]));
    g.union(a, b);
    g.rebuild();
    assert_eq!(g[a].data, 0b11);
    assert_eq!(g[fa].data, 0b11);
    assert_eq!(g[sum].data, 0b11);
}

#[test]
fn frontier_explanations() {
    let start: RecExpr<GuardLang> = "a".parse().unwrap();
    let end: RecExpr<GuardLang> = "b".parse().unwrap();
    let rules: Vec<Rewrite<GuardLang, ()>> = vec![rewrite!("a-to-b"; "a" => "b")];
    let mut runner = Runner::default().with_explanations_enabled().with_expr(&start).run(&rules);
    let proof = runner.explain_equivalence(&start, &end).get_flat_strings();
    assert!(!proof.is_empty());
}
'''
# The fixed upstream harness (src/test.rs, src/run.rs; outside the editable surface)
# prints exactly one Runner report per workload process. Its Apply and Rebuild
# lines are Runner's own Instant measurements around applying matches and
# EGraph::rebuild, i.e. the paper's congruence-maintenance time (Section 3.4).
RUNNER_REPORT_RE = re.compile(
    # libtest prints "test <name> ... " immediately before the report on stdout,
    # so the block may start mid-line; its internal line structure is exact.
    r"(?m)Runner report\n=+\n  Stop reason: [^\n]*\n  Iterations: (\d+)\n"
    r"  Egraph size: (\d+) nodes, (\d+) classes, (\d+) memo\n  Rebuilds: (\d+)\n"
    r"  Total time: ([^\s]+)\n    Search:  \([^)\n]*\) ([^\s]+)\n"
    r"    Apply:   \([^)\n]*\) ([^\s]+)\n    Rebuild: \([^)\n]*\) ([^\s]+)$")
# Measurement rules (the sandbox makes judge files, environment and network unreachable, so the
# old verifier-access tokens are gone): `unsafe` would let solver code patch the in-process Runner
# timers; files and helper processes would let one timed invocation hand work to the next within
# a sample; process::exit could end a process after forging its test summary.
FORBIDDEN = re.compile(r"(?:unsafe\b|std\s*::\s*(?:fs|process)|Command\s*::|process\s*::\s*exit)",
                       re.I)


def _remaining(deadline, wanted):
    left = deadline - time.monotonic()
    if left <= 5:
        raise TimeoutError("overall evaluation deadline reached")
    return min(wanted, max(1.0, left - 5))


def _instance(argv, deadline, wanted, env=ENV, **kwargs):
    """One sandboxed instance; an instance cut by the deadline ends the evaluation."""
    timeout = _remaining(deadline, wanted)
    result = isolate.run(argv, timeout_seconds=timeout, env=env, path=PATH, **kwargs)
    if result.timed_out:
        raise TimeoutError("sandboxed instance exceeded its time limit")
    return result


def _validate_source(tree):
    base = PRISTINE
    for rel in ("src/egraph.rs", "src/tutorials/_01_background.rs", "src/tutorials/mod.rs"):
        delta = difflib.ndiff((base / rel).read_text(errors="replace").splitlines(),
                             (tree / rel).read_text(errors="replace").splitlines())
        additions = "\n".join(x[2:] for x in delta if x.startswith("+ "))
        if FORBIDDEN.search(additions):
            raise ValueError("editable source uses forbidden verifier-access capability")
    ext = tree / "src" / "solver_extensions"
    if ext.exists():
        for path in ext.rglob("*.rs"):
            if FORBIDDEN.search(path.read_text(errors="replace")):
                raise ValueError("solver extension uses forbidden verifier-access capability")


def _require_one_test(proc, name):
    summary = re.search(r"test result: ok\. 1 passed; 0 failed; 0 ignored; 0 measured; "
                        r"\d+ filtered out; finished in [0-9.]+s", proc.stdout)
    named = (f"test {name} ... " in proc.stdout and
             (re.search(rf"(?m)^test {re.escape(name)} \.\.\. ok$", proc.stdout)
              or re.search(r"(?m)^ok$", proc.stdout)))
    if proc.returncode or not summary or not named or "running 1 test" not in proc.stdout:
        raise ValueError("workload did not execute exactly one expected test")


def _invoke(binaries, workload, deadline, min_runs=1, min_seconds=0.0):
    """Run one workload process (or a sample of them) in one instance; returns (elapsed, procs)."""
    _, target, name, *rest = workload
    extra = rest[0] if rest else {}
    spec = {"binary": "{work}/" + target, "min_runs": min_runs, "min_seconds": min_seconds,
            "args": ["--exact", name, "--nocapture", "--test-threads=1"],
            "timeout": _remaining(deadline, RUN_TIMEOUT)}
    argv = [sys.executable, "-I", "-S", "-c", DRIVER, json.dumps(spec)]
    result = _instance(argv, deadline, EVALUATION_DEADLINE_SECONDS, tree=binaries[target],
                       env={**ENV, **extra}, output_bytes=256 << 20)
    if result.returncode or result.limit:
        raise ValueError("workload driver failed")
    report = json.loads(result.stdout)
    if "error" in report:
        raise TimeoutError("workload process exceeded its time limit")
    procs = [SimpleNamespace(**run) for run in report["runs"]]
    return report["elapsed"], procs


def _prepare_tree(tree, deadline):
    _validate_source(tree)
    (tree / "tests" / "frontier_correctness.rs").write_text(GUARD_SOURCE)
    build = _instance(["sh", "-c", BUILD_SCRIPT, "build", "{out}"], deadline, BUILD_TIMEOUT,
                      tree=tree, outputs=TARGETS, output_bytes=1 << 30)
    if build.returncode == 4: raise ValueError("could not identify a test binary")
    if build.returncode or build.limit: raise ValueError("release build failed")
    binaries = {}
    for target in TARGETS:
        data = build.outputs.get(target)
        if not data: raise ValueError(f"could not identify {target} test binary")
        # Each binary in a directory of its own: the instance copies only that one.
        directory = tree.parent / "bin" / target
        directory.mkdir(parents=True)
        (directory / target).write_bytes(data)
        (directory / target).chmod(0o755)
        binaries[target] = directory
    for name in ("frontier_invariants_and_extraction", "frontier_analysis_propagation",
                 "frontier_explanations"):
        guard = ("guard", "frontier_correctness", name)
        _, procs = _invoke(binaries, guard, deadline)
        _require_one_test(procs[0], name)
    upstream = ("upstream", "simple", "simple_tests")
    _, procs = _invoke(binaries, upstream, deadline)
    _require_one_test(procs[0], upstream[2])
    return binaries


def _runner_report(proc):
    """Parse the single trusted Runner report of one workload process.

    A second report (for example text printed by solver code) or a malformed
    one rejects the invocation instead of being averaged.
    """
    text = proc.stdout + "\n" + proc.stderr
    if text.count("Runner report") != 1:
        raise ValueError("workload did not print exactly one Runner report")
    match = RUNNER_REPORT_RE.search(text)
    if not match:
        raise ValueError("malformed Runner report")
    search, apply_, rebuild = (float(match.group(i)) for i in (7, 8, 9))
    if not all(math.isfinite(v) and v >= 0 for v in (search, apply_, rebuild)):
        raise ValueError("malformed Runner report timing")
    return {"apply_seconds": apply_, "rebuild_seconds": rebuild,
            "congruence_seconds": apply_ + rebuild,
            "iterations": int(match.group(1)), "runner_rebuilds": int(match.group(5))}


def _sample(binaries, workload, deadline):
    elapsed, outputs = _invoke(binaries, workload, deadline, MIN_SAMPLE_RUNS, MIN_SAMPLE_SECONDS)
    for proc in outputs:
        _require_one_test(proc, workload[2])
    count = len(outputs)
    reports = [_runner_report(proc) for proc in outputs]
    congruence = math.fsum(r["congruence_seconds"] for r in reports) / count
    return elapsed, count, congruence, reports[-1]


def _measure(candidate, reference, suite, deadline):
    out_c, out_r, raw, cong_c, cong_r = {}, {}, {}, {}, {}
    rng = random.Random(0xE66)
    for workload in suite:
        key = workload[0]
        for _ in range(WARMUPS):
            _sample(candidate, workload, deadline)
            _sample(reference, workload, deadline)
        cv, rv, ccv, rcv, pairs = [], [], [], [], []
        for ref_first in [bool(rng.getrandbits(1)) for _ in range(REPETITIONS)]:
            if ref_first:
                r, rn, rc, rrep = _sample(reference, workload, deadline)
                c, cn, cc, crep = _sample(candidate, workload, deadline)
            else:
                c, cn, cc, crep = _sample(candidate, workload, deadline)
                r, rn, rc, rrep = _sample(reference, workload, deadline)
            cv.append(c); rv.append(r); ccv.append(cc); rcv.append(rc)
            pairs.append({"candidate_seconds": c, "reference_seconds": r,
                          "candidate_runs": cn, "reference_runs": rn,
                          "candidate_congruence_seconds": cc,
                          "reference_congruence_seconds": rc,
                          # Diagnostics only: Runner's last-invocation report.
                          # "runner_rebuilds" sums values returned by the
                          # candidate's own EGraph::rebuild and is not a
                          # trusted repair-call count.
                          "candidate_last_report": crep,
                          "reference_last_report": rrep})
        out_c[key], out_r[key], raw[key] = statistics.median(cv), statistics.median(rv), pairs
        cong_c[key], cong_r[key] = statistics.median(ccv), statistics.median(rcv)
        if not (cong_c[key] > 0 and cong_r[key] > 0):
            raise ValueError("congruence-maintenance time was not measured")
    return out_c, out_r, raw, cong_c, cong_r


def _score_timings(candidate_times, reference_times, suite, final=False,
                   candidate_congruence=None, reference_congruence=None):
    scale_seconds = 0.020
    candidate_objective = math.exp(statistics.mean(
        math.log(candidate_times[k]) for k, *_ in suite))
    reference_objective = math.exp(statistics.mean(
        math.log(reference_times[k]) for k, *_ in suite))
    score = 100.0 * scale_seconds / (scale_seconds + candidate_objective)
    reference_score = 100.0 * scale_seconds / (scale_seconds + reference_objective)
    if candidate_congruence is None or reference_congruence is None:
        raise ValueError("congruence-maintenance measurements are required")
    rows = [{"workload": f"workload_{i + 1}",
             "metrics": {"wall_clock_seconds":
                         candidate_times[k] <= MATERIAL_SPEEDUP_RATIO * reference_times[k],
                         "congruence_seconds":
                         candidate_congruence[k]
                         <= MATERIAL_SPEEDUP_RATIO * reference_congruence[k]}}
            for i, (k, *_rest) in enumerate(suite)]
    metrics = {"beat_spec_workload_passes": rows,
               "beats_reference": int(all(value for r in rows for value in r["metrics"].values())),
               "reference_score": reference_score,
               "margin": score - reference_score}
    if not final:
        metrics["candidate_dev_wall_clock_seconds_median"] = statistics.median(candidate_times.values())
        metrics["candidate_dev_congruence_seconds_median"] = statistics.median(
            candidate_congruence[k] for k, *_ in suite)
    return score, score, metrics


class _Tree:
    """A disposable judge-side copy (plus its built binaries) in its own directory."""

    def __init__(self, source):
        self.parent = Path(tempfile.mkdtemp(prefix="egraph-"))
        self.workdir = self.parent / "tree"
        shutil.copytree(source, self.workdir, symlinks=True)
        for directory, _names, files in os.walk(self.workdir):
            for name in [directory, *(os.path.join(directory, f) for f in files)]:
                if not os.path.islink(name):
                    os.chmod(name, os.stat(name).st_mode | 0o200)

    def dispose(self):
        shutil.rmtree(self.parent, ignore_errors=True)


def _reference_tree():
    tree = _Tree(PRISTINE)
    proc = subprocess.run(["git", "apply", "--whitespace=nowarn", str(TASK_DIR / "reference.patch")],
                          cwd=tree.workdir, capture_output=True, text=True, timeout=300,
                          env={**os.environ, "GIT_CEILING_DIRECTORIES": str(tree.parent)})
    if proc.returncode:
        tree.dispose()
        raise RuntimeError("reference unavailable")
    return tree


def check_tree(tree, pristine, changes):
    """Shared-verifier hook: the edited sources use no forbidden verifier-access capability."""
    try:
        _validate_source(Path(tree))
    except ValueError:
        return "content_policy", sorted(c.path for c in changes if c.status != "D")
    return None


def evaluate(tree):
    candidate = reference = None
    deadline = time.monotonic() + EVALUATION_DEADLINE_SECONDS
    try:
        candidate = _Tree(tree)
        try:
            reference = _reference_tree()
        except RuntimeError:
            return 0.0, 0.0, "reference unavailable", {}
        cb = _prepare_tree(candidate.workdir, deadline)
        rb = _prepare_tree(reference.workdir, deadline)
        final = os.environ.get("FRONTIER_SUBMISSION_ROLE") == "final"
        suite = DEV + FINAL if final else DEV
        c, r, private_raw, cc, rc = _measure(cb, rb, suite, deadline)
        score, unbounded, metrics = _score_timings(c, r, suite, final, cc, rc)
        metrics["private_timing_evidence"] = {
            key: {"candidate_median_seconds": c[key],
                  "reference_median_seconds": r[key],
                  "candidate_median_congruence_seconds": cc[key],
                  "reference_median_congruence_seconds": rc[key],
                  "pairs": private_raw.get(key, [])}
            for key, *_ in suite
        }
        metrics["candidate_geometric_mean_wall_clock_seconds"] = math.exp(
            statistics.mean(math.log(c[key]) for key, *_ in suite))
        metrics["reference_geometric_mean_wall_clock_seconds"] = math.exp(
            statistics.mean(math.log(r[key]) for key, *_ in suite))
        return score, unbounded, "valid; independent guards and workloads passed", metrics
    except isolate.IsolateError:
        raise  # the sandbox could not be built: a verification failure, not a score
    except (Exception, subprocess.SubprocessError, OSError, ValueError):
        return 0.0, 0.0, "build, sandbox, deadline, or correctness failure", {}
    finally:
        for result in (candidate, reference):
            if result: result.dispose()


if __name__ == "__main__":
    import sys
    print(json.dumps(dict(zip(("score", "score_unbounded", "message", "metrics"),
                              evaluate(sys.argv[1]))), sort_keys=True))
