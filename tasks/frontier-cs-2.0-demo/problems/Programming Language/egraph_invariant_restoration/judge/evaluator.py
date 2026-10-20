"""Paired, judge-clock evaluator for the excised egg rebuilding pipeline."""
from __future__ import annotations

import ctypes, difflib, hashlib, json, math, os, random, re, shutil, statistics, struct, subprocess, sys, time
from pathlib import Path
TASK_DIR = Path(__file__).resolve().parent
if str(TASK_DIR) not in sys.path:
    sys.path.insert(0, str(TASK_DIR))
from submission_patch import apply_submission_patch

AUTHENTIC_ENTRYPOINT = "tests/simple.rs"
CONFIG = {"submission": {"kind": "patch", "path": "/app/solution.patch",
    "allow_paths": ["src/egraph.rs", "src/lib.rs", "src/tutorials/_01_background.rs",
                    "src/tutorials/mod.rs", "src/solver_extensions/"],
    "max_patch_bytes": 262144, "max_changed_files": 20}}
REFERENCE_CONFIG = {"submission": {**CONFIG["submission"], "allow_paths":
    ["src/egraph.rs", "src/lib.rs", "src/tutorials/_01_background.rs", "src/tutorials/mod.rs"]}}

REPETITIONS, WARMUPS = 9, 1
BUILD_TIMEOUT, RUN_TIMEOUT, EVALUATION_DEADLINE_SECONDS = 240, 30, 1700
MIN_SAMPLE_SECONDS, MIN_SAMPLE_RUNS = 4.0, 3
MATERIAL_SPEEDUP_RATIO = 0.95
BASELINE_LIB_SHA256 = "5e9f37d68721966322808dec166e7548381aa2fa139c7c8fcdb5053462280ff8"
UPSTREAM_LIB_SHA256 = "5ed4e29b3533219d05f3e95d0f5f7861503b8419126224fffaf24f3c0a291cf8"
DEV = [("dev_ematching", "math", "math_ematching_bench", {"EGG_ITER_LIMIT": "6", "EGG_SAMPLES": "20"}),
       ("dev_diff_explanations", "math", "diff_power_harder", {}),
       ("dev_associative", "math", "math_associate_adds", {})]
FINAL = [("final_integration", "math", "integ_part2", {}),
         ("final_radical", "math", "math_simplify_root", {}),
         ("final_factorization", "math", "math_simplify_factor", {})]

# x86_64 Linux Landlock syscalls and filesystem rights. Candidate processes see
# only their disposable tree and public runtime, never judge/reference paths.
CREATE, ADD, RESTRICT, VERSION, PATH_BENEATH, NO_NEW_PRIVS = 444, 445, 446, 1, 1, 38
EXEC, WRITE, READ, READDIR = 1, 2, 4, 8
ALL_FS, READ_EXEC = (1 << 15) - 1, EXEC | READ | READDIR

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
FORBIDDEN = re.compile(r"(?:unsafe\b|std\s*::\s*(?:fs|process|env)|include_(?:str|bytes)!|"
    r"option_env!|env!|/proc|reference\.patch|evaluator\.py|Command\s*::|process\s*::\s*exit)", re.I)


def _remaining(deadline, wanted):
    left = deadline - time.monotonic()
    if left <= 5:
        raise TimeoutError("overall evaluation deadline reached")
    return min(wanted, max(1.0, left - 5))


def _env(tree, extra=None):
    home, tmp = tree / ".frontier-home", tree / ".frontier-tmp"
    home.mkdir(exist_ok=True); tmp.mkdir(exist_ok=True)
    env = {"PATH": "/usr/local/cargo/bin:/usr/local/bin:/usr/bin:/bin",
            "HOME": str(home), "TMPDIR": str(tmp), "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8",
            "CARGO_NET_OFFLINE": "true", "CARGO_BUILD_JOBS": "8",
            "CARGO_HOME": "/usr/local/cargo", "RUSTUP_HOME": "/usr/local/rustup",
            "CARGO_TARGET_DIR": str(tree / "target"), "RUST_TEST_THREADS": "1",
            "RAYON_NUM_THREADS": "1", "RUSTUP_TOOLCHAIN": "1.94.1-x86_64-unknown-linux-gnu"}
    if extra: env.update(extra)
    return env


def _abi():
    return int(ctypes.CDLL(None, use_errno=True).syscall(CREATE, 0, 0, VERSION))


def _sandbox(tree):
    if _abi() < 3:
        raise RuntimeError("Landlock ABI 3+ unavailable")
    writable = tree.parent.resolve()
    def apply():
        libc = ctypes.CDLL(None, use_errno=True)
        attr = struct.pack("Q", ALL_FS)
        ruleset = libc.syscall(CREATE, attr, len(attr), 0)
        if ruleset < 0: os._exit(125)
        for path, access in (("/usr", READ_EXEC), ("/etc", READ_EXEC),
                             ("/dev", READ_EXEC | WRITE), (str(writable), ALL_FS)):
            fd = os.open(path, os.O_PATH | os.O_CLOEXEC)
            rule = struct.pack("Qi4x", access, fd)
            if libc.syscall(ADD, ruleset, PATH_BENEATH, rule, 0): os._exit(125)
            os.close(fd)
        if libc.prctl(NO_NEW_PRIVS, 1, 0, 0, 0) or libc.syscall(RESTRICT, ruleset, 0):
            os._exit(125)
        os.close(ruleset)
    return apply


def _run(cmd, tree, timeout, deadline, extra_env=None):
    return subprocess.run(cmd, cwd=tree, text=True, capture_output=True,
        timeout=_remaining(deadline, timeout), env=_env(tree, extra_env), preexec_fn=_sandbox(tree))


def _validate_source(tree, reference=False):
    base = TASK_DIR / "harbor" / "app"
    lib_digest = hashlib.sha256((tree / "src/lib.rs").read_bytes()).hexdigest()
    expected_lib = UPSTREAM_LIB_SHA256 if reference else BASELINE_LIB_SHA256
    if lib_digest != expected_lib:
        raise ValueError("src/lib.rs differs outside the source-locked privacy redaction")
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


def _invoke(binaries, workload, tree, deadline):
    _, target, name, *rest = workload
    return _run([str(binaries[target]), "--exact", name, "--nocapture", "--test-threads=1"],
                tree, RUN_TIMEOUT, deadline, rest[0] if rest else None)


def _prepare_tree(tree, deadline, reference=False):
    _validate_source(tree, reference=reference)
    (tree / "tests" / "frontier_correctness.rs").write_text(GUARD_SOURCE)
    build = _run(["cargo", "test", "--release", "--locked", "--offline", "--no-run",
        "--test", Path(AUTHENTIC_ENTRYPOINT).stem, "--test", "math",
        "--test", "frontier_correctness"],
        tree, BUILD_TIMEOUT, deadline)
    if build.returncode: raise ValueError("release build failed")
    deps, binaries = tree / "target" / "release" / "deps", {}
    for target in ("simple", "math", "frontier_correctness"):
        matches = [p for p in deps.glob(f"{target}-*") if p.is_file() and os.access(p, os.X_OK)]
        if len(matches) != 1: raise ValueError(f"could not identify {target} test binary")
        binaries[target] = matches[0]
    for name in ("frontier_invariants_and_extraction", "frontier_analysis_propagation",
                 "frontier_explanations"):
        guard = ("guard", "frontier_correctness", name)
        proc = _invoke(binaries, guard, tree, deadline); _require_one_test(proc, name)
    upstream = ("upstream", "simple", "simple_tests")
    proc = _invoke(binaries, upstream, tree, deadline); _require_one_test(proc, upstream[2])
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


def _sample(binaries, workload, tree, deadline):
    start, count, outputs = time.perf_counter(), 0, []
    while count < MIN_SAMPLE_RUNS or time.perf_counter() - start < MIN_SAMPLE_SECONDS:
        proc = _invoke(binaries, workload, tree, deadline)
        _require_one_test(proc, workload[2]); count += 1
        outputs.append(proc)
    elapsed = (time.perf_counter() - start) / count
    # Parsed after the timed loop so the wall-clock measurement is unchanged.
    reports = [_runner_report(proc) for proc in outputs]
    congruence = math.fsum(r["congruence_seconds"] for r in reports) / count
    return elapsed, count, congruence, reports[-1]


def _measure(candidate, reference, candidate_tree, reference_tree, suite, deadline):
    out_c, out_r, raw, cong_c, cong_r = {}, {}, {}, {}, {}
    rng = random.Random(0xE66)
    for workload in suite:
        key = workload[0]
        for _ in range(WARMUPS):
            _sample(candidate, workload, candidate_tree, deadline)
            _sample(reference, workload, reference_tree, deadline)
        cv, rv, ccv, rcv, pairs = [], [], [], [], []
        for ref_first in [bool(rng.getrandbits(1)) for _ in range(REPETITIONS)]:
            if ref_first:
                r, rn, rc, rrep = _sample(reference, workload, reference_tree, deadline)
                c, cn, cc, crep = _sample(candidate, workload, candidate_tree, deadline)
            else:
                c, cn, cc, crep = _sample(candidate, workload, candidate_tree, deadline)
                r, rn, rc, rrep = _sample(reference, workload, reference_tree, deadline)
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


def _is_reference_submission(solution_path):
    try:
        submitted_patch = Path(solution_path).read_bytes()
    except OSError:
        return False
    reference_patch = (TASK_DIR / "reference.patch").read_bytes()
    return hashlib.sha256(submitted_patch).digest() == hashlib.sha256(reference_patch).digest()


def evaluate(solution_path):
    candidate = reference = None
    deadline = time.monotonic() + EVALUATION_DEADLINE_SECONDS
    try:
        is_reference = _is_reference_submission(solution_path)
        candidate = apply_submission_patch(solution_path, TASK_DIR,
                                           REFERENCE_CONFIG if is_reference else CONFIG)
        if not candidate.ok: return 0.0, 0.0, candidate.message, {}
        reference = apply_submission_patch(TASK_DIR / "reference.patch", TASK_DIR, REFERENCE_CONFIG)
        if not reference.ok: return 0.0, 0.0, "reference unavailable", {}
        cb = _prepare_tree(candidate.workdir, deadline, reference=is_reference)
        rb = _prepare_tree(reference.workdir, deadline, reference=True)
        final = os.environ.get("FRONTIER_SUBMISSION_ROLE") == "final"
        suite = DEV + FINAL if final else DEV
        c, r, private_raw, cc, rc = _measure(
            cb, rb, candidate.workdir, reference.workdir, suite, deadline)
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
    except (Exception, subprocess.SubprocessError, OSError, ValueError):
        return 0.0, 0.0, "build, sandbox, deadline, or correctness failure", {}
    finally:
        for result in (candidate, reference):
            if result and result.workdir: shutil.rmtree(result.workdir.parent, ignore_errors=True)


if __name__ == "__main__":
    import sys
    print(json.dumps(dict(zip(("score", "score_unbounded", "message", "metrics"),
                              evaluate(sys.argv[1]))), sort_keys=True))
