"""FrontierCS's conventions, which every ported task shares, and `fcs_task`, which writes the
workflow of the task in the entry file's package, `tasks/<problem id>/`, from the task's own
values. Each value is the published task's, but where the README's "How the ports differ" names
another."""

load("@pumpkin//harness/claude_code/claude_code.star", "claude_code")
load("@pumpkin//harness/codex/codex.star", "codex")
load("@pumpkin//harness/pi/pi.star", "pi")
load("@pumpkin//harness/qwen_code/qwen_code.star", "qwen_code")
load("@pumpkin//harness/zcode/zcode.star", "zcode")
load("//lib/codex-catalog/entries.star", "codex_entry")
load("//lib/submit.star", "FINAL_HELP", "submit")
load("//lib/verify.star", "PAYLOAD", "final_call", "train_call")

# The run's choices, each given as `--arg <name>=<value>`, or else its default; a task's
# `task.star` may declare more of its own.
# A harness of `HARNESSES`, which works on `model`, or `reference`, which submits the task's
# reference solution, with no model.
AGENT = arg("agent", str, default = "codex")

# The model the harness works on: the `name` of a model under some upstream's `models` in the
# machine's configuration. Required when the agent is a harness.
MODEL = arg("model", str, default = None)

# None: the effort FrontierCS evaluated the harness at, its `effort` in `HARNESSES`.
EFFORT = arg("effort", str, default = None)

# How Codex's tools reach the model; Codex's alone. `functions` is the form every Responses API
# upstream forwards, for a model whose catalog entry `lib/codex-catalog` carries; `native` sends
# forms that only some upstreams (OpenAI's own) forward, and through any other Codex has no shell or
# file tools and exits at once. The task can't know the upstream, so the default is the one that
# works everywhere.
TOOLS = arg("tools", str, default = "functions")

# The harnesses, each with its `function`, one of Pumpkin's loaded above; its `effort`, the one
# FrontierCS evaluated it at (`config/candidates.json`); and its `keywords`, the arguments that
# only it takes, which `_harness` passes beside those that every harness takes.
HARNESSES = {
    "codex": {
        "function": codex,
        "effort": "xhigh",
        "keywords": {"tools": TOOLS, "catalog_entry": codex_entry(MODEL)},
    },
    "claude-code": {
        "function": claude_code,
        # The cap FrontierCS gives Claude Code, the model's own, above Claude Code's default
        # (`candidates/claude-code/launcher.py:25-27`).
        "effort": "max",
        "keywords": {"max_output_tokens": 128000},
    },
    "zcode": {
        "function": zcode,
        # The window FrontierCS gives every model, and half its cap of 262144 output tokens, the
        # most z.ai's endpoint takes (`candidates/zcode/README.md:30-31`).
        "effort": "max",
        "keywords": {"context_tokens": 1048576, "max_output_tokens": 131072},
    },
    "qwen-code": {
        "function": qwen_code,
        # The window and the body FrontierCS gives Qwen 3.8 Max, the model it evaluated Qwen Code
        # on (`candidates/qwen-code/launcher.py:63-87`). FrontierCS leaves the output cap to the
        # upstream; this is the most Qwen 3.8 Max answers with, as the catalog Qwen Code is built
        # with says.
        "effort": "high",
        "keywords": {
            "context_tokens": 1000000,
            "max_output_tokens": 131072,
            "extra_body": {"temperature": 1, "enable_thinking": True},
        },
    },
    "pi": {
        "function": pi,
        # The window and the cap FrontierCS gives Pi (`candidates/pi-gemini/launcher.py:27-28`).
        "effort": "high",
        "keywords": {"context_tokens": 1048576, "max_output_tokens": 1048576},
    },
}

# The work phase's deadline; none, `WALL_TIME_SECONDS`.
DEADLINE = arg("deadline_seconds", int, default = None)

# Each train call's timeout, which covers the whole call; none, the task's `judge_timeout_seconds`,
# as FrontierCS times a train call, and the time `_moved_seconds` gives.
TRAIN_TIMEOUT = arg("train_call_timeout_seconds", int, default = None)

# The work phase's deadline that FrontierCS gives every task (`[quota]`'s `wall_time_seconds`).
WALL_TIME_SECONDS = 18000

# How long the agent may keep running after its final submission is accepted.
FINALIZE_GRACE_SECONDS = 60

# The user every FrontierCS verifier runs as: each task's verifier image names it, on a base that
# makes it the owner of /frontier, the verifier's home.
VERIFIER_USER = {"uid": 65532, "gid": 65532}

# The agent's user, the base image's `ubuntu`, which owns /workspace.
AGENT_USER = {"uid": 1000, "gid": 1000}

TRAIN_HELP = (
    "Evaluate a submission tar on training workloads. " +
    "See Feedback in /frontier/task/README.md for the returned public score and metrics."
)

# The metrics the shared verifier adds to every result's `metrics`, after the task's own.
VERIFIER_METRICS = ["score_unbounded", "beat_spec_progress", "rejected"]

# Pinned fetch helpers needed by individual recipes.
FETCH = {"fetch": "//fetch"}

# The modes the verifier judges in: train calls in `train`, the final submission in `eval`.
MODES = ["train", "eval"]

# Where the verifier's image holds each mode's selector, unless a task's `datasets` says.
SELECTORS = {
    "train": "/opt/frontier/datasets/train.json",
    "eval": "/opt/frontier/datasets/eval.json",
}

# The member of a submission's tar that holds its patch, which every FrontierCS task fixes
# (`payload/solution.patch`, in the source's `tasks/shared/task-defaults.toml`).
SUBMISSION_MEMBER = "payload/solution.patch"

# A package with its own agent recipe keeps it beside the agent README.
AGENT_DIR = "agent-base"
AGENT_RECIPE = AGENT_DIR + "/Dockerfile"

# The directory of the recipe that joins the agent's README from `AGENT_DIR`'s and the platform's
# text there (`join.py` says how), in an image that the agent's recipe names `readme`.
AGENT_README = "//lib/agent-readme"
VERIFIER_RECIPE = "verifier/Dockerfile"

# What a machine can have besides CPU and memory, as Pumpkin's `needs` names it: a GPU
# (`nvidia-smi`'s name for the device), the CPU flags that `/proc/cpuinfo` lists, or the CPU model
# that it names.
H100 = {"count": 1, "model": "NVIDIA H100 80GB HBM3"}
A100_80 = {"count": 1, "model": "NVIDIA A100-SXM4-80GB"}
X86_64_V2 = ["sse4_1", "ssse3"]
X86_64_V3 = ["avx2", "bmi2", "fma", "abm", "movbe"]
AVX512 = ["avx512f", "avx512dq", "avx512bw"]
XEON_8375C = "Intel(R) Xeon(R) Platinum 8375C CPU @ 2.90GHz"

# The size of FrontierCS's GPU runners, where a task's source states none: a task's `[verifier]` and
# `[agent]` are the CPU control pod's and the agent's CPU box's sizes, while FrontierCS starts the
# Verifier GPU runner with 8 CPUs and 64 GiB (`scripts/maintenance/isolate_runner.py:57`,
# `scripts/maintenance/sandbox_runner.py:117`) and runs the agent's bench commands on a leased
# runner of that size. A smaller sandbox gets a judge killed for memory or past its profile's
# runner bound.
GPU_RUNNER = {"cpu_millis": 8000, "memory_bytes": 64 * GiB}

# The GPUs that an agent's runners may have, by the name of FrontierCS's `[gpu.agent]`
# `allowed_types`.
GPU_TYPES = {"H100": H100, "A100-80GB": A100_80}

# The role of the agent's GPU runners, and the agent's session on them.
GPU_RUNNER_ROLE = "gpu_runner"
GPU_GRANT = "gpu"

# The directory of `frontier-gpu`, the agent's command line for its GPU runners, with FrontierCS's
# commands over the session's functions, `sandboxctl call gpu.<op>`; of the skill that tells the
# agent how to use it; and of the recipe that adds both to the agent's image, which it names
# `agent-base`. The program reads its session's name and its runners' GPU type from the
# environment that the agent's command sets.
FRONTIER_GPU = "//lib/frontier-gpu"

# Where a role with model routes reaches them: the endpoint that its sandbox serves, as the base URL
# of an OpenAI client, below which the Chat Completions API is `/chat/completions`.
MODEL_BASE_URL = "http://127.0.0.1:11235/v1"

# Where a leased runner's commands start: FrontierCS's runner user's home, below which every test
# sandbox's `workdir` is (`scripts/task_manifest.py`'s `_command_sandboxes`).
RUNNER_HOME = "/workspace"

# What the reference agent needs: it only copies a file and calls `sandboxctl`, so it states
# neither a GPU nor a CPU feature whatever the task's agent needs, and never holds a specialised
# machine through the train call it waits on.
REFERENCE_NEEDS = {"cpu_millis": 1000, "memory_bytes": 512 * MiB}

# Every harness starts from the same task statement and public grant guide.
PROMPT_START = """Solve the evaluation task in /frontier/task/README.md. Use /.pumpkin/bin/sandboxctl for the
grant guide's commands.

Before doing any task work, read that file completely. It gives the train and final archive
layout, scoring rules and feedback. Work in /workspace. Use the train grant to evaluate your
patch, inspect its score and remaining calls, and iterate while useful. Save your first usable
archive with the final grant and replace it when you have a better answer.
"""

# What the agent's prompt adds for its GPU runners, as FrontierCS's Codex candidate's prompt
# does (`candidates/codex/launcher.py`), with the skill's and the helper's paths here.
GPU_PROMPT = """
This task supports Agent GPU execution: read the `frontier-gpu` skill at
/opt/frontier/candidate/skills/frontier-gpu/SKILL.md and use its
create/push/exec/pull/delete workflow, with /opt/frontier/candidate/bin/frontier-gpu,
within the task's GPU budget. Use `frontier-gpu quota` to check the GPU seconds and
commands left before debugging.
"""

PROMPT_END = """
When the answer is ready, submit its archive with finalization enabled exactly once as your
last platform action, then exit. Web search is disabled; use only local task files and tools.
"""

# The reference agent's script uses files from /opt/frontier/task. It makes one train call,
# waits for its end, and submits the same tar as the final one. The tar is the layout every
# task's README gives. The patch it holds is the task's reference patch or, for a task that has
# none, an empty one, which the starter's own code answers, or, where the judge refuses an empty
# patch, one that only appends a comment to a file of the starter (`noop_patch_file`).
REFERENCE_HEAD = """set -eu
ctl=/.pumpkin/bin/sandboxctl
member=""" + SUBMISSION_MEMBER + """
mkdir --parents "$(dirname "$member")"
"""
REFERENCE_PATCH = 'cp /opt/frontier/task/reference.patch "$member"\n'
EMPTY_PATCH = ': > "$member"\n'

# The suffixes of files that read `#` as a comment, which the no-op patch's comment needs.
NOOP_PATCH_SUFFIXES = [".py", ".sh", ".yaml", ".yml", ".toml"]

# What `noop_patch_file` may not hold, since the script puts it in a shell command.
NOOP_PATCH_SHELL = [" ", "\t", "\n", "\"", "'", "`", "$", "\\", ";", "&", "|", "<", ">", "(", ")", "*", "?", "[", "]", "{", "}", "!", "#", "~"]

# The starter's copy in the judge's image is `pristine/`, read-only, so the two copies are written
# with `cat`, not `cp`; `diff` exits 1 when the files differ. The copies go below the working
# directory, since the agent's `/tmp` may be read-only.
NOOP_PATCH = """work="$PWD/noop-work"
mkdir --parents "$work/a/$(dirname {file})" "$work/b/$(dirname {file})"
cat /opt/frontier/task/pristine/{file} > "$work/a/{file}"
cat /opt/frontier/task/pristine/{file} > "$work/b/{file}"
printf '\\n# Unchanged: the starting point, as the reference run submits it.\\n' >> "$work/b/{file}"
(cd "$work" && diff --unified --label a/{file} --label b/{file} a/{file} b/{file}) > "$member" || test $? -eq 1
"""
REFERENCE_TAIL = """tar --create --file=reference.tar --format=ustar --sort=name --mtime='UTC 1970-01-01' \\
    --owner=0 --group=0 "$member"
"$ctl" call train reference.tar --wait {seconds} || {{
    code=$?
    test "$code" -eq 2 || exit "$code"
}}
exec "$ctl" call final reference.tar --arg finalize=true
"""

# What a URL or an identifier of `source` may not hold, and the digits of a commit's ID.
URL_SPACE = [" ", "\t", "\n"]
HEX_DIGITS = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "a", "b", "c", "d", "e", "f"]

# The years a paper's `year` may be.
YEARS = (1900, 2100)

# FrontierCS's categories, each the name of the folder that holds a task of it.
CATEGORIES = [
    "Arch",
    "Bioinformatics",
    "Crypto",
    "Database",
    "HPC",
    "MLSys",
    "Machine Learning",
    "Network",
    "Programming Language",
    "Robotics",
    "Security",
    "Software Engineering",
    "Theory",
]

def _text(what, value, instead = "give the source's, or leave it out"):
    """`value`, given as `what`, once checked: a string that is not blank, or else `instead`."""
    if not value.replace(" ", ""):
        fail("{} is empty: {}".format(what, instead))
    return value

def _url(what, url):
    """`url`, given as `what`, once checked: an `https://` address of a host, with no space."""
    if not url.startswith("https://"):
        fail("{} is `{}`: give its https:// address".format(what, url))
    if "." not in url.split("/")[2] or [c for c in URL_SPACE if c in url]:
        fail("{} is `{}`: give one URL, `https://<host>/<path>`, with no space".format(what, url))
    return url

def _authors(authors):
    """`paper()`'s `authors` once checked: a list of names, none of them blank."""
    if authors == []:
        fail("`paper()`'s `authors` is empty: give the source's names, or leave it out")
    if [name for name in authors if not name.replace(" ", "")]:
        fail("`paper()`'s `authors` holds an empty name: give each author's name as the source writes it")
    return authors

def _year(year):
    """`paper()`'s `year` once checked: an int in `YEARS`."""
    if year != int(year):
        fail("`paper()`'s `year` is `\"{}\"`: give it as an int, as `year = {}`".format(year, int(year)))
    if year < YEARS[0] or year > YEARS[1]:
        fail("`paper()`'s `year` is {}: give the year the paper appeared, from {} to {}".format(year, YEARS[0], YEARS[1]))
    return year

def _doi(doi):
    """`paper()`'s `doi` once checked: `10.<registrant>/<suffix>`, with no space."""
    prefix = doi.split("/")[0]
    if doi.startswith("https://") or not prefix.startswith("10.") or prefix == "10." or doi == prefix or doi.endswith("/") or [c for c in URL_SPACE if c in doi]:
        fail("`paper()`'s `doi` is `{}`: give the DOI itself, `10.<registrant>/<suffix>`, as `10.1145/3729303`".format(doi))
    return doi

def paper(url, *, title = None, venue = None, authors = None, year = None, doi = None, summary = None):
    """The paper that a task's problem comes from, for `fcs_task`'s `source`.

    Each value is the source's, where it gives one.

    Args:
      url: where the paper reads: an `https://` address, of any host.
      title: the paper's title.
      venue: where the paper appeared, as `PLDI 2025`.
      authors: the authors' names, in the source's order.
      year: the year it appeared, an int.
      doi: its DOI, as `10.1145/3729303`, not a URL.
      summary: what the task makes of the paper, as the source words it.

    Returns:
      A value of `source`.
    """
    return {
        "of": paper,
        "url": _url("`paper()`'s URL", url),
        "title": title if title == None else _text("`paper()`'s `title`", title),
        "venue": venue if venue == None else _text("`paper()`'s `venue`", venue),
        "authors": authors if authors == None else _authors(authors),
        "year": year if year == None else _year(year),
        "doi": doi if doi == None else _doi(doi),
        "summary": summary if summary == None else _text("`paper()`'s `summary`", summary),
    }

def artifacts(url, *, commit = None, license = None):
    """The code that a task's paper published, for `fcs_task`'s `source`.

    Args:
      url: the repository: an `https://` address, of any host.
      commit: the full ID of the commit that the source names, where it names one.
      license: the code's license, as its SPDX identifier (`MIT`), where the source gives it.

    Returns:
      A value of `source`.
    """
    _url("`artifacts()`'s URL", url)
    if commit != None:
        # Splitting the ID at a digit gives one piece more than the digit occurs in it. The
        # language takes no `len` of a string, so a character besides the digits goes unseen.
        digits = len([piece for digit in HEX_DIGITS for piece in commit.split(digit)]) - len(HEX_DIGITS)
        if digits not in [40, 64] or [c for c in URL_SPACE if c in commit]:
            fail("`artifacts()`'s `commit` is `{}`: give the commit's full ID, 40 hexadecimal digits in lower case (64 in a SHA-256 repository)"
                .format(commit))
        if commit in url:
            fail("`artifacts()`'s URL `{}` holds its commit: give the repository's own URL, and the commit as `commit`"
                .format(url))
    if license != None and (not license.replace(" ", "") or [c for c in URL_SPACE if c in license]):
        fail("`artifacts()`'s `license` is `{}`: give its SPDX identifier, as `MIT` or `Apache-2.0`".format(license))
    return {"of": artifacts, "url": url, "commit": commit, "license": license}

# What `source` may hold: the values these functions give, each a dict whose `of` is the function.
SOURCE_KINDS = [paper, artifacts]

def _check_source(source):
    """`source` once checked: a list of values of `paper()` and `artifacts()`.

    The language has no `type()`, so an entry that is no dict, as a bare URL, is refused by the
    language itself, at the `.get` below."""
    if "of" in source:
        fail("`source` is one value: give a list of them, as `source = [paper(\"https://…\")]`")
    made = [entry for entry in source if entry.get("of") not in SOURCE_KINDS]
    if made:
        keys = ", ".join(["`" + key + "`" for key in sorted(made[0].keys())])
        fail("`source` holds a dict of {}, which neither `paper()` nor `artifacts()` made: write `paper(\"https://…\")` or `artifacts(\"https://…\", commit = \"…\")`"
            .format(keys or "no key"))
    return source

def _verifier(name, image, mode, dataset, needs, sessions, model_routes, model_budget, env):
    """The target `name`, of the shared verifier on `image` in `mode`, holding `sessions`.

    A body of `lib/verify.star` runs the verifier there. Its programs reach `model_routes`, under
    `model_budget` where it is given, and `env` is added to the verifier's environment."""
    target = {
        "cwd": "/pumpkin/home",
        "env": [
            "FRONTIER_VERIFICATION_MODE=" + mode,
            "FRONTIER_DATASET_PATH=" + dataset,
            "FRONTIER_PAYLOAD_PATH=" + PAYLOAD,
            "FRONTIER_RESULT_PATH=/frontier/output/result.json",
            "FRONTIER_EVIDENCE_PATH=/frontier/output/evidence/evidence.tar.zst",
        ] + env,
        "sessions": sessions,
    }
    if model_routes:
        # A route reaches one model at least, so a judge that calls none has no route.
        models = [model for entry in model_routes for model in entry["models"]]
        target = target | {"egress": {"model": _route(name, model_routes, models, model_budget)}}
    return {
        "name": name,
        "image": image,
        "user": VERIFIER_USER,
        "needs": needs,
        "runtime": {"target": target},
    }

def _route(role, entries, models, microusd):
    """The model route of `entries`, with `models` under a budget of `microusd` if it is given.

    The budget bounds what the requests of role `role` to `models` cost in all. Those models
    leave each entry for one of their own, with its API and headers and the budget; an entry left
    with no model goes. A budget bounds one entry, so it refuses `models` in entries of more than
    one API."""
    if microusd == None:
        return entries
    kept = [entry | {"models": [m for m in entry["models"] if m not in models]} for entry in entries]
    split = [
        entry | {"models": [m for m in entry["models"] if m in models], "budget": {"microusd": microusd}}
        for entry in entries
        if [m for m in entry["models"] if m in models]
    ]
    if len(split) > 1:
        fail("`model_budgets` gives `{}` a budget over models of the APIs {}: a budget bounds one entry of one API, so give the role's budgeted models of one API"
            .format(role, ", ".join(["`" + entry["api"] + "`" for entry in split])))
    return [entry for entry in kept if entry["models"]] + split

def _harness(base, needs, gpu_type, model_routes, model_budget, env):
    """The agent role of `AGENT`, a harness, on `MODEL`.

    Its image is the harness's layer on `base`. Its programs reach `model_routes` besides the
    harness's own route, under `model_budget` where it is given, which counts only their models,
    and `env` is added to its command's environment. With `gpu_type`, the type of the GPU of the
    agent's runners, `base` holds `frontier-gpu`, the agent's command also sets what it reads,
    and its prompt says to use it."""
    if MODEL == None:
        fail("`--arg model` is not given, and the agent is `{}`: give the model it works on".format(AGENT))
    entry = HARNESSES[AGENT]
    prompt = PROMPT_START + (GPU_PROMPT if gpu_type else "") + PROMPT_END
    if AGENT not in ["codex", "claude-code"]:
        prompt = "Read your grant guide with `/.pumpkin/bin/sandboxctl ops --text`.\n\n" + prompt
    harness = entry["function"](
        base,
        model = MODEL,
        effort = EFFORT or entry["effort"],
        workspace = "/workspace",
        prompt = prompt,
        finalize_grace_seconds = FINALIZE_GRACE_SECONDS,
        **entry["keywords"]
    )
    if gpu_type:
        env = env + ["FRONTIER_GPU_GRANT=" + GPU_GRANT, "FRONTIER_GPU_TYPE=" + gpu_type]
    agent = harness["runtime"]["agent"]
    if AGENT == "claude-code":
        agent = agent | {"prepend_grant_help": True}
    command = agent["command"] | {"env": agent["command"]["env"] + env}
    models = [model for entry in model_routes for model in entry["models"]]
    route = _route("agent", _routes(agent["egress"]["model"], model_routes), models, model_budget)
    return {
        "name": "agent",
        "image": harness["image"],
        "user": AGENT_USER,
        "needs": needs,
        "runtime": {"agent": agent | {"command": command, "egress": {"model": route}}},
    }

def _routes(harness_routes, model_routes):
    """`harness_routes` and `model_routes` as one route, with one entry for each API.

    Each of the harness's entries gains the models of `model_routes`' entry of its API; the
    entries of the other APIs follow."""
    apis = [entry["api"] for entry in harness_routes]
    merged = [
        entry | {"models": entry["models"] + [
            model
            for other in model_routes
            if other["api"] == entry["api"]
            for model in other["models"]
            if model not in entry["models"]
        ]}
        for entry in harness_routes
    ]
    return merged + [entry for entry in model_routes if entry["api"] not in apis]

def _session(name, runner, lease, help):
    """The session `name` on the runners of role `runner`, with `lease`'s session fields and public `help`."""
    fields = {key: value for key, value in lease.items() if key not in ["machine", "trusted"]}
    return {"name": name, "help": help, "session": {"runner": runner} | fields}

def _runner(name, image, user, machine):
    """The role `name` of leased runners on `image`.

    They run their holder's commands as `user` in `RUNNER_HOME`."""
    return {
        "name": name,
        "image": image,
        "user": user,
        "needs": machine,
        "runtime": {"runner": {"cwd": RUNNER_HOME}},
    }

def _gpu_type(agent, agent_lease):
    """The FrontierCS type of the GPU of the agent's runners, which `frontier-gpu` names, or none.

    Refuses an agent with a GPU of its own, and runners with no GPU of `GPU_TYPES`."""
    if "gpu" in agent:
        fail("`agent` has a `gpu`: the agent leases its GPU runners, so give the GPU in `agent_lease`'s `machine`")
    if agent_lease == None:
        return None
    types = [name for name, gpu in GPU_TYPES.items() if gpu == agent_lease["machine"].get("gpu")]
    if not types:
        fail("`agent_lease`'s `machine` has no GPU that an agent leases: give it a `gpu` of {}"
            .format(", ".join(GPU_TYPES.keys())))
    return types[0]

def _check_judge_lease(name, lease, timeouts):
    """Judge lease `name`, `lease`, once checked.

    Refuses it if it does not say whether it is trusted, or does not last from 1 second to the
    judge's timeout in each mode, `timeouts` by mode, as FrontierCS refuses one."""
    if lease.get("trusted") not in [True, False]:
        fail("judge lease `{}` gives no `trusted` of True or False: say whether its machine may run the judge's image"
            .format(name))
    seconds = lease["instance_seconds"]
    if sorted(seconds) != sorted(MODES):
        fail("judge lease `{}`'s `instance_seconds` gives {}: give one for each mode, {}"
            .format(name, ", ".join(sorted(seconds)), " and ".join(MODES)))
    long = [mode for mode in MODES if seconds[mode] <= 0 or seconds[mode] > timeouts[mode]]
    if long:
        fail("judge lease `{}`'s `instance_seconds` for {} is {}: give one from 1 to the judge's timeout, {}"
            .format(name, long[0], seconds[long[0]], timeouts[long[0]]))
    return lease

def _reference(image, reference_patch, noop_patch_file, deadline_seconds):
    """The agent role that submits the task's reference solution, with no route out.

    The patch is the judge's `reference.patch` if `reference_patch`, else one that appends a
    comment to `noop_patch_file`, else an empty one. The role runs until `deadline_seconds`."""
    if reference_patch:
        patch = REFERENCE_PATCH
    elif noop_patch_file:
        patch = NOOP_PATCH.format(file = noop_patch_file)
    else:
        patch = EMPTY_PATCH
    script = REFERENCE_HEAD + patch + REFERENCE_TAIL.format(seconds = deadline_seconds)
    return {
        "name": "agent",
        "image": image,
        "user": VERIFIER_USER,
        "needs": REFERENCE_NEEDS,
        "runtime": {"agent": {
            "command": {
                "argv": ["/bin/sh", "-c", script],
                "cwd": "/frontier",
            },
            "finalize_grace_seconds": FINALIZE_GRACE_SECONDS,
        }},
    }

# The slowest that a payload or a result moves between the host and a judge's sandbox, in bytes a
# second: a sandbox with a GPU is a Modal one, whose first transfer can run under 75 kB a second,
# and any other an EC2 one, which moves over 15 MB a second.
GPU_SANDBOX_BYTES_PER_SECOND = 70000
CPU_SANDBOX_BYTES_PER_SECOND = 15000000

# The most that a body gets of the verifier's result, Pumpkin's bound on a file a body gets.
RESULT_MAX_BYTES = 1 * MiB

def _moved_seconds(machine, payload_max_bytes):
    """The longest that putting a payload and getting the result take in a judge on `machine`.

    A call's timeout covers them, besides the verifier's run, which alone the source's
    `eval_timeout_seconds` bounds, so a judge's timeouts add this, in whole seconds."""
    rate = GPU_SANDBOX_BYTES_PER_SECOND if "gpu" in machine else CPU_SANDBOX_BYTES_PER_SECOND
    moved = payload_max_bytes + RESULT_MAX_BYTES
    return (moved + rate - 1) // rate

def _seconds(name, given, default):
    """`given`, the value of `--arg <name>`, in seconds, or `default` if it is not given."""
    if given == None:
        return default
    if given <= 0:
        fail("`--arg {}` is {}: give a number of seconds above 0, or none for the task's own"
            .format(name, given))
    return given

# PyTorch 2.6 with CUDA 12.4 and Triton, the kernel tasks' development environment.
PYTORCH_KERNEL_AGENT = ("//lib/pytorch-kernel-agent", {})
UBUNTU_AGENT_FAMILY = "//lib/ubuntu-agent"
UBUNTU_VERIFIER_FAMILY = "//lib/ubuntu-verifier"
UBUNTU_BASES = {UBUNTU_AGENT_FAMILY: "agent-base", UBUNTU_VERIFIER_FAMILY: "verifier-base"}

def ubuntu_agent(packages, executable = ""):
    """An Ubuntu agent with space-separated apt packages and a find -name pattern for starter executables."""
    return (UBUNTU_AGENT_FAMILY, {"APT_PACKAGES": packages, "REPO_EXECUTABLE": executable})

def ubuntu_verifier(packages):
    """The shared Ubuntu verifier stage with space-separated apt packages."""
    return (UBUNTU_VERIFIER_FAMILY, {"APT_PACKAGES": packages})

def _task_tools():
    return image("Dockerfile", context = "//lib/task-tools", mount = {"shared": "//shared"})

def _ubuntu_base(name):
    return image("Dockerfile", context = "//lib/" + name, mount = {"shared": "//shared"})

def ubuntu_agent_base(_tools):
    """The shared Ubuntu base with the repository's submission helpers."""
    return _ubuntu_base("agent-base")

def ubuntu_verifier_image(recipe, *, context = ".", mount = FETCH, target = None, contexts = {}):
    """Build a package recipe that names the shared Ubuntu verifier image through VERIFIER_BASE."""
    return image(
        recipe,
        context = context,
        mount = mount,
        target = target,
        args = {"VERIFIER_BASE": "frontier-verifier-base"},
        contexts = contexts | {"frontier-verifier-base": _ubuntu_base("verifier-base")},
    )

def package_agent(base):
    """The package's own agent Dockerfile, with `base(tools)` as its `AGENT_BASE` image."""
    return (AGENT_RECIPE, base)

def _family_image(family, contexts, sources, package_context):
    directory, args = family
    name = UBUNTU_BASES.get(directory)
    if name:
        contexts = contexts | {"frontier-" + name: _ubuntu_base(name)}
    if package_context:
        # The package already holds the sources mounted at their own names.
        sources = {at: src for at, src in sources.items() if at != src} | {"family": directory}
    return image(
        "family/Dockerfile" if package_context else "Dockerfile",
        context = "." if package_context else directory,
        mount = FETCH | sources,
        args = args | {"TASK_ID": package_name().split("/")[-1]},
        contexts = contexts,
    )

def fcs_task(
        *,
        agent,
        judge,
        train_calls,
        train_cumulative_seconds,
        judge_timeout_seconds,
        submission_max_bytes,
        source,
        display_name,
        visible_metrics: dict[str, str] = {},
        category = None,
        agent_lease = None,
        agent_model_routes = [],
        agent_env = [],
        judge_leases = {},
        judge_model_routes = [],
        judge_env = [],
        model_budgets = {},
        final_timeout_seconds = None,
        datasets = SELECTORS,
        mode_stages = True,
        contexts = {},
        context_images = {},
        agent_recipe = None,
        agent_runner_image = None,
        reference_image = None,
        verifier_image = image,
        verifier_base = None,
        package_context = False,
        reference_patch = True,
        noop_patch_file = None):
    """The workflow of the FrontierCS task in the entry file's package.

    It holds an agent, as `--arg agent` chooses; a train judge, the target of the agent's job
    function `train`; a final judge, the target of the final call, which scores the agent's final
    submission, which the agent submits by its now function `final`; with `agent_lease`, a harness
    agent's GPU runners, which it leases through `frontier-gpu`; and the runners each judge leases.
    In each judge, a body of `lib/verify.star` runs the verifier, on the image of the package's
    verifier recipe, which reads its mode's selector at `datasets[mode]`.

    A machine is a role's `needs` in Pumpkin's form: `cpu_millis` and `memory_bytes`, and, where
    the role needs them, `gpu`, `cpu_features` and `cpu_model`, as the constants above give them.
    A lease is the fields of Pumpkin's session, and `machine`, its runners' machine. Each
    value is the source's: its `task.toml` over FrontierCS's `task-defaults.toml`, or its
    `policy.json`:

    Args:
      agent: the agent's machine: `[agent]`'s CPU and memory. It has no GPU.
      judge: both judges' machine: `[verifier]`'s CPU and memory; for a judge with a GPU, the
        GPU runner's, `gpu_parent_resources.verifier` where the source states it, else
        `GPU_RUNNER`.
      train_calls: how many times the agent may call the train judge: `max_train_calls`.
      train_cumulative_seconds: the train judge's time in all: `max_train_wall_seconds`.
      judge_timeout_seconds: each judge call's timeout: `eval_timeout_seconds`, to which
        `_moved_seconds` is added.
      submission_max_bytes: the largest submission: `[submission]`'s `max_bytes`.
      visible_metrics: the public train metrics in display order. Maps each name to its
        description for the agent's Feedback section. A train call returns only the declared
        task metrics and the shared verifier metrics.
      source: where the problem comes from, for readers, as are `display_name` and `category`:
        nothing in a run reads them. A list of `paper(…)`, from `[paper]` (`policy.json`'s
        `metadata.paper`) and the authoring records, and `artifacts(…)`, the code the paper
        published; `[]` where the source names neither.
      display_name: the task's name: `display_name`.
      category: the task's category, of `CATEGORIES`: the folder that holds it, where one does.
      agent_lease: the GPU runners that a harness agent leases, from `[gpu.agent]`: the
        `machine`, of a GPU of `GPU_TYPES` (`allowed_types`) and `gpu_parent_resources.agent`'s
        size, else `GPU_RUNNER`'s; `instances` (`max_instances`), `instance_seconds`
        (`max_instance_seconds`), `cumulative_seconds` (`max_gpu_seconds`), `commands`
        (`[network.audit]`'s `ssh_session_budgets.agent`) and `transfer_bytes`
        (`max_ssh_bytes_per_direction`). The runners are a role `gpu_runner`, on the agent's
        base image unless `agent_runner_image` is given, and the agent's session `gpu` on it, which
        `frontier-gpu` drives.
      agent_model_routes: the model routes that a harness agent's own programs call, besides the
        harness's own, in Pumpkin's form, as `[{"api": "chat", "models": [M]}]`: from
        `[[additional_keys]]` whose `envs` hold `agent`, each key's `model_aliases`. The programs
        reach them at `MODEL_BASE_URL`, with any key, which the route drops. The reference agent
        calls no model.
      agent_env: `NAME=value` entries added to a harness agent's command's environment, where the
        task's programs read how to reach its models; the reference agent's command has none.
      judge_leases: maps the name of each `[test_sandboxes.<name>]` of `access = "command"` to
        its lease: `trusted` (`trust`), `machine` (`resources`; a GPU's of `GPU_RUNNER`'s size),
        `instances` (`max_concurrent`), `instance_seconds` (`lease_seconds`: one for each mode, at
        most `judge_timeout_seconds`) and `transfer_bytes`
        (`max_ssh_bytes_per_direction`). Each judge holds a session of that name, on a runner
        role of its own, `<judge>_<name>`: a trusted machine runs the judge's own image, an
        untrusted one the agent's base image, which holds no judge file.
      judge_model_routes: the model routes that both judges' programs call, as
        `agent_model_routes` gives the agent's: from `[[additional_keys]]` whose `envs` hold
        `verifier-train` and `verifier-eval`.
      judge_env: `NAME=value` entries added to both judges' environment.
      model_budgets: maps `agent`, `train_judge` or `final_judge` to the most, in millionths of a
        dollar (`15 * USD`), that the role's requests to the models of its routes above may cost
        over the run, at the prices of the run's configuration: from `[[additional_keys]]`, each
        key's `budget_usd` for the role of its `envs`. The agent's counts only the models of
        `agent_model_routes`, not the harness's own; the reference agent, which calls no model,
        has none. A budget bounds one entry of the role's route, so its models are of one API.
      final_timeout_seconds: the final judge's timeout, where it is not `judge_timeout_seconds`.
      datasets: each mode's selector, where the verifier's image holds it.
      mode_stages: whether each judge runs the verifier recipe's stage of its mode's name;
        otherwise both run its last stage.
      contexts: maps the name by which the recipes name an image to the recipe in the package
        that builds it, which every recipe is built with.
      context_images: additional named image contexts built by the package.
      agent_recipe: a shared agent recipe or a `package_agent` with its source base.
        Without one, the package's Dockerfile names its own base.
      agent_runner_image: constructs the leased GPU runner image from the solver-visible agent base.
      reference_image: constructs a trusted image with /frontier writable by the reference agent
        and its patch or starter file under /opt/frontier/task. Defaults to the train verifier image.
      verifier_image: the image constructor for the package's verifier Dockerfile.
      verifier_base: a shared verifier stage that the package recipe names `verifier-base`.
      package_context: keep the package root for a verifier with additional source trees or package-wide .dockerignore rules.
      reference_patch: whether the judge holds the task's `reference.patch`, which the reference
        agent submits; a task without one has the reference agent submit an empty patch, or,
        where its judge refuses an empty patch, a patch that appends a comment to
        `noop_patch_file`.
      noop_patch_file: a path in the starter that the task's edit scope allows, of a file that
        reads `#` as a comment.
    """
    if noop_patch_file != None:
        if reference_patch:
            fail("`noop_patch_file` is given with `reference_patch`: the reference agent submits one patch, give one of them")
        if not [x for x in NOOP_PATCH_SUFFIXES if noop_patch_file.endswith(x)]:
            fail("`noop_patch_file` is `{}`: its suffix is not one of a language that reads `#` as a comment: give a file that ends in {}"
                .format(noop_patch_file, ", ".join(NOOP_PATCH_SUFFIXES)))
        shell = [c for c in NOOP_PATCH_SHELL if c in noop_patch_file]
        if shell or [n for n in noop_patch_file.split("/") if n in ["", ".", ".."]]:
            fail("`noop_patch_file` is `{}`: give a relative path of file names, with no empty, `.` or `..` name and no character that a shell reads"
                .format(noop_patch_file))
    _check_source(source)
    _text("`display_name`", display_name, "give the source's `display_name`")
    if category != None and category not in CATEGORIES:
        fail("`category` is `{}`: give one of {}".format(category, ", ".join(["`" + c + "`" for c in CATEGORIES])))
    gpu_type = _gpu_type(agent, agent_lease)
    routes = {"agent": agent_model_routes, "train_judge": judge_model_routes, "final_judge": judge_model_routes}
    unknown = [role for role in model_budgets if role not in routes]
    if unknown:
        fail("`model_budgets` names `{}`: give a budget of `agent`, `train_judge` or `final_judge`".format(unknown[0]))
    unrouted = [role for role in model_budgets if not routes[role]]
    if unrouted:
        fail("`model_budgets` gives `{}` a budget, but its model routes are empty: give them, or remove the budget"
            .format(unrouted[0]))

    # A lease lasts at most the call of its mode: a train call's `judge_timeout_seconds`, the final's
    # `final_timeout_seconds` where it is given.
    timeouts = {"train": judge_timeout_seconds, "eval": final_timeout_seconds or judge_timeout_seconds}
    judge_leases = {name: _check_judge_lease(name, lease, timeouts) for name, lease in judge_leases.items()}
    built = {name: image(recipe, mount = FETCH) for name, recipe in contexts.items()} | context_images
    tools = _task_tools()
    built = built | {"task-tools": tools}
    recipe, directory, sources = VERIFIER_RECIPE, ".", FETCH
    verifier_contexts = built
    if verifier_base != None:
        common = _family_image(verifier_base, {"task-tools": tools}, {
            "repo": "repo",
            "datasets": "datasets",
            "judge-config": "generated/judge",
            "verifier-config": "generated/verifier",
        }, package_context)
        verifier_contexts = built | {"verifier-base": common}
        if not package_context:
            recipe, directory, sources = "Dockerfile", "verifier", {"judge": "judge"}
    images = {
        mode: verifier_image(recipe, context = directory, mount = sources, target = mode if mode_stages else None, contexts = verifier_contexts)
        for mode in MODES
    }
    deadline_seconds = _seconds("deadline_seconds", DEADLINE, WALL_TIME_SECONDS)
    gpu_roles, gpu_grants = [], []
    untrusted = [name for name, lease in judge_leases.items() if not lease["trusted"]]
    base = None
    if AGENT in HARNESSES or untrusted:
        readme = image(
            "Dockerfile",
            context = AGENT_README,
            mount = {"task": AGENT_DIR},
            args = {"PUBLIC_METRICS": json.encode(visible_metrics.items())},
        )
        agent_contexts = built | {"readme": readme}
        if agent_recipe == None or agent_recipe[0] == AGENT_RECIPE:
            args = {}
            if agent_recipe != None:
                build_base = agent_recipe[1]
                agent_contexts = agent_contexts | {"frontier-agent-base": build_base(tools)}
                args = {"AGENT_BASE": "frontier-agent-base"}
            base = image(AGENT_RECIPE, mount = FETCH, args = args, contexts = agent_contexts)
        else:
            base = _family_image(agent_recipe, agent_contexts, {
                "repo": "repo",
                "policy": "generated/agent",
            }, package_context)
    if AGENT in HARNESSES:
        agent_image = base
        if agent_lease:
            agent_image = image("Dockerfile", context = FRONTIER_GPU, contexts = {"agent-base": base})
            runner = agent_runner_image(base) if agent_runner_image else base
            gpu_roles = [_runner(GPU_RUNNER_ROLE, runner, AGENT_USER, agent_lease["machine"])]
            gpu_grants = [_session(GPU_GRANT, GPU_RUNNER_ROLE, agent_lease, "Use frontier-gpu to run your benchmarks on a leased GPU.")]
        agent_role = _harness(agent_image, agent, gpu_type, agent_model_routes, model_budgets.get("agent"), agent_env)
    elif AGENT == "reference":
        driver = reference_image() if reference_image else images["train"]
        agent_role = _reference(driver, reference_patch, noop_patch_file, deadline_seconds)
    else:
        fail("`--arg agent` is `{}`: give one of {}".format(AGENT, ", ".join(sorted(HARNESSES.keys()) + ["reference"])))
    judge_modes = [("train_judge", "train"), ("final_judge", "eval")]
    moved_seconds = _moved_seconds(judge, submission_max_bytes)
    runners = [
        _runner(
            role + "_" + name,
            images[mode] if lease["trusted"] else base,
            VERIFIER_USER if lease["trusted"] else AGENT_USER,
            lease["machine"],
        )
        for role, mode in judge_modes
        for name, lease in judge_leases.items()
    ]
    judges = [
        _verifier(
            role,
            images[mode],
            mode,
            datasets[mode],
            judge,
            [
                _session(
                    name,
                    role + "_" + name,
                    lease | {"instance_seconds": lease["instance_seconds"][mode]},
                    "Run this verification step on its declared machine.",
                )
                for name, lease in judge_leases.items()
            ],
            judge_model_routes,
            model_budgets.get(role),
            judge_env,
        )
        for role, mode in judge_modes
    ]
    grants = [
        {"name": "train", "help": TRAIN_HELP, "job": {
            "target": "train_judge",
            "body": train_call,
            "keywords": {"metrics": visible_metrics.keys() + VERIFIER_METRICS},
            "payload_max_bytes": submission_max_bytes,
            "calls": train_calls,
            "cumulative_seconds": train_cumulative_seconds,
            "call_timeout_seconds": _seconds(
                "train_call_timeout_seconds",
                TRAIN_TIMEOUT,
                judge_timeout_seconds + moved_seconds,
            ),
        }},
    ] + gpu_grants + [
        {"name": "final", "help": FINAL_HELP, "now": {"body": submit, "payload_max_bytes": submission_max_bytes}},
    ]
    agent_role = agent_role | {"runtime": {"agent": agent_role["runtime"]["agent"] | {"grants": grants}}}
    workflow(
        deadline_seconds = deadline_seconds,
        name = "frontier-cs/" + package_name().split("/")[-1],
        roles = [agent_role] + judges + gpu_roles + runners,
        final = {
            "target": "final_judge",
            "body": final_call,
            "payload_max_bytes": submission_max_bytes,
            "timeout_seconds": (final_timeout_seconds or judge_timeout_seconds) + moved_seconds,
        },
    )
