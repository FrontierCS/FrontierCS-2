"""The images of the task's GPU machines, and the lease of a judge's parents: their machine, their
image, the model's assets and their pod, its companions, each with its image."""

load("//lib/fcs.star", "FETCH", "H100", "VERIFIER_USER", "ubuntu_verifier_image")
load("//tasks/vllm_llm_serving_optimization/swe.star", "FINAL_ONLY_SWE_INSTANCES", "TRAIN_SWE_INSTANCES")

# The user each SWE container runs as, whole: the one the source runs the model's commands as
# there (`judge/serving_eval/swe_tool_exec.py`), which owns /testbed.
TOOL_USER = {"uid": 10001, "gid": 10001}

# The model's files, which each runner of the task's runtime holds read-only at /opt/frontier/model,
# where the launch helper serves it from: all 28 files of Qwen3-Coder-30B-A3B-Instruct at its
# pinned revision, the weights and the rest. Each size and SHA-256 is Hugging Face's own for the
# file at that revision: its `tree` listing's `lfs.size` and `lfs.oid` for the 16 weight files. The
# listing gives the other 12 only their Git blobs, so their SHA-256s are of the files themselves,
# each checked against its blob. That this is the revision the task's baselines ran on is unproven.
MODEL_REVISION = "https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct/resolve/b2cff646eb4bb1d68355c01b18ae02e7cf42d120/"

def _file(path, size, sha256):
    return {"path": path, "url": MODEL_REVISION + path, "size": size, "sha256": "sha256:" + sha256}

MODEL_ASSETS = [{"path": "/opt/frontier/model", "files": [
    _file(".gitattributes", 1519, "11ad7efa24975ee4b0c3c3a38ed18737f0658a5f75a0a96787b576a78a023361"),
    _file("LICENSE", 11343, "832dd9e00a68dd83b3c3fb9f5588dad7dcf337a0db50f7d9483f310cd292e92e"),
    _file("README.md", 5425, "98e8f0e00f2d6c7f1743fc4ef2b6297dcc57bec73d71f36588aed984370ac3f3"),
    _file("chat_template.jinja", 6211, "5a38bfa05833266240066aedc497decc9b00cc0d3e3b8cceea98cf530196ab06"),
    _file("config.json", 963, "e2c8d8eea39471785cd93379d8b48241ad7dcda299013155dd02526e34a0de62"),
    _file("generation_config.json", 180, "c99723ab3ba28630d26ae23def77603b540a46924895af5cf234740d3b27b51d"),
    _file("merges.txt", 1671839, "599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3"),
    _file("model-00001-of-00016.safetensors", 3998893112, "c8f4ee157c2bdb00521d0f422c08ef16f0f3f5f7e234786d829401d627d2e167"),
    _file("model-00002-of-00016.safetensors", 3999974192, "8cf85f0e5a0ea8d8cd3fd1f411175d811ca735571210873fd42bb5eec0b35406"),
    _file("model-00003-of-00016.safetensors", 3997360832, "4fa5cc233e965c8671d32a335045697fc4317c8d2939d8ca5fb9ef62d0233207"),
    _file("model-00004-of-00016.safetensors", 3999975056, "c10ff15f08abea0ff60b2fc345dcc34191bc8049d05d3bece6b081a99cbfc4c8"),
    _file("model-00005-of-00016.safetensors", 3999975400, "65d3701e2354a8cf6e5afa3f84520bf81effcdc29b0c481dbbeb089f80349100"),
    _file("model-00006-of-00016.safetensors", 3999975400, "7c832bf9e527b7b6f3184b561de05f30981bdcfdffe9c69e69064cdaa97071b5"),
    _file("model-00007-of-00016.safetensors", 3999975472, "71a710c3ff52e85d22ae3f5b10b9a4dd49821c9533460fb1d4bb21593f71b585"),
    _file("model-00008-of-00016.safetensors", 3997362064, "cd2861625e1bd8a212151445072b9d826c9754fd4618f8b83c2857fc0854b778"),
    _file("model-00009-of-00016.safetensors", 3999975408, "8335225fd61c0fe283a83463a8d9c5c19e211ea8048925b031a088f274eb9c4e"),
    _file("model-00010-of-00016.safetensors", 3999975400, "421481c42e8d85891c9d88492d7530d6b6d8b15c23a6db6a9b7438e0b6161330"),
    _file("model-00011-of-00016.safetensors", 3999975408, "d7dd447928ddec67a3ec06d5bc4fd99dde9ac5bdf0b6ab500faca5dfdfb963b4"),
    _file("model-00012-of-00016.safetensors", 3987924896, "349a0b0247d244ffef34bc23b403ce756f515242323f024a5f8028c109a798af"),
    _file("model-00013-of-00016.safetensors", 3999975088, "52e7f570649b27bc35b8cd27ac643d2158c02a5c76078c80a8afc449377b508e"),
    _file("model-00014-of-00016.safetensors", 3999975400, "0b30aed1b55b3f25fbf2c5e81784438d0a08461c9518e26e11987eb10d80ebd7"),
    _file("model-00015-of-00016.safetensors", 3999975400, "cb8b3b13c0519874ed0cbe6832ab57e51476e3821c83f0ffe4ec83a8d23fc5f4"),
    _file("model-00016-of-00016.safetensors", 1085307128, "778479fa2b8b766ba1c7bfc56a71ed9b060081e0fbc5eb4edafd19a1d6b4c116"),
    _file("model.safetensors.index.json", 1699758, "8dde190b862c7c80ec7403c6495de00c60bbaf246ed479cee4506284989c584c"),
    _file("qwen3coder_tool_parser.py", 31613, "f1ef95f6cf76354b6d7f5bc19e65976ae315981315c22f07874c9eff9720de84"),
    _file("tokenizer.json", 7032399, "19564a48c4f71a2a1b937cce34c737a1e662b171c5f5d7edf641a15cd896f07d"),
    _file("tokenizer_config.json", 13055, "60f6e8cb15c98dd07300a3cc465ea662de245d2095e4245616af21b2324db3fc"),
    _file("vocab.json", 2776833, "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910"),
]}]

def _runtime():
    return image("Dockerfile", context = "runtime", mount = FETCH)

def gpu_runner_image(_agent_base):
    """The agent's H100 runner: the task's runtime with the agent's user, and none of the agent's files, as the source's.

    Args:
      _agent_base: the agent's base image, which the runner holds nothing of.

    Returns:
      The runner's image.
    """
    return image("Dockerfile", context = "gpu", contexts = {"runtime": _runtime()})

def _parent_image():
    """The judges' H100 parent, where the candidate's vLLM server runs."""
    return image("Dockerfile", context = "parent", mount = {"judge": "judge"}, contexts = {"runtime": _runtime()})

def _swe(instance, role):
    """The container `swe-<slot>-<role>` of `instance`, on its image of that target."""
    built = image(
        "Dockerfile",
        context = "swe",
        mount = {"judge": "judge"},
        target = role,
        args = {"TEST_SANDBOX_BASE": instance["base"], "TEST_SANDBOX_INSTANCE": instance["id"]} | instance.get("source", {}),
    )
    return {"name": "swe-{}-{}".format(instance["slot"], role), "image": built, "user": TOOL_USER, "cwd": "/testbed"}

def _pod(workload, mode):
    """The pod of a judge's parent of `workload` in `mode`, as the fields of Pumpkin's runner.

    The trusted scheduler comes first, then, for SWE, each instance's interactive container and
    then each instance's check container: the order of `judge/sidecar_catalog.py`'s
    `expected_sidecars`. Each container reaches the others by its name. Only the scheduler
    connects: to the parent's server, and to each SWE container's service, which refuses a
    caller without the token that the Verifier gives the scheduler (`verifier/pod.py`). Train
    has its own instances, the final those and its own."""
    scheduler = {
        "name": "scheduler",
        "image": ubuntu_verifier_image("Dockerfile", context = "scheduler", mount = FETCH | {"judge": "judge"}),
        "user": VERIFIER_USER,
        "cwd": "/var/tmp",
    }
    if workload == "bfcl":
        return {"companions": [scheduler]}
    instances = TRAIN_SWE_INSTANCES if mode == "train" else TRAIN_SWE_INSTANCES + FINAL_ONLY_SWE_INSTANCES
    containers = [_swe(instance, "candidate") for instance in instances] + [_swe(instance, "check") for instance in instances]
    return {"companions": [scheduler] + containers}

def parents(workload):
    """The lease of the judges' parents of `workload`, for `fcs_task`'s `judge_leases`.

    A parent serves one load group: its server shares the machine with the scheduler and the
    test sandboxes of its pod (`[gpu_parent_resources.verifier]`). The final has two groups a
    workload, of which a train call runs one, and a parent lasts its phase's `gpu_seconds`
    (`verifier/runtime-plan.json`). The candidate's code runs there, so the lease is untrusted.

    Args:
      workload: `swe` or `bfcl`, which the judge's session is named after.

    Returns:
      The lease.
    """
    return {
        "trusted": False,
        "machine": {"cpu_millis": 16000, "memory_bytes": 64 * GiB, "gpu": H100},
        "instances": 2,
        "instance_seconds": {"train": 5400, "eval": 7200},
        "transfer_bytes": 256 * MiB,
        "image": _parent_image(),
        "assets": MODEL_ASSETS,
        "pod": {"train": _pod(workload, "train"), "eval": _pod(workload, "eval")},
    }
