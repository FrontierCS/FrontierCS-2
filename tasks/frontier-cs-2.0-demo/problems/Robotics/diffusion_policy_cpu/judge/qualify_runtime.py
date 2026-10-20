"""Non-inference runtime/process check inside the standard platform Verifier."""

import json
import os
from pathlib import Path
import subprocess
import tempfile

from private_context import file_sha256, hardware_identity, runtime_identity, verify_cgroup
from runtime_layout import validate_layout
from sampler_bridge import worker_command, worker_environment


def main():
    import yaml
    task = Path(__file__).resolve().parent
    config = yaml.safe_load((task / "config.yaml").read_text())
    declaration = config["runtime"]["measurement"]
    assert os.getuid() == os.getgid() == 65532
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:2])
    verify_cgroup({"cpus": 2, "memory_bytes": 4 * 1024**3})
    forbidden = Path("/qualification-root-must-be-readonly")
    try:
        forbidden.write_text("unexpected writable root")
    except OSError:
        pass
    else:
        forbidden.unlink()
        raise AssertionError("Verifier root filesystem is writable")
    identity = runtime_identity(declaration["prefix"])
    assert identity["python"].startswith("3.10.20 ")
    assert dict(identity["distributions"])["torch"] == "2.2.2+cpu"
    assert file_sha256(declaration["checkpoint"]) == declaration["checkpoint_sha256"]
    validate_layout(declaration, declaration)
    with tempfile.TemporaryDirectory(prefix="runtime-qualification-", dir="/frontier") as temporary:
        root = Path(temporary)
        app = root / "public"
        app.mkdir()
        (app / "public.txt").write_text("public")
        worker = root / "probe.py"
        worker.write_text('''import json, os
from pathlib import Path
assert os.getpid() != PARENT_PID
assert Path('public.txt').read_text() == 'public'
assert 'QUALIFICATION_PRIVATE_CANARY' not in os.environ
assert 'FRONTIER_VERIFIER_TOKEN' not in os.environ
print(json.dumps({'separate_process': True, 'public_cwd': True, 'environment_scrubbed': True}))
'''.replace("PARENT_PID", str(os.getpid())))
        command = worker_command(app, worker, task / "bridge_protocol.py", runtime={
            "prefix": declaration["prefix"], "base": identity["base_prefix"], "version": "3.10"})
        result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=30,
                                cwd=app, close_fds=True,
                                env=worker_environment(app, task / "bridge_protocol.py"))
        checks = json.loads(result.stdout)
    import sys
    assert "torch" not in sys.modules
    print(json.dumps({"runtime_identity": identity, "hardware": hardware_identity(), "process": checks,
                      "execution_policy": "standard_platform_sandbox_with_post_trial_review",
                      "uid": os.getuid(), "inference": False}, sort_keys=True))


if __name__ == "__main__":
    main()
