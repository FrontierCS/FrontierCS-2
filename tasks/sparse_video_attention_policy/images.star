"""Task images for GPU workloads and the CPU reference agent."""

load("//lib/fcs.star", "FETCH")

def _runtime():
    return image("Dockerfile", context = "runtime", mount = FETCH)

def gpu_runner_image(agent_base):
    return image(
        "Dockerfile",
        context = "gpu",
        mount = {"pins": "pins"},
        contexts = {"runtime": _runtime(), "agent-base": agent_base},
    )

def gpu_verifier_image(recipe, *, context, mount, target, contexts):
    return image(
        recipe,
        context = context,
        mount = mount,
        target = target,
        contexts = contexts | {"runtime": _runtime()},
    )

def reference_image():
    return image("Dockerfile", context = "reference", mount = {"judge": "judge"})
