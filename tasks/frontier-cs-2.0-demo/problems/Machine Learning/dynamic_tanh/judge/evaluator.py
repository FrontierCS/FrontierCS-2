"""GPU evaluator for the model-wide LayerNorm replacement task."""
from __future__ import annotations

import hashlib, importlib.util, json, math, os, re, shutil, statistics, subprocess, sys, tempfile, types
from contextlib import contextmanager
from pathlib import Path
from typing import Any
import candidate_failure
from submission_patch import _make_owner_writable, apply_submission_patch

TASK_DIR = Path(__file__).resolve().parent
APP_DIR = TASK_DIR / "harbor" / "app"
REFERENCE_PATCH = TASK_DIR / "reference.patch"
ASSET_ROOT = TASK_DIR / "judge_assets" / "flower_photos"
MANIFEST_PATH = ASSET_ROOT / "manifest.json"
CLASSES = ("daisy", "dandelion", "roses", "sunflowers", "tulips")
EPOCHS, WARMUP_BLOCKS, TIMED_BLOCKS, SUBPROCESS_TIMEOUT = 2, 3, 11, 180
# Timed-path validation. Every warmup and timed repeat of both paired protocols
# uses new input values, the same for candidate and reference within a repeat,
# and every candidate call is compared with a trusted oracle (the reference
# implementation holding the candidate's trained state) run through the same
# block function on that repeat's values.
# Tolerance for outputs and loss computed under fp16 autocast, as (rtol, atol)
# per tensor: ||got - want||_2 <= rtol * ||want||_2 + atol * sqrt(numel).
TIMED_OUTPUT_TOLERANCE = (1e-2, 1e-4)
# Gradients are not required to be close to the reference's own fp16 rounding:
# on the trained states the fp16 backward signal is near half-precision
# resolution and two honest arrangements can differ by tens of percent. Each
# repeat instead yields four trusted gradients of the same state and inputs:
# the reference under autocast, the reference with every replacement site in
# fp16 in two forms (separate operations and one fused multiply-add), and an
# fp32 gradient. The candidate must be near one of them, as near as they are
# to each other:
#   distance(candidate) <= floor + TIMED_GRADIENT_FACTOR * spread,
# where distance is the smallest ||candidate - trusted||_2 and spread the
# largest ||trusted - trusted'||_2, both divided by ||fp32||_2, for all
# parameters jointly and for each group of TIMED_GRADIENT_FLOORS. Three fp16
# arrangements rather than two: on the L4 a single pair underestimated the
# fp16 noise of one repeat by a factor of two. Factor and floors are the
# values fixed before the L4 calibration and confirmed by it
# (task_tests/dynamic_tanh/calibrate_timed_path.py); rule, measurements and
# decision record: docs/task-authoring/dynamic_tanh/timed-path-validation.md.
TIMED_GRADIENT_FACTOR = 3.0
TIMED_GRADIENT_FLOORS = {"joint": 0.02, "other": 0.02, "alpha": 0.10, "scale": 0.05, "shift": 0.05}
# A repeat has detection power when its joint bound stays below this distance:
# a zeroed gradient is about 1 away, a halved one 0.5.
TIMED_GRADIENT_POWER_BOUND = 0.5
# The trusted fp16 gradients of one repeat, by their key in the oracle's result.
TIMED_TRUSTED_FP16 = ("gradients", "half_gradients", "fused_gradients")
REFERENCE_PATCH_SHA256 = "2da4be0b556d65bf7ad472de3709c6a5cc467ea2e762a23f9ab4e6b6898af2ad"
REFERENCE_DYNAMIC_SHA256 = "609a1f4412d1d570c74d4deb725436fda3322a3bbb05c1d2c6a0de6b7b6c5039"
RETAINED_SHA256 = {
    "main.py": "d8741525a66575718ffe4172dc0be1abdcacf7b2d9a915454069e76014876fc6",
    "engine.py": "ca4313d4cae84e5fa7e978e70fae0f53b5bf7d0d8735a36b642e177da6bd1384",
    "datasets.py": "6fe0cd7b3909f7fb602e86c464d2f8c047bf293bf2957f8a98b44ebb26cc9dd8",
    "optim_factory.py": "93f3223335c825fed6c98fc7c5fb42525c58ee22741c76238fb8f801cf3f7225",
    "utils.py": "4b189315f67e25f612b9e7a6df6fb20111f96ca4a3ac989b6eeb1ab4264c40c9",
}

# Process and harness introspection (Preview issue #647). The accuracy run
# (frontier-bench.sh -> main.py) and the evaluator's timed and validated calls
# must execute the same implementation; submitted code may not inspect which
# process, entry point, caller or environment it runs in. Every file the
# candidate owns is scanned after the patch is applied, comments included.
INTROSPECTION_PATTERN = re.compile(
    r"argv|__main__|\binspect\b|_getframe|currentframe|f_back|f_globals|f_locals|f_code"
    r"|\btraceback\b|getpid|getppid|/proc|environ|getenv|sys\.modules|cmdline|program_invocation",
    re.IGNORECASE)
INTROSPECTION_MESSAGE = "patch inspects the process, entry point, caller or environment"

def _introspection_violation(tree: Path) -> bool:
    owned = [tree / "dynamic_tanh.py"]
    extensions = tree / "solver_extensions"
    if extensions.is_dir():
        owned += sorted(path for path in extensions.rglob("*") if path.is_file())
    for path in owned:
        if path.is_file() and INTROSPECTION_PATTERN.search(path.read_bytes().decode("utf-8", errors="replace")):
            return True
    return False

@contextmanager
def _training_argv(command):
    """Candidate calls in this process see the argv of the authentic training process."""
    saved = sys.argv
    sys.argv = ["main.py", *command[1:]]
    try:
        yield
    finally:
        sys.argv = saved

class InfrastructureError(RuntimeError):
    """A trusted task prerequisite or reference execution failed."""

class CandidateError(RuntimeError):
    """The submitted implementation could not satisfy the task contract."""

# The only facts a timed-path rejection may add to the public message and to
# the solver-visible diagnostic: which block and which quantity failed first.
# Fixed strings from candidate_failure.py, one pair per rejected submission;
# no value, threshold, workload or repeat.
TIMED_PATH_PROTOCOLS = candidate_failure.BLOCKS
TIMED_PATH_QUANTITIES = candidate_failure.QUANTITIES
CANDIDATE_FAILURE_MESSAGE = "submission failed validation or authentic model execution"

class TimedPathRejection(CandidateError):
    """A candidate result under the timed conditions failed one named comparison."""
    def __init__(self, quantity, text):
        super().__init__(text)
        self.quantity, self.protocol = quantity, None

def _candidate_message(error):
    # Table lookups only: nothing from the exception, which may carry
    # submission-controlled text, is formatted into the message.
    category = _timed_path_category(error)
    if category is not None:
        return f"{CANDIDATE_FAILURE_MESSAGE}: timed-path validation, {TIMED_PATH_PROTOCOLS[category[0]]}, {TIMED_PATH_QUANTITIES[category[1]]}"
    return CANDIDATE_FAILURE_MESSAGE

def _timed_path_category(error):
    """(block, quantity) of a properly tagged timed-path rejection, else None."""
    if type(error) is TimedPathRejection:
        protocol, quantity = error.__dict__.get("protocol"), error.__dict__.get("quantity")
        if type(protocol) is str and type(quantity) is str and protocol in TIMED_PATH_PROTOCOLS and quantity in TIMED_PATH_QUANTITIES:
            return protocol, quantity
    return None

def _candidate_diagnostic(error, stage):
    """The solver-visible rejection class of a development submission, or None."""
    category = _timed_path_category(error)
    if category is not None: return candidate_failure.diagnostic("timed_path", *category)
    if stage in ("patch_policy", "semantic_validation", "model_workload"): return candidate_failure.diagnostic(stage)
    return None

def _development():
    return os.environ.get("FRONTIER_SUBMISSION_ROLE") != "final"

@contextmanager
def _trusted_execution(phase):
    try:
        yield
    except InfrastructureError:
        raise
    except Exception as error:
        raise InfrastructureError(phase) from error

@contextmanager
def _candidate_execution():
    try:
        yield
    except (InfrastructureError, CandidateError):
        raise
    except Exception as error:
        raise CandidateError(CANDIDATE_FAILURE_MESSAGE) from error

def _invalid(reason: str):
    return 0.0, 0.0, reason[:180], {}

def _private_candidate_failure(error, stage):
    # Fixed labels only: exception text, class names and process output can
    # contain submission-controlled data. Do not format or traverse the cause.
    cause = BaseException.__cause__.__get__(error)
    kinds = ((type(None), "none"), (RuntimeError, "runtime_error"),
             (ValueError, "value_error"), (TypeError, "type_error"),
             (AssertionError, "assertion_error"), (ImportError, "import_error"),
             (ModuleNotFoundError, "module_not_found"), (OSError, "os_error"),
             (subprocess.TimeoutExpired, "subprocess_timeout"))
    kind = next((label for cls, label in kinds if type(cause) is cls), "other")
    return {"schema": "dyt.private-candidate-failure/v1",
            "stage": stage if stage in ("semantic_validation", "model_workload") else "unknown",
            "cause_kind": kind}

def _config():
    return {"submission": {"kind": "patch", "path": "/app/solution.patch",
            "allow_paths": ["dynamic_tanh.py", "solver_extensions/"],
            "max_changed_files": 32, "max_patch_bytes": 262144}}

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def _verify_sources() -> None:
    if _sha256(REFERENCE_PATCH) != REFERENCE_PATCH_SHA256:
        raise RuntimeError("trusted reference source lock failed")
    for relative, expected in RETAINED_SHA256.items():
        if _sha256(APP_DIR / relative) != expected:
            raise RuntimeError("retained upstream source lock failed")

def _load_manifest():
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    workloads, files = manifest.get("workloads"), manifest.get("files")
    if not isinstance(workloads, list) or len(workloads) != 6 or not isinstance(files, list):
        raise RuntimeError("flower workload manifest is malformed")
    ids, roles = [x.get("id") for x in workloads], [x.get("role") for x in workloads]
    if len(set(ids)) != 6 or roles.count("development") != 3 or roles.count("final") != 3:
        raise RuntimeError("flower workload partition is malformed")
    paths, hashes, counts = set(), set(), {}
    for row in files:
        relative, expected = row.get("asset_path"), row.get("sha256")
        if not isinstance(relative, str) or not isinstance(expected, str):
            raise RuntimeError("flower asset record is malformed")
        path = ASSET_ROOT / relative
        if not path.is_file() or _sha256(path) != expected:
            raise RuntimeError("flower asset source lock failed")
        if relative in paths or expected in hashes:
            raise RuntimeError("flower workloads reuse an original image")
        paths.add(relative); hashes.add(expected)
        key = (str(row.get("workload")), str(row.get("split")), str(row.get("class")))
        counts[key] = counts.get(key, 0) + 1
    for workload in workloads:
        for class_name in CLASSES:
            if counts.get((workload["id"], "train", class_name)) != 4 or counts.get((workload["id"], "val", class_name)) != 2:
                raise RuntimeError("flower shards are not class-stratified")
    return manifest, workloads

def _selected_workloads(workloads):
    role = "final" if os.environ.get("FRONTIER_SUBMISSION_ROLE") == "final" else "development"
    selected = [row for row in workloads if row["role"] == role]
    if len(selected) != 3: raise RuntimeError("evaluation suite is incomplete")
    return selected

def _reference_tree(root: Path) -> Path:
    destination = root / "reference_repo"
    shutil.copytree(APP_DIR, destination)
    # Release substrates are deliberately 0555/0444. Only this disposable
    # judge-side copy is made writable; the pristine public tree is untouched.
    _make_owner_writable(destination)
    proc = subprocess.run(["git", "apply", "--whitespace=nowarn", str(REFERENCE_PATCH)], cwd=destination,
                          text=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=20)
    if proc.returncode: raise RuntimeError("trusted reference patch did not apply")
    if _sha256(destination / "dynamic_tanh.py") != REFERENCE_DYNAMIC_SHA256:
        raise RuntimeError("trusted reference does not restore upstream bytes")
    return destination

@contextmanager
def _candidate_imports(repo: Path):
    # Keep lazy constructor/forward/backward imports available for the full
    # evaluation, while isolating helper caches between submissions.
    old_path = sys.path[:]
    def owned_module(name):
        return name == "solver_extensions" or name.startswith(("solver_extensions.", "frontier_mapping_"))

    prior_modules = {name: module for name, module in sys.modules.items() if owned_module(name)}
    for name in prior_modules:
        sys.modules.pop(name, None)
    try:
        sys.path.insert(0, str(repo.resolve()))
        yield
    finally:
        sys.path[:] = old_path
        for name in list(sys.modules):
            if owned_module(name):
                sys.modules.pop(name, None)
        sys.modules.update(prior_modules)

def _converter(repo: Path, label: str):
    spec = importlib.util.spec_from_file_location(f"frontier_mapping_{label}", repo / "dynamic_tanh.py")
    if spec is None or spec.loader is None: raise RuntimeError("replacement module cannot be imported")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    converter = getattr(module, "convert_ln_to_dyt", None)
    if not callable(converter): raise RuntimeError("conversion hook is not callable")
    return converter

@_trusted_execution("base model construction failed")
def _new_model(name: str):
    from timm.models import create_model
    return create_model(name, pretrained=False, num_classes=len(CLASSES), global_pool="avg", drop_path_rate=0.0)

@_trusted_execution("trusted validation data loading failed")
def _real_batch(workload, count=2):
    import torch
    from PIL import Image
    from torchvision import transforms
    transform = transforms.Compose([transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize((.485, .456, .406), (.229, .224, .225))])
    rows = []
    root = ASSET_ROOT / "images" / workload["id"] / "val"
    for class_index, class_name in enumerate(CLASSES):
        for path in sorted((root / class_name).glob("*.jpg")):
            with Image.open(path) as image: rows.append((transform(image.convert("RGB")), class_index))
            if len(rows) == count:
                return torch.stack([x[0] for x in rows]).cuda(), torch.tensor([x[1] for x in rows], device="cuda")
    raise RuntimeError("validation images are missing")

def _validate_semantics(candidate_repo, reference_repo, workloads):
    import torch
    import torch.nn as nn
    coverage = {}
    for index, model_name in enumerate(sorted({x["model"] for x in workloads})):
        seed = 7000 + index
        torch.manual_seed(seed); base_reference = _new_model(model_name)
        original_count = sum(isinstance(m, nn.LayerNorm) for m in base_reference.modules())
        torch.manual_seed(seed); base_candidate = _new_model(model_name)
        reference = _converter(reference_repo, f"ref_{index}")(base_reference)
        if original_count < 2:
            raise InfrastructureError("trusted model has insufficient normalization sites")
        state = reference.state_dict()
        alpha_names = [name for name in state if name.endswith(".alpha")]
        if len(alpha_names) != original_count:
            raise InfrastructureError("trusted reference conversion is incomplete")
        guarded = alpha_names + [name for name in state if name.endswith((".weight", ".bias")) and name.rsplit(".", 1)[0] + ".alpha" in state]
        with _candidate_execution():
            candidate = _converter(candidate_repo, f"cand_{index}")(base_candidate)
            if any(isinstance(m, nn.LayerNorm) for m in candidate.modules()):
                raise CandidateError("conversion coverage is incomplete")
            if set(candidate.state_dict()) != set(state): raise CandidateError("replacement state is incompatible")
            candidate.load_state_dict(state, strict=True)
            params = dict(candidate.named_parameters())
            if len({params[x].data_ptr() for x in alpha_names}) != len(alpha_names):
                raise CandidateError("replacement parameters are incomplete or shared")
        reference = reference.cuda().eval()
        with _candidate_execution():
            candidate = candidate.cuda().eval()
        images, _ = _real_batch(next(x for x in workloads if x["model"] == model_name))
        ri, ci = images.detach().clone().requires_grad_(True), images.detach().clone().requires_grad_(True)
        ro = reference(ri)
        if not torch.isfinite(ro).all(): raise InfrastructureError("trusted reference output is nonfinite")
        with _candidate_execution():
            co = candidate(ci)
            if not torch.isfinite(co).all() or not torch.allclose(co, ro, rtol=3e-4, atol=3e-5):
                raise CandidateError("model forward output does not match the reference")
        ro.square().mean().backward()
        rp = dict(reference.named_parameters())
        if ri.grad is None or not torch.isfinite(ri.grad).all():
            raise InfrastructureError("trusted reference input gradient is invalid")
        for name in guarded:
            if rp[name].grad is None or not torch.isfinite(rp[name].grad).all():
                raise InfrastructureError("trusted reference parameter gradient is invalid")
        with _candidate_execution():
            co.square().mean().backward()
            if not torch.allclose(ci.grad, ri.grad, rtol=2e-3, atol=2e-5):
                raise CandidateError("model input gradient does not match the reference")
            cp = dict(candidate.named_parameters())
            before = {name: cp[name].detach().clone() for name in guarded}
            reference_before = {name: rp[name].detach().clone() for name in guarded}
            for name in guarded:
                if cp[name].grad is None or not torch.isfinite(cp[name].grad).all() or not torch.allclose(cp[name].grad, rp[name].grad, rtol=3e-3, atol=3e-5):
                    raise CandidateError("replacement parameter gradient does not match the reference")
            torch.optim.SGD(candidate.parameters(), lr=1e-3).step()
        torch.optim.SGD(reference.parameters(), lr=1e-3).step()
        with _candidate_execution():
            for name in guarded:
                # A valid SGD step can round to no change; require movement only
                # when the same reference tensor has a representable update.
                if (not torch.equal(reference_before[name], rp[name]) and torch.equal(before[name], cp[name])) or not torch.allclose(cp[name], rp[name], rtol=3e-3, atol=3e-5):
                    raise CandidateError("replacement parameter update does not match the reference")
        coverage[model_name] = original_count
        del reference, candidate, images, ri, ci; torch.cuda.empty_cache()
    return {"converted_sites": coverage}

def _train_command(repo, workload, output):
    root = ASSET_ROOT / "images" / workload["id"]
    # timm disables auto-augmentation for an empty policy string.
    return [str(repo / "frontier-bench.sh"), "--model", workload["model"], "--drop_path", "0",
        "--batch_size", str(workload["batch_size"]), "--epochs", str(EPOCHS), "--lr", "0.004",
        "--min_lr", "0.000001", "--warmup_epochs", "0", "--weight_decay", "0.05",
        "--data_set", "image_folder", "--data_path", str(root / "train"), "--eval_data_path", str(root / "val"),
        "--nb_classes", "5", "--dynamic_tanh", "true", "--device", "cuda", "--num_workers", "2",
        "--pin_mem", "true", "--dist_eval", "false", "--use_amp", "true", "--mixup", "0", "--cutmix", "0",
        "--smoothing", "0", "--aa", "", "--reprob", "0", "--model_ema", "false",
        "--save_ckpt", "true", "--save_ckpt_num", "1", "--seed", str(workload["seed"]), "--output_dir", str(output)]

def _train(repo, workload, output):
    with _trusted_execution("training output workspace failed"):
        output.mkdir(parents=True, exist_ok=True)
        command = _train_command(repo, workload, output)
    env = os.environ.copy(); env.update({"FRONTIER_REPO_ROOT": str(repo), "PYTHONHASHSEED": str(workload["seed"]), "CUBLAS_WORKSPACE_CONFIG": ":4096:8"})
    try:
        proc = subprocess.run(command, cwd=repo, env=env, text=True,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=SUBPROCESS_TIMEOUT)
    except OSError as error:
        raise InfrastructureError("trusted training process could not start") from error
    except subprocess.TimeoutExpired as error:
        tail = error.stdout or ""
        if isinstance(tail, bytes): tail = tail.decode("utf-8", errors="replace")
        raise RuntimeError("authentic training command timed out\n" + tail[-6000:]) from error
    if proc.returncode: raise RuntimeError("authentic training command failed\n" + proc.stdout[-6000:])
    matches = re.findall(r"Accuracy of the model on the \d+ test images: ([0-9.]+)%", proc.stdout[-30000:])
    lines = (output / "log.txt").read_text(encoding="utf-8").strip().splitlines() if (output / "log.txt").is_file() else []
    if not matches or not lines: raise RuntimeError("training metrics are missing")
    loss = float(json.loads(lines[-1])["train_loss"])
    checkpoint = output / f"checkpoint-{EPOCHS - 1}.pth"
    if not math.isfinite(loss) or not checkpoint.is_file(): raise RuntimeError("training loss or checkpoint is invalid")
    return float(matches[-1]), loss, checkpoint

def _load_trained_model(repo, checkpoint, workload, label):
    import torch
    model = _converter(repo, label)(_new_model(workload["model"]))
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)["model"]
    model.load_state_dict(state, strict=True)
    names = [x for x in state if x.endswith((".alpha", ".weight", ".bias")) and x.rsplit(".", 1)[0] + ".alpha" in state]
    if not names or any(not torch.isfinite(state[x]).all() for x in names): raise RuntimeError("trained replacement state is invalid")
    changed = any(x.endswith(".alpha") and not torch.allclose(state[x], torch.full_like(state[x], .5)) for x in names)
    changed &= any(x.endswith(".weight") and not torch.allclose(state[x], torch.ones_like(state[x])) for x in names)
    changed &= any(x.endswith(".bias") and not torch.allclose(state[x], torch.zeros_like(state[x])) for x in names)
    if not changed: raise RuntimeError("training did not update learned replacement parameters")
    return model.cuda()

def _timed_block(model, batches):
    import torch
    import torch.nn.functional as functional
    outputs, losses = [], []
    model.zero_grad(set_to_none=True); torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True); start.record()
    for images, targets in batches:
        with torch.autocast(device_type="cuda", dtype=torch.float16): logits = model(images); loss = functional.cross_entropy(logits, targets)
        loss.backward()
        # Keeping the two handles launches no GPU work inside the window.
        outputs.append(logits); losses.append(loss)
    end.record(); torch.cuda.synchronize(); value = float(start.elapsed_time(end))
    if not math.isfinite(value) or value <= 0: raise RuntimeError("CUDA timing is invalid")
    observed = {"outputs": [x.detach() for x in outputs], "losses": [x.detach() for x in losses],
                "gradients": {name: parameter.grad for name, parameter in model.named_parameters()}}
    return value, observed

def _inference_block(model, batches):
    """Beat spec v3: forward-only inference latency (the paper's Table 7 axis)."""
    import torch
    outputs = []
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True); start.record()
    with torch.no_grad():
        for images, _ in batches:
            with torch.autocast(device_type="cuda", dtype=torch.float16): outputs.append(model(images))
    end.record(); torch.cuda.synchronize(); value = float(start.elapsed_time(end))
    if not math.isfinite(value) or value <= 0: raise RuntimeError("CUDA timing is invalid")
    return value, {"outputs": outputs, "losses": [], "gradients": {}}

def _norms(torch, got, want, masked):
    """L2 norms of (got - want) and of want for one tensor.

    The masked form is used when the plain one is nonfinite: positions where
    the oracle is nonfinite must match its kind (NaN or signed infinity)
    in the candidate and are otherwise ignored, so the oracle always matches itself.
    """
    if not torch.is_tensor(got) or tuple(got.shape) != tuple(want.shape):
        raise TimedPathRejection("shape", "timed-path result is missing or has the wrong shape")
    got, want = got.detach().float(), want.detach().float()
    difference = got - want
    if masked:
        finite, zero = torch.isfinite(want), torch.zeros_like(want)
        difference = torch.where(finite, difference, zero)
        same_value_kind = (torch.isfinite(got) & finite) | (torch.isnan(got) & torch.isnan(want)) | (torch.isinf(got) & torch.isinf(want) & (got == want))
        difference = torch.where(same_value_kind, difference, torch.full_like(want, math.inf))
        want = torch.where(finite, want, zero)
    return difference.norm(), want.norm()

def _deviation(pairs, tolerance, joint_tolerance=None):
    """Largest tolerance-normalised deviation over (got, want) pairs; above 1 fails.

    Returns the per-tensor maximum and, when requested, the deviation of all
    pairs taken as one vector. One host read, after the timing window closed.
    """
    import torch
    if not pairs: return 0.0, 0.0
    sizes = [max(1, want.numel()) for _, want in pairs]
    for masked in (False, True):
        norms = [_norms(torch, got, want, masked) for got, want in pairs]
        difference, scale = torch.stack([x[0] for x in norms]), torch.stack([x[1] for x in norms])
        each = (difference / (tolerance[0] * scale + tolerance[1] * scale.new_tensor(sizes).sqrt())).max()
        joint = torch.zeros_like(each) if joint_tolerance is None else difference.norm() / (
            joint_tolerance[0] * scale.norm() + joint_tolerance[1] * math.sqrt(sum(sizes)))
        each, joint = (float(x) for x in torch.stack([each, joint]))
        if math.isfinite(each) and math.isfinite(joint): break
    return each, joint

def _half_site_forward(self, x):
    """The replacement mapping with every operand and operation in half precision."""
    import torch
    weight, bias = self.weight.half(), self.bias.half()
    if not self.channels_last: weight, bias = weight[:, None, None], bias[:, None, None]
    return (torch.tanh(self.alpha.half() * x.half()) * weight + bias).to(x.dtype)

def _fused_half_site_forward(self, x):
    """The same mapping in half precision as one fused multiply-add: a third honest fp16 arrangement."""
    import torch
    weight, bias = self.weight.half(), self.bias.half()
    if not self.channels_last: weight, bias = weight[:, None, None], bias[:, None, None]
    return torch.addcmul(bias, torch.tanh(x.half() * self.alpha.half()), weight).to(x.dtype)

@contextmanager
def _half_precision_sites(model, forward=_half_site_forward):
    """Run the trusted model with all-fp16 replacement sites: another honest fp16 arrangement."""
    sites = [module for module in model.modules() if hasattr(module, "alpha") and hasattr(module, "channels_last")]
    if not sites: raise InfrastructureError("trusted model has no replacement sites")
    try:
        for module in sites: module.forward = types.MethodType(forward, module)
        yield
    finally:
        for module in sites: vars(module).pop("forward", None)

def _truth_gradients(model, batches):
    """fp32 gradients of the trusted model, without autocast: the reference point for fp16 error."""
    import torch.nn.functional as functional
    model.zero_grad(set_to_none=True)
    for images, targets in batches: functional.cross_entropy(model(images), targets).backward()
    return {name: parameter.grad for name, parameter in model.named_parameters()}

def _oracle_block(block, oracle, data):
    """What the trusted implementation produces for one repeat, with the gradient yardsticks."""
    def clones(): return [(images.clone(), targets.clone()) for images, targets in data]
    expected = block(oracle, clones())[1]
    if expected["gradients"]:
        with _half_precision_sites(oracle): expected["half_gradients"] = block(oracle, clones())[1]["gradients"]
        with _half_precision_sites(oracle, _fused_half_site_forward):
            expected["fused_gradients"] = block(oracle, clones())[1]["gradients"]
        expected["truth_gradients"] = _truth_gradients(oracle, clones())
        oracle.zero_grad(set_to_none=True)
    return expected

def _gradient_group(name, names):
    """alpha, scale and shift of the replacement sites; every other parameter is 'other'."""
    prefix, _, leaf = name.rpartition(".")
    if prefix + ".alpha" not in names: return "other"
    return {"alpha": "alpha", "weight": "scale", "bias": "shift"}.get(leaf, "other")

def _gradient_errors(observed, expected):
    """Per group: (candidate distance, spread of the trusted gradients), relative to the fp32 norm.

    The distance is to the nearest trusted gradient (the fp16 arrangements of
    TIMED_TRUSTED_FP16 and fp32); the spread is the largest distance between
    two of them. When anything is nonfinite the sums are repeated entry by
    entry: only entries finite in every trusted gradient are compared
    numerically; where all trusted fp16 gradients are nonfinite of the same
    kind (NaN, or infinity of one sign) the candidate must be of that kind;
    elsewhere it is unconstrained. A group without a finite positive fp32
    norm is not judged. Host reads happen after the timing window closed.
    """
    import torch
    truth = {name: value for name, value in expected["truth_gradients"].items() if value is not None}
    members = [_gradient_group(name, truth) for name in truth]
    count = len(TIMED_TRUSTED_FP16) + 1
    def same_kind(a, b): return (torch.isnan(a) & torch.isnan(b)) | (torch.isinf(a) & torch.isinf(b) & (a == b))
    def total(value): return value.square().sum(dtype=torch.float64)
    def sums(name, masked):
        got = observed.get(name)
        if not torch.is_tensor(got) or tuple(got.shape) != tuple(truth[name].shape):
            raise TimedPathRejection("shape", "timed-path result is missing or has the wrong shape")
        trusted = [expected[key].get(name) for key in TIMED_TRUSTED_FP16] + [truth[name]]
        if any(value is None for value in trusted): raise InfrastructureError("trusted gradient is missing")
        got, trusted = got.detach().float(), [value.detach().float() for value in trusted]
        if masked:
            valid = torch.isfinite(trusted[0])
            for value in trusted[1:]: valid = valid & torch.isfinite(value)
            agreed = ~valid
            for value in trusted[1:-1]: agreed = agreed & same_kind(trusted[0], value)
            zero = torch.zeros_like(got)
            got = torch.where(agreed & ~same_kind(got, trusted[0]), torch.full_like(got, math.inf), torch.where(valid, got, zero))
            trusted = [torch.where(valid, value, zero) for value in trusted]
        return torch.stack([total(got - value) for value in trusted]
                           + [total(trusted[i] - trusted[j]) for i in range(count) for j in range(i)]
                           + [total(trusted[-1])])
    if not truth: return {}
    table = torch.stack([sums(name, False) for name in truth])
    if not bool(torch.isfinite(table).all()):
        table = torch.stack([sums(name, True) for name in truth])
    result = {}
    for group in ("joint", "other", "alpha", "scale", "shift"):
        index = [i for i, member in enumerate(members) if group in ("joint", member)]
        if not index: continue
        *values, scale = (float(x) for x in table[index].sum(dim=0))
        if not (math.isfinite(scale) and scale > 0 and all(math.isfinite(x) for x in values[count:])): continue
        nearest = min((x for x in values[:count] if math.isfinite(x)), default=math.inf)
        result[group] = (math.sqrt(nearest / scale), math.sqrt(max(values[count:]) / scale))
    return result

def _gradient_deviation(observed, expected):
    """Tolerance-normalised gradient deviations of one repeat (above 1 fails) and the joint bound."""
    errors = _gradient_errors(observed, expected)
    ratios = {group: distance / (TIMED_GRADIENT_FLOORS[group] + TIMED_GRADIENT_FACTOR * spread)
              for group, (distance, spread) in errors.items()}
    bound = TIMED_GRADIENT_FLOORS["joint"] + TIMED_GRADIENT_FACTOR * errors["joint"][1] if "joint" in errors else math.inf
    return ratios, bound

def _timed_path_deviation(observed, expected):
    """Compare one candidate block with the oracle block on the same inputs.

    The first failing comparison, in a fixed order, raises one
    TimedPathRejection naming its quantity.
    """
    if len(observed["outputs"]) != len(expected["outputs"]) or len(observed["losses"]) != len(expected["losses"]):
        raise TimedPathRejection("shape", "timed-path result is incomplete")
    output = 0.0
    for key, quantity in (("outputs", "output"), ("losses", "loss")):
        value = _deviation(list(zip(observed[key], expected[key])), TIMED_OUTPUT_TOLERANCE)[0]
        if not value <= 1.0:
            raise TimedPathRejection(quantity if math.isfinite(value) else "nonfinite",
                                     "model output under the timed conditions does not match the reference")
        output = max(output, value)
    result = {"output": output, "gradient": 0.0, "group_gradient": 0.0, "gradient_power": 0}
    if expected["gradients"]:
        ratios, bound = _gradient_deviation(observed["gradients"], expected)
        for group in ("joint", "other", "alpha", "scale", "shift"):
            if group in ratios and not ratios[group] <= 1.0:
                raise TimedPathRejection(f"gradient_{group}" if math.isfinite(ratios[group]) else "nonfinite",
                                         "parameter gradient under the timed conditions does not match the reference")
        result.update(gradient=ratios.get("joint", 0.0), group_gradient=max(ratios.values(), default=0.0),
                      gradient_power=int(bound < TIMED_GRADIENT_POWER_BOUND))
    return result

def _replay_separation(previous, current):
    """How far the oracle's outputs moved between two consecutive repeats.

    Private evidence only (the smallest value over the repeats is kept): above
    1, replaying the previous repeat's outputs would fail on outputs alone.
    """
    value = _deviation(list(zip(previous["outputs"], current["outputs"])), TIMED_OUTPUT_TOLERANCE)[0]
    return value if math.isfinite(value) else 1e9   # evidence is strict JSON

def _fresh_batch(pool, count, generator):
    """One unpredictable batch drawn from the workload's own photographs.

    A random selection of the validation photographs, one circular shift and
    an optional horizontal flip, plus Gaussian noise so no value repeats;
    random targets. Shape, dtype, device and contiguity never vary.
    """
    import torch
    device = pool.device
    def integers(high, size): return torch.randint(high, (size,), generator=generator, device=device)
    values = pool[integers(pool.shape[0], count)]
    values = torch.roll(values, shifts=[int(x) for x in integers(values.shape[-1], 2)], dims=(-2, -1))
    if int(integers(2, 1)): values = values.flip(-1)
    values = values + 0.05 * torch.randn(values.shape, generator=generator, device=device, dtype=values.dtype)
    return [(values.contiguous(), integers(len(CLASSES), count))]

@_trusted_execution("trusted timed inputs failed")
def _repeat_batches(workload, protocols=2):
    """Per protocol, one batch for every warmup and timed repeat, drawn before any timing."""
    import torch
    pool, _ = _real_batch(workload, 2 * len(CLASSES))
    # Seeded from the operating system for every workload: not reproducible by a submission.
    generator = torch.Generator(device=pool.device); generator.manual_seed(int.from_bytes(os.urandom(7), "big"))
    count = int(workload["batch_size"])
    return [[_fresh_batch(pool, count, generator) for _ in range(WARMUP_BLOCKS + TIMED_BLOCKS)] for _ in range(protocols)]

def _paired_protocol(block, candidate, reference, oracle, repeats, phase, protocol):
    """Paired timing in which every candidate call is also a validated call.

    `repeats` holds one batch per warmup and timed repeat. Within a repeat the
    candidate and the reference receive their own clones of the same values,
    in alternating order; no later call sees those values again. After both
    windows have closed the oracle runs the same block on that batch and the
    results the candidate produced inside its window are compared with it;
    for a block with gradients the oracle also supplies two all-fp16-site
    runs and an fp32 run, against which the candidate's gradients are judged.
    """
    if len(repeats) != WARMUP_BLOCKS + TIMED_BLOCKS: raise InfrastructureError("timed input schedule is incomplete")
    # Flat scalars only: the verifier bounds the nesting depth of private metrics.
    worst = {"output": 0.0, "gradient": 0.0, "group_gradient": 0.0, "gradient_power_blocks": 0, "validated_blocks": 0,
             "replay_output_separation": 1e9}
    def call(model, data):
        data = [(images.clone(), targets.clone()) for images, targets in data]
        with (_candidate_execution() if model is candidate else _trusted_execution(phase)):
            return block(model, data)
    ct, rt, previous = [], [], None
    for index, data in enumerate(repeats):
        repeat = index if index < WARMUP_BLOCKS else index - WARMUP_BLOCKS
        results = {}
        for model in ((reference, candidate) if repeat % 2 == 0 else (candidate, reference)):
            results[model is candidate] = call(model, data)
        with _trusted_execution(phase):
            expected = _oracle_block(block, oracle, data)
        with _candidate_execution():
            try:
                deviation = _timed_path_deviation(results[True][1], expected)
            except TimedPathRejection as rejection:
                rejection.protocol = protocol
                raise
        for key in ("output", "gradient", "group_gradient"): worst[key] = max(worst[key], deviation[key])
        worst["gradient_power_blocks"] += deviation["gradient_power"]
        worst["validated_blocks"] += 1
        if previous is not None:
            with _trusted_execution(phase):
                worst["replay_output_separation"] = min(worst["replay_output_separation"], _replay_separation(previous, expected))
        previous = {"outputs": expected["outputs"]}
        if index >= WARMUP_BLOCKS: ct.append(results[True][0]); rt.append(results[False][0])
    return statistics.median(ct), statistics.median(rt), ct, rt, worst

def _paired_inference_latency(candidate, reference, oracle, repeats):
    with _trusted_execution("reference inference setup failed"):
        reference.eval(); oracle.eval()
    with _candidate_execution():
        candidate.eval()
    return _paired_protocol(_inference_block, candidate, reference, oracle, repeats, "reference inference timing failed", "inference")

def _paired_latency(candidate, reference, oracle, repeats):
    with _trusted_execution("reference timing setup failed"):
        reference.train(); oracle.train()
    with _candidate_execution():
        candidate.train()
    return _paired_protocol(_timed_block, candidate, reference, oracle, repeats, "reference timing failed", "training")

def _oracle_model(reference, candidate):
    """The trusted implementation holding the candidate's trained state."""
    import copy
    with _trusted_execution("trusted oracle construction failed"):
        oracle = copy.deepcopy(reference)
    with _candidate_execution():
        oracle.load_state_dict(candidate.state_dict(), strict=True)
    return oracle

def _run_workload(candidate_repo, reference_repo, root, workload):
    with _trusted_execution("trusted reference training failed"):
        ra, rl, rc = _train(reference_repo, workload, root / "reference_output")
    with _candidate_execution():
        ca, cl, cc = _train(candidate_repo, workload, root / "candidate_output")
    with _trusted_execution("trusted reference checkpoint failed"):
        reference = _load_trained_model(reference_repo, rc, workload, "ref_trained")
    with _candidate_execution():
        candidate = _load_trained_model(candidate_repo, cc, workload, "cand_trained")
    oracle = _oracle_model(reference, candidate)
    training_repeats, inference_repeats = _repeat_batches(workload)
    ct, rt, cs, rs, tv = _paired_latency(candidate, reference, oracle, training_repeats)
    ci, ri, cis, ris, iv = _paired_inference_latency(candidate, reference, oracle, inference_repeats)
    return {"training_path_validation": tv, "inference_path_validation": iv,
            "candidate_accuracy": ca, "reference_accuracy": ra, "candidate_latency": ct, "reference_latency": rt,
            "candidate_loss": cl, "reference_loss": rl, "candidate_timing_samples": cs, "reference_timing_samples": rs,
            "candidate_inference_latency": ci, "reference_inference_latency": ri,
            "candidate_inference_samples": cis, "reference_inference_samples": ris}

def _score_rows(rows):
    values = []
    for row in rows:
        quality = min(100., max(0., 50. + 2. * (row["candidate_accuracy"] - row["reference_accuracy"])))
        latency = min(100., max(0., 50. + 200. * math.log2(row["reference_latency"] / row["candidate_latency"])))
        values.append(.6 * quality + .4 * latency)
    return min(100., max(0., statistics.fmean(values)))

def _format_result(rows, suite, guards):
    if len(rows) != 3 or len(suite) != 3: raise RuntimeError("workload results are incomplete")
    numeric = ("candidate_accuracy", "reference_accuracy", "candidate_latency", "reference_latency",
               "candidate_loss", "reference_loss", "candidate_inference_latency", "reference_inference_latency")
    if any(not math.isfinite(float(row[key])) for row in rows for key in numeric):
        raise RuntimeError("workload produced a nonfinite metric")
    if any(row["candidate_latency"] <= 0 or row["reference_latency"] <= 0
           or row["candidate_inference_latency"] <= 0 or row["reference_inference_latency"] <= 0 for row in rows):
        raise RuntimeError("workload produced an invalid latency")
    score, audit, private = _score_rows(rows), [], {}
    for workload, row in zip(suite, rows):
        ap, lp = row["candidate_accuracy"] >= row["reference_accuracy"], row["candidate_latency"] < row["reference_latency"]
        ip = row["candidate_inference_latency"] < row["reference_inference_latency"]
        audit.append({"workload": workload["id"], "metrics": {"top1_accuracy_pct": bool(ap), "training_latency_ms": bool(lp),
                                                              "inference_latency_ms": bool(ip)}})
        private[workload["id"]] = row
    beats = int(all(v for x in audit for v in x["metrics"].values()))
    reference_rows = [{"candidate_accuracy": x["reference_accuracy"], "reference_accuracy": x["reference_accuracy"],
                       "candidate_latency": x["reference_latency"], "reference_latency": x["reference_latency"]} for x in rows]
    reference_score = _score_rows(reference_rows)
    metrics = {"top1_accuracy_pct": statistics.fmean(x["candidate_accuracy"] for x in rows),
        "training_latency_ms": statistics.fmean(x["candidate_latency"] for x in rows),
        "inference_latency_ms": statistics.fmean(x["candidate_inference_latency"] for x in rows),
        "beat_spec_workload_passes": audit, "beats_reference": beats, "reference_score": reference_score,
        "margin": score - reference_score, "per_workload_metrics": private,
        "private_workload_ids": [x["id"] for x in suite],
        "all_final_metrics": {"correctness_guards": guards, "workloads": private}}
    return score, score, f"three GPU workloads completed; score {score:.2f}/100", metrics

def _runtime_receipt(torch):
    # Keep hardware identity in private evidence, including later failure tails.
    device = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device)
    return {"uid": os.getuid(), "gid": os.getgid(), "visible_gpu_count": torch.cuda.device_count(),
            "selected_device": device, "gpu_name": str(properties.name)[:128],
            "gpu_memory_bytes": int(properties.total_memory), "torch": str(torch.__version__)[:64],
            "cuda_runtime": str(torch.version.cuda)[:64]}

def evaluate(solution_path: str):
    runtime = None
    candidate_stage = "unknown"
    try:
        _verify_sources(); _, workloads = _load_manifest(); suite = _selected_workloads(workloads)
        result = apply_submission_patch(solution_path, TASK_DIR, _config())
        if not result.ok:
            invalid = _invalid(result.message)
            # Development feedback names the rejection class; the patch tool's
            # own text, which quotes submitted paths, stays in the message.
            if _development(): invalid[3]["candidate_diagnostic"] = candidate_failure.diagnostic("patch_policy")
            return invalid
        candidate = Path(result.workdir)
        if _introspection_violation(candidate):
            shutil.rmtree(candidate.parent, ignore_errors=True)
            invalid = _invalid(INTROSPECTION_MESSAGE)
            if _development(): invalid[3]["candidate_diagnostic"] = candidate_failure.diagnostic("patch_policy")
            return invalid
        try:
            import torch
            if not torch.cuda.is_available():
                raise InfrastructureError("Verifier NVIDIA GPU is required")
            runtime = _runtime_receipt(torch)
            parent = Path(os.environ.get("FRONTIER_EVAL_SCRATCH", "/workspace")); parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix="dyt-eval-", dir=parent) as temporary, _candidate_imports(candidate):
                root = Path(temporary); reference = _reference_tree(root)
                candidate_stage = "semantic_validation"
                with _training_argv(_train_command(candidate, suite[0], root / suite[0]["id"] / "candidate_output")):
                    guards = _validate_semantics(candidate, reference, suite)
                candidate_stage = "model_workload"
                rows = []
                for x in suite:
                    with _training_argv(_train_command(candidate, x, root / x["id"] / "candidate_output")):
                        rows.append(_run_workload(candidate, reference, root / x["id"], x))
        finally:
            shutil.rmtree(candidate.parent, ignore_errors=True)
        result = _format_result(rows, suite, guards)
        result[3]["all_final_metrics"]["runtime"] = runtime
        return result
    except CandidateError as error:
        result = _invalid(_candidate_message(error))
        visible = _candidate_diagnostic(error, candidate_stage) if _development() else None
        if visible is not None:
            # The verifier admits this to train feedback only as the sole metric
            # of a scored zero, so the private diagnostic travels in final mode.
            result[3]["candidate_diagnostic"] = visible
        else:
            result[3]["all_final_metrics"] = {
                "candidate_failure": _private_candidate_failure(error, candidate_stage), "runtime": runtime}
        return result
    except InfrastructureError as error:
        if runtime is not None:
            error.add_note("private runtime receipt: " + json.dumps(runtime, sort_keys=True))
        raise
    except Exception as error:
        failure = InfrastructureError("trusted task validation or execution failed")
        if runtime is not None:
            failure.add_note("private runtime receipt: " + json.dumps(runtime, sort_keys=True))
        raise failure from error

def prepare():
    _verify_sources(); manifest, workloads = _load_manifest()
    return {"task": "model_wide_elementwise_mapping", "accelerator_required": "nvidia_gpu",
            "workload_count": len(workloads), "dataset_archive_sha256": manifest["source"]["sha256"]}

def main(argv):
    if len(argv) != 2: print("usage: evaluator.py SOLUTION_PATCH", file=sys.stderr); return 2
    score, unbounded, message, metrics = evaluate(argv[1])
    print(json.dumps({"score": score, "score_unbounded": unbounded, "message": message, "metrics": metrics}, sort_keys=True)); return 0

if __name__ == "__main__": raise SystemExit(main(sys.argv))
