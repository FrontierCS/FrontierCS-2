"""Native Task dataset publication for the explicit local-k8s profile.

The trusted operator supplies exact S3-version readbacks, publishes the resulting
single-layer transport OCI using the existing approved registry publisher, runs
the existing trusted seed Job, then calls prepare_registration against the actual
read-only object mount. Management independently checks those same bytes again.
"""
from __future__ import annotations

import copy
import hashlib
import io
import inspect
import json
import os
from pathlib import Path
import re
import stat
import tarfile
from urllib.parse import urlsplit

from .infra import (DeployError, canonical_object_sha256, canonical_json,
                    REPOSITORY_ROOT)


def _sha(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _key(uri: str) -> str:
    parsed = urlsplit(uri)
    key = parsed.path.removeprefix("/")
    if (parsed.scheme != "s3" or not parsed.netloc or parsed.query or parsed.fragment
            or "@" in parsed.netloc or ":" in parsed.netloc or "%" in key
            or "\\" in key or not key.startswith("datasets/")
            or any(p in {"", ".", ".."} for p in key.split("/"))):
        raise DeployError("invalid native dataset URI")
    return key


def _dataset_key_binding(key: str, mode: str, digest_hex: str) -> tuple[str, str]:
    """Accept both published native key shapes without normalizing their paths.

    Legacy keys omit the mode segment; the caller's exact TaskConfig/receipt pair
    supplies the mode. If the segment exists, it must agree with that pair.
    """
    match = re.fullmatch(
        r"datasets/frontier-cs-2\.0/([a-z][a-z0-9_-]*)/(?:(train|eval)/)?sha256/"
        + re.escape(digest_hex) + r"\.(json|tar)", key)
    if not match or (match[2] is not None and match[2] != mode):
        raise DeployError("native dataset key does not bind task/mode/content")
    return match[1], match[3]


def dataset_targets(document: dict) -> list[dict[str, str]]:
    """Validate native config/author bindings; never consult the public catalog."""
    import jsonschema
    from .config import reject_secret_fields
    reject_secret_fields(document, "native document")
    config = document["driver_config"]
    jsonschema.validate(config, json.loads((REPOSITORY_ROOT / "docs/contract/schemas/task-config-v1.schema.json").read_text()))
    receipt = document["author_receipt"]
    if canonical_object_sha256(config) != document["definition_digest"]:
        raise DeployError("native definition digest mismatch")
    if receipt["task_version_id"] != config["task_version_id"]:
        raise DeployError("native receipt TaskVersion mismatch")
    if receipt["definition"] != {"ref": document["definition_ref"], "digest": document["definition_digest"]}:
        raise DeployError("native receipt definition mismatch")
    from .publication import receipt_images
    # The receipt binds published images in Management's receipt shape: declared
    # sidecars by name to tag/digest only, parent resources omitted.
    if receipt_images(config["images"]) != receipt["images"]:
        raise DeployError("native receipt image mismatch")
    if set(config["datasets"]) != {"train", "eval"} or set(receipt["datasets"]) != {"train", "eval"}:
        raise DeployError("native publication requires train and eval")
    targets = []
    for mode in ("train", "eval"):
        obj = config["datasets"][mode]
        original = receipt["datasets"][mode]
        digest = obj["version"]
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
            raise DeployError("invalid native dataset hash")
        key = _key(obj["uri"])
        task_id, extension = _dataset_key_binding(key, mode, digest[7:])
        if task_id != document["task_id"]:
            raise DeployError("native dataset key does not bind task/mode/content")
        if (original["uri"] != obj["uri"] or original["body_sha256"] != digest
                or type(original["size_bytes"]) is not int or original["size_bytes"] <= 0
                or not isinstance(original["s3_version_id"], str) or not original["s3_version_id"]
                or original["s3_version_id"].startswith("local-")):
            raise DeployError("native dataset original provenance mismatch")
        targets.append({"member": f"datasets/{mode}/sha256/{digest[7:]}.{extension}",
                        "target": "/data/" + key, "sha256": digest[7:]})
    if config["datasets"]["train"]["uri"] == config["datasets"]["eval"]["uri"]:
        raise DeployError("train/eval URI alias")
    return validate_seed_targets(targets)


def validate_seed_targets(targets: list[dict[str, str]]) -> list[dict[str, str]]:
    if not isinstance(targets, list) or len(targets) != 2:
        raise DeployError("native seed requires two targets")
    result = copy.deepcopy(targets)
    task_ids, destinations = set(), set()
    for mode, row in zip(("train", "eval"), result):
        if set(row) != {"member", "target", "sha256"} or not re.fullmatch(r"[0-9a-f]{64}", row["sha256"]):
            raise DeployError("invalid native seed target")
        if not re.fullmatch(r"datasets/" + mode + r"/sha256/" + row["sha256"] + r"\.(json|tar)", row["member"]):
            raise DeployError("native seed member does not bind mode/hash")
        if not row["target"].startswith("/data/"):
            raise DeployError("unsafe native seed destination")
        task_id, _ = _dataset_key_binding(row["target"][len("/data/"):], mode, row["sha256"])
        task_ids.add(task_id)
        destinations.add(row["target"])
        if row["member"].rsplit(".", 1)[1] != row["target"].rsplit(".", 1)[1]:
            raise DeployError("native seed extension mismatch")
    if len(task_ids) != 1 or len(destinations) != 2:
        raise DeployError("native seed task mismatch or aliased destination")
    return result


def render_native_seed(config: dict, document: dict, mirror: dict, runner_image: str) -> bytes:
    from .publication import render_task_dataset_seed_manifest
    targets = dataset_targets(document)
    task = {"name": document["task_id"], "datasets": {
        mode + "_sha256": document["driver_config"]["datasets"][mode]["version"]
        for mode in ("train", "eval")}}
    return render_task_dataset_seed_manifest(config, task, {"dataset_image": mirror}, runner_image,
                                             native_targets=targets)


def read_pinned_s3(document: dict, s3_client) -> tuple[dict[str, bytes], dict]:
    """Supported SDK read only: GET exact versions, bound memory by declared sizes.

    Inject an already-authorized client; this module never discovers credentials,
    changes identity, PUTs objects, or writes returned secret/client configuration.
    """
    dataset_targets(document)
    bodies, readback = {}, {}
    for mode in ("train", "eval"):
        original = document["author_receipt"]["datasets"][mode]
        uri = urlsplit(original["uri"])
        response = s3_client.get_object(Bucket=uri.netloc, Key=uri.path.lstrip("/"),
                                        VersionId=original["s3_version_id"])
        stream = response["Body"]
        try:
            if (response.get("VersionId") != original["s3_version_id"]
                    or response.get("ContentLength") != original["size_bytes"]):
                raise DeployError("S3 response version/size differs from original")
            chunks, size = [], 0
            while True:
                chunk = stream.read(min(1024 * 1024, original["size_bytes"] + 1 - size))
                if not chunk:
                    break
                size += len(chunk)
                if size > original["size_bytes"]:
                    raise DeployError("S3 response exceeds declared bytes")
                chunks.append(chunk)
            body = b"".join(chunks)
            if size != original["size_bytes"] or _sha(body) != original["body_sha256"]:
                raise DeployError("S3 response hash/size mismatch")
        finally:
            stream.close()
        bodies[mode] = body
        readback[mode] = {"uri": original["uri"], "version_id": response["VersionId"],
                          "body_sha256": _sha(body), "size_bytes": len(body)}
    return bodies, readback


def transport_oci(document: dict, bodies: dict[str, bytes], s3_readback: dict) -> dict:
    """Prepare an OCI transport locally; no image/source/TaskConfig rewrite.

    s3_readback is the trusted versioned GetObject response, not a version label
    guessed from a receipt. The caller must persist it with the original receipt.
    """
    targets = dataset_targets(document)
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w", format=tarfile.USTAR_FORMAT) as archive:
        for mode, row in zip(("train", "eval"), targets):
            original = document["author_receipt"]["datasets"][mode]
            evidence = s3_readback[mode]
            body = bodies[mode]
            if evidence != {"uri": original["uri"], "version_id": original["s3_version_id"],
                            "body_sha256": original["body_sha256"], "size_bytes": original["size_bytes"]}:
                raise DeployError("S3 readback differs from pinned original")
            if _sha(body) != original["body_sha256"] or len(body) != original["size_bytes"]:
                raise DeployError("transport body differs from original")
            member = tarfile.TarInfo(row["member"])
            member.size = len(body)
            member.mode = 0o444
            member.mtime = 0
            archive.addfile(member, io.BytesIO(body))
    layer = raw.getvalue()
    config = canonical_json({"architecture": "amd64", "os": "linux", "config": {},
                             "rootfs": {"type": "layers", "diff_ids": [_sha(layer)]}})
    manifest = canonical_json({"schemaVersion": 2, "mediaType": "application/vnd.oci.image.manifest.v1+json",
        "config": {"mediaType": "application/vnd.oci.image.config.v1+json", "digest": _sha(config), "size": len(config)},
        "layers": [{"mediaType": "application/vnd.oci.image.layer.v1.tar", "digest": _sha(layer), "size": len(layer)}]})
    return {"manifest": manifest, "config": config, "layer": layer, "digest": _sha(manifest),
            "original_receipt": copy.deepcopy(document["author_receipt"]), "s3_readback": copy.deepcopy(s3_readback)}


def _inspect_rows(root, rows):
    """Stdlib-only checker, also executed in the trusted read-only PVC pod."""
    sizes = {}
    root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for row in rows:
            parent_fd = os.dup(root_fd)
            parts = row["key"].split("/")
            try:
                for part in parts[:-1]:
                    child_fd = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent_fd)
                    os.close(parent_fd)
                    parent_fd = child_fd
                fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent_fd)
                with os.fdopen(fd, "rb") as stream:
                    before = os.fstat(stream.fileno())
                    if not stat.S_ISREG(before.st_mode):
                        raise ValueError("seeded object must be regular")
                    digest = hashlib.sha256()
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(chunk)
                    after = os.fstat(stream.fileno())
                if (before.st_size != row["size"] or digest.hexdigest() != row["sha256"]
                        or (before.st_size, before.st_mtime_ns, before.st_ctime_ns) !=
                           (after.st_size, after.st_mtime_ns, after.st_ctime_ns)):
                    raise ValueError("seeded object hash/size or stable-read mismatch")
                sizes[row["sha256"]] = before.st_size
            finally:
                os.close(parent_fd)
    finally:
        os.close(root_fd)
    return sizes


def _inspection_rows(document):
    dataset_targets(document)
    return [{"key": _key(obj["uri"]), "sha256": obj["version"][7:],
             "size": document["author_receipt"]["datasets"][mode]["size_bytes"]}
            for mode, obj in document["driver_config"]["datasets"].items()]


def inspect_objects(document: dict, root: Path) -> dict[str, int]:
    """Open the actual mount; reject links, special files and unstable reads."""
    try:
        return _inspect_rows(root, _inspection_rows(document))
    except (OSError, ValueError) as error:
        raise DeployError("seeded object absent or unsafe symlink/path: " + str(error)) from error


def render_inspector(config: dict, document: dict, mirror: dict, runner: str) -> bytes:
    """Render only: operator creates this bounded, unprivileged inspection pod."""
    pod = json.loads(render_native_seed(config, document, mirror, runner))["items"][0]["spec"]["template"]
    pod.update({"apiVersion": "v1", "kind": "Pod"})
    name = "native-inspect-" + document["task_id"].replace("_", "-")[:24] + "-" + canonical_object_sha256(document)[7:23]
    pod["metadata"].update({"name": name, "namespace": "frontier-system"})
    pod["spec"]["restartPolicy"] = "Never"
    pod["spec"]["activeDeadlineSeconds"] = 3600
    container = pod["spec"]["containers"][0]
    container["name"] = "inspect"
    container["command"] = ["/usr/local/bin/python3", "-c", "import time; time.sleep(3600)"]
    container["volumeMounts"][0]["readOnly"] = True
    pod["spec"]["volumes"][0]["persistentVolumeClaim"]["readOnly"] = True
    return canonical_json(pod)


def inspect_pvc_objects(document: dict, config: dict, pod_name: str) -> dict[str, int]:
    """GET mount identity, execute the same checker on PVC bytes, GET identity again."""
    from .infra import executable, run_command, kubernetes_environment, state_directory
    if not re.fullmatch(r"[a-z0-9][a-z0-9.-]*", pod_name):
        raise DeployError("invalid inspector pod name")
    command = [executable("kubectl"), "-n", "frontier-system"]
    environment = kubernetes_environment(state_directory(config) / "kubeconfig")
    def run(args, **kwargs):
        return run_command(command + args, environment=environment, capture=True, timeout=300, **kwargs)
    before = json.loads(run(["get", "pod", pod_name, "-o", "json"]))
    if (before["metadata"].get("name") != pod_name
            or before["metadata"].get("namespace") != "frontier-system"
            or before["metadata"].get("deletionTimestamp")):
        raise DeployError("inspector pod identity differs or is terminating")
    spec = before["spec"]
    volumes = {v["name"]: v for v in spec.get("volumes", [])}
    containers = [c for c in spec.get("containers", []) if c["name"] == "inspect"]
    if len(containers) != 1 or before.get("status", {}).get("phase") != "Running":
        raise DeployError("inspector pod must be Running with an inspect container")
    from .images import load_runtime_images
    if containers[0]["image"] != load_runtime_images(config)["storage_controller"]["reference"]:
        raise DeployError("inspector must use the pinned trusted runtime")
    mounts = [m for m in containers[0].get("volumeMounts", []) if m["mountPath"] == "/data"]
    if (len(mounts) != 1 or mounts[0].get("readOnly") is not True
            or "subPath" in mounts[0] or "subPathExpr" in mounts[0]
            or volumes.get(mounts[0]["name"], {}).get("persistentVolumeClaim") != {
                "claimName": "frontier-object-seed", "readOnly": True}):
        raise DeployError("inspector requires the actual read-only object PVC at /data")
    rows = _inspection_rows(document)
    script = "import os, stat, hashlib, json, sys\n" + inspect.getsource(_inspect_rows)
    script += '\nprint(json.dumps(_inspect_rows("/data", json.loads(sys.argv[1]))))\n'
    sizes = json.loads(run(["exec", "-i", pod_name, "-c", "inspect", "--", "/usr/local/bin/python3", "-", json.dumps(rows)], input_text=script))
    after = json.loads(run(["get", "pod", pod_name, "-o", "json"]))
    if (not before["metadata"].get("uid") or before["metadata"]["uid"] != after["metadata"].get("uid")
            or before["spec"] != after["spec"] or after["metadata"].get("deletionTimestamp")
            or sizes != {row["sha256"]: row["size"] for row in rows}):
        raise DeployError("inspector identity or observed byte sizes differ")
    return sizes


def prepare_registration(document: dict, mounted_root: Path | None, transport_provenance: dict,
                         *, config: dict | None = None, pvc_pod: str | None = None) -> dict:
    """Called only after trusted seed completion. No network or publication here."""
    from .publication import local_author_receipt
    sizes = (inspect_pvc_objects(document, config, pvc_pod) if pvc_pod
             else inspect_objects(document, mounted_root))
    original = document["author_receipt"]
    if transport_provenance.get("original_receipt") != original:
        raise DeployError("transport lost original S3 provenance")
    result = copy.deepcopy(document)
    receipt = local_author_receipt(document, sizes, {"source_set_sha256": original["source"]["sha256"]})
    # Task images were not rebuilt. Retain their original source/build facts.
    receipt["source"] = copy.deepcopy(original["source"])
    receipt["build"] = copy.deepcopy(original["build"])
    result["author_receipt"] = receipt
    assert result["driver_config"] == document["driver_config"]
    return {"document": result, "provenance": {"schema": "frontier.native-local-dataset-custody/v1",
        "original_document_sha256": canonical_object_sha256(document),
        "original_receipt": copy.deepcopy(original), "transport": copy.deepcopy(transport_provenance),
        "mounted_observed_sizes": sizes, "local_document_sha256": canonical_object_sha256(result)}}


def write_transport_bundle(document: dict, bodies: dict[str, bytes], readback: dict, output: Path) -> dict:
    """Write a self-contained OCI layout and custody inputs for ordinary OCI tools."""
    transport = transport_oci(document, bodies, readback)
    output.mkdir(parents=True, exist_ok=False)
    blobs = output / "blobs/sha256"
    blobs.mkdir(parents=True)
    for role in ("manifest", "config", "layer"):
        data = transport[role]
        (blobs / _sha(data)[7:]).write_bytes(data)
    (output / "oci-layout").write_bytes(canonical_json({"imageLayoutVersion": "1.0.0"}))
    (output / "index.json").write_bytes(canonical_json({"schemaVersion": 2, "manifests": [{
        "mediaType": "application/vnd.oci.image.manifest.v1+json", "digest": transport["digest"],
        "size": len(transport["manifest"]), "annotations": {"org.opencontainers.image.ref.name": "datasets"}}]}))
    provenance = {"original_receipt": transport["original_receipt"], "s3_readback": readback,
                  "transport_manifest_digest": transport["digest"]}
    (output / "source-document.json").write_bytes(canonical_json(document))
    (output / "provenance.json").write_bytes(canonical_json(provenance))
    return provenance


def verify_transport_bundle(document: dict, bundle: Path) -> dict:
    """Recompute all OCI hashes and both body bindings; external JSON is not proof."""
    from .infra import load_json
    provenance = load_json(bundle / "provenance.json")
    if load_json(bundle / "source-document.json") != document:
        raise DeployError("transport source document differs")
    if load_json(bundle / "oci-layout") != {"imageLayoutVersion": "1.0.0"}:
        raise DeployError("transport OCI layout version differs")
    index = load_json(bundle / "index.json")
    if index.get("schemaVersion") != 2 or len(index.get("manifests", [])) != 1:
        raise DeployError("transport must have one manifest")
    desc = index["manifests"][0]
    if (desc.get("mediaType") != "application/vnd.oci.image.manifest.v1+json"
            or desc.get("annotations") != {"org.opencontainers.image.ref.name": "datasets"}):
        raise DeployError("transport OCI index identity differs")
    def blob(d):
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", d["digest"]):
            raise DeployError("invalid transport blob digest")
        path = bundle / "blobs/sha256" / d["digest"][7:]
        if path.is_symlink() or not path.is_file():
            raise DeployError("transport blob must be regular")
        body = path.read_bytes()
        if _sha(body) != d["digest"] or len(body) != d["size"]:
            raise DeployError("transport blob hash/size mismatch")
        return body
    manifest = json.loads(blob(desc))
    config = json.loads(blob(manifest["config"]))
    if len(manifest.get("layers", [])) != 1:
        raise DeployError("transport must have one layer")
    layer = blob(manifest["layers"][0])
    if config.get("rootfs") != {"type": "layers", "diff_ids": [_sha(layer)]}:
        raise DeployError("transport rootfs differs")
    targets = dataset_targets(document)
    with tarfile.open(fileobj=io.BytesIO(layer), mode="r:") as archive:
        members = archive.getmembers()
        if [m.name for m in members] != [r["member"] for r in targets] or not all(m.isfile() for m in members):
            raise DeployError("transport archive members differ")
        bodies = {mode: archive.extractfile(member).read() for mode, member in zip(("train", "eval"), members)}
    expected = transport_oci(document, bodies, provenance.get("s3_readback", {}))
    if (expected["digest"] != desc["digest"] or provenance != {
            "original_receipt": expected["original_receipt"], "s3_readback": expected["s3_readback"],
            "transport_manifest_digest": expected["digest"]}):
        raise DeployError("transport custody differs")
    return provenance


def validate_registration_readback(expected: dict, actual: dict) -> None:
    """Compare the public record plus its independently fetched definition bytes."""
    if actual.get("id") != expected["driver_config"]["task_version_id"]:
        raise DeployError("native TaskVersion readback identity differs")
    for key, value in expected.items():
        if key != "author_receipt" and actual.get(key) != value:
            raise DeployError("native TaskVersion readback differs: " + key)
    if canonical_object_sha256(actual["driver_config"]) != actual["definition_digest"]:
        raise DeployError("native TaskVersion readback definition digest differs")


def read_definition(actual: dict, config: dict) -> dict:
    """Fetch the API's actual definition ref; absent artifacts fail closed."""
    from .infra import executable, run_command
    import tempfile
    uri = urlsplit(actual["definition_ref"])
    if uri.scheme != "s3" or not uri.netloc or not uri.path or uri.query or uri.fragment:
        raise DeployError("native registration requires a readable S3 definition artifact")
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "definition.json"
        run_command([executable("aws"), "s3api", "get-object", "--bucket", uri.netloc,
                     "--key", uri.path.lstrip("/"), "--region", config["site"]["aws_region"],
                     str(path)], capture=True, timeout=300)
        definition = json.loads(path.read_bytes())
    if canonical_object_sha256(definition) != actual["definition_digest"]:
        raise DeployError("stored definition artifact digest differs from Management")
    result = copy.deepcopy(actual)
    if result.get("driver_config") is not None and result["driver_config"] != definition:
        raise DeployError("API driver config differs from stored definition artifact")
    result["driver_config"] = definition
    return result


def command(arguments) -> dict:
    """Typed offline publication stages; final registration uses official helper."""
    from .config import read_and_validate
    from .infra import load_json, atomic_write
    document = load_json(arguments.document)
    targets = dataset_targets(document)
    config, _ = read_and_validate(arguments.config)
    if arguments.stage == "plan":
        return {"task_version_id": document["driver_config"]["task_version_id"], "targets": targets,
                "definition_digest": document["definition_digest"], "network_calls": 0}
    if arguments.stage == "prepare":
        # The existing AWS CLI credential chain is used without copying credentials.
        from .infra import executable, run_command
        import tempfile
        class S3Client:
            def get_object(self, **kwargs):
                with tempfile.TemporaryDirectory() as temporary:
                    path = Path(temporary) / "body"
                    raw = run_command([executable("aws"), "s3api", "get-object", "--bucket", kwargs["Bucket"],
                        "--key", kwargs["Key"], "--version-id", kwargs["VersionId"], "--region",
                        config["site"]["aws_region"], str(path)], capture=True, timeout=300)
                    response = json.loads(raw)
                    return {"VersionId": response.get("VersionId"), "ContentLength": response.get("ContentLength"),
                            "Body": io.BytesIO(path.read_bytes())}
        bodies, readback = read_pinned_s3(document, S3Client())
        return write_transport_bundle(document, bodies, readback, arguments.output)
    provenance = verify_transport_bundle(document, arguments.bundle)
    if arguments.stage in {"render-seed", "render-inspector"}:
        digest = provenance["transport_manifest_digest"]
        endpoint = config["network"]["registry_cluster_ip"] + ":5000"
        mirror = {"digest": digest, "reference": endpoint + "/frontier/tasks/" + document["task_id"].replace("_", "-") + "/dataset@" + digest}
        from .images import load_runtime_images
        runner = load_runtime_images(config)["storage_controller"]["reference"]
        manifest = (render_native_seed(config, document, mirror, runner) if arguments.stage == "render-seed"
                    else render_inspector(config, document, mirror, runner))
        atomic_write(arguments.output, manifest, mode=0o600)
        return {"seed_manifest": str(arguments.output), "transport_digest": digest,
                "instruction": "Publish the verified OCI layout to the exact registry repository, execute this trusted seed Job, then use finalize with its actual object mount."}
    result = prepare_registration(document, arguments.mounted_root, provenance,
                                  config=config, pvc_pod=getattr(arguments, "pvc_pod", None))
    if arguments.stage == "register":
        from .publication import ensure_version, find_version
        from .infra import MANAGEMENT_SERVICE, port_forward, state_directory
        expected = result["document"]
        version = expected["driver_config"]["task_version_id"]
        with port_forward(state_directory(config) / "kubeconfig", MANAGEMENT_SERVICE, 8080) as port:
            base = f"http://127.0.0.1:{port}"
            existing = find_version(base, "/api/v1/task-versions", version)
            # Verify the expected artifact even before a possible first POST.
            read_definition({"definition_ref": expected["definition_ref"],
                             "definition_digest": expected["definition_digest"]}, config)
            if existing is not None:
                validate_registration_readback(expected, read_definition(existing, config))
            if arguments.check:
                return {"task_version_id": version, "status": "MATCH" if existing else "MISSING", "registration_performed": False}
            ensure_version(base, "/api/v1/task-versions", version, expected)
            actual = find_version(base, "/api/v1/task-versions", version)
            if actual is None:
                raise DeployError("native TaskVersion readback missing")
            actual = read_definition(actual, config)
            validate_registration_readback(expected, actual)
        result["registration"] = {"task_version_id": version, "definition_digest": expected["definition_digest"], "readback": actual}
    atomic_write(arguments.output, canonical_json(result), mode=0o600)
    return {"envelope": str(arguments.output), "definition_digest": document["definition_digest"],
            "task_version_id": document["driver_config"]["task_version_id"], "registration_performed": arguments.stage == "register"}
