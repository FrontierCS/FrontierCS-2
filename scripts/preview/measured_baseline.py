"""Offline qualification of retained independent native reference measurements.

The packet is an integrity manifest, not a platform signature. Its operator must
supply authentic Management/Driver receipts; this module proves their consistency
with the frozen export, published images and the actual submitted/artifact bytes.
Only the native prepared-reference CPU profile is supported, never Preview parity.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import re
import stat
import tarfile
from pathlib import Path, PurePosixPath

import jsonschema

from .acceptance import REFERENCE_EXECUTION
from .infra import DeployError, duplicate_safe_object
from .publication import RULE

SCHEMA = "frontier.independent-reference-measurement/v1"
DOCUMENTS = {
    "task_create",
    "task_catalog",
    "task_publication",
    "candidate_create",
    "candidate_adoption",
    "candidate_binding",
    "evaluation_create",
    "evaluation",
    "run",
    "events",
}
ARTIFACTS = {"platform_evidence", "train_evidence", "final_evidence"}
MAX_JSON = 4 * 1024 * 1024
MAX_ARCHIVE = 64 * 1024 * 1024
MAX_EXPANDED = 128 * 1024 * 1024
MAX_MEMBER = 32 * 1024 * 1024
PLATFORM = {"architecture": "amd64", "os": "linux"}
GPU = {
    "agent": {
        "allowed_types": [],
        "max_instances": 0,
        "max_gpu_seconds": 0,
        "max_instance_seconds": 0,
    },
    "verifier": {"max_gpu_instances": 0, "max_gpu_seconds": 0},
}


def require(condition, message):
    if not condition:
        raise ValueError("measured baseline: " + message)


def sha(body):
    return "sha256:" + hashlib.sha256(body).hexdigest()


def native_bytes(value):
    # Interoperate with native author receipts; do not change Preview's hashers.
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def parse(body):
    def reject_constant(value):
        raise ValueError("nonfinite JSON constant: " + value)

    try:
        value = json.loads(
            body,
            object_pairs_hook=duplicate_safe_object,
            parse_constant=reject_constant,
        )
        pending = [value]
        while pending:
            item = pending.pop()
            require(
                not isinstance(item, float) or math.isfinite(item),
                "nonfinite JSON number",
            )
            if isinstance(item, dict):
                pending.extend(item.values())
            elif isinstance(item, list):
                pending.extend(item)
        return value
    except (UnicodeError, json.JSONDecodeError, RecursionError, DeployError) as exc:
        raise ValueError("measured baseline: invalid JSON") from exc


def keys(value, expected, label):
    require(
        isinstance(value, dict) and set(value) == set(expected),
        label + " fields differ",
    )


def member_path(value):
    require(
        isinstance(value, str)
        and value
        and "\\" not in value
        and "\x00" not in value
        and not PurePosixPath(value).is_absolute()
        and all(part not in ("", ".", "..") for part in value.split("/")),
        "unsafe relative path",
    )
    return value


def read_regular(path, limit):
    path = Path(path).absolute()
    # Open each component relative to its already-open parent. O_NOFOLLOW on
    # only the leaf would miss parent links and leave a check/open race.
    directory = os.open(path.anchor, os.O_RDONLY | os.O_DIRECTORY)
    try:
        for component in path.parts[1:-1]:
            next_directory = os.open(
                component,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=directory,
            )
            os.close(directory)
            directory = next_directory
        descriptor = os.open(
            path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory
        )
    except OSError as exc:
        raise ValueError(
            "measured baseline: unreadable, nonregular or symlink evidence path"
        ) from exc
    finally:
        os.close(directory)
    with os.fdopen(descriptor, "rb") as stream:
        info = os.fstat(stream.fileno())
        require(
            stat.S_ISREG(info.st_mode) and info.st_size <= limit,
            "nonregular or oversized file",
        )
        body = stream.read(limit + 1)
    require(
        len(body) == info.st_size and len(body) <= limit,
        "file changed or exceeds bound",
    )
    return body


def pinned_file(directory, pin, limit):
    keys(pin, {"path", "sha256", "size_bytes"}, "file pin")
    require(
        type(pin["size_bytes"]) is int and 0 <= pin["size_bytes"] <= limit,
        "invalid file size",
    )
    body = read_regular(directory / member_path(pin["path"]), limit)
    require(
        len(body) == pin["size_bytes"] and sha(body) == pin["sha256"],
        "file pin differs",
    )
    return body


def finite(value):
    return type(value) in (float, int) and math.isfinite(value)


def same_score(left, right):
    # Native JSON f64 transports can differ by one adjacent positive value.
    return (
        finite(left)
        and finite(right)
        and (
            left == right
            or type(left) is float
            and type(right) is float
            and left > 0
            and right > 0
            and math.nextafter(left, right) == right
        )
    )


def typed_id(value, prefix):
    require(
        isinstance(value, str)
        and re.fullmatch(prefix + r"_[0-9A-HJKMNP-TV-Z]{26}", value),
        "invalid " + prefix + " ID",
    )
    return value


def digest(value):
    require(
        isinstance(value, str) and re.fullmatch(r"(sha256:)?[0-9a-f]{64}", value),
        "invalid SHA256",
    )
    return "sha256:" + value.removeprefix("sha256:")


def staging_uri(value, suffix):
    require(
        isinstance(value, str)
        and re.fullmatch(
            r"s3://[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]/" + suffix, value
        ),
        "not an immutable native build URI",
    )


def staging_image(value, role, image_digest):
    require(
        isinstance(value, str)
        and re.fullmatch(
            r"[a-z0-9.-]+(?::[0-9]+)?(?:/[a-z0-9._-]+)*/"
            + role
            + r"(?::[a-zA-Z0-9_.-]+|@"
            + re.escape(image_digest)
            + ")",
            value,
        ),
        "not a native staging image",
    )


def unwrap_catalog(value, task_id):
    if isinstance(value, dict) and "body" in value:
        require(
            value.get("method") == "GET"
            and value.get("http_status") == 200
            and value.get("path") == "/api/v1/task-versions/" + task_id
            and value.get("body_within_bound") is True
            and value.get("final_origin_equal") is True,
            "TaskVersion GET receipt is not successful and bound",
        )
        return value["body"]
    return value


def task_binding(docs, source, definition, entry, repository):
    request, publication = docs["task_create"], docs["task_publication"]
    config, author = request["driver_config"], request["author_receipt"]
    task_id = typed_id(config["task_version_id"], "taskver")
    catalog = unwrap_catalog(docs["task_catalog"], task_id)
    identity = definition["source"]["source_set_sha256"]
    policy = parse(read_regular(source / "policy.json", MAX_JSON))
    logical_ids = {source.name, "frontier-cs-2.0-" + source.name.replace("_", "-")}
    require(
        request["task_id"] in logical_ids and policy["problem_id"] == source.name,
        "TaskVersion belongs to another task",
    )
    for key in (
        "task_id",
        "version",
        "definition_ref",
        "definition_digest",
        "metadata",
    ):
        require(
            catalog.get(key) == request.get(key),
            "TaskVersion catalog/request differ: " + key,
        )
    require(
        catalog.get("id") == task_id and catalog.get("archived_at") is None,
        "TaskVersion ID/archive differs",
    )
    metadata = request["metadata"]
    source_algorithm = "preview-source-identity/v1"
    prepared_profile = (
        metadata.get("source_identity_algorithm") == source_algorithm
        and metadata.get("publication_mode") == "prepared-exact-preview-image-adoption"
    )
    qualified = metadata.get("reference_qualification", {})
    qualified_profile = (
        isinstance(qualified, dict)
        and qualified.get("source_identity_algorithm") == source_algorithm
        and qualified.get("no_rebuild") is True
    )
    require(
        metadata.get("source_digest") == identity
        and (prepared_profile or qualified_profile)
        and author["build"].get("tool_versions", {}).get("source-identity")
        == source_algorithm,
        "TaskVersion source/profile differs",
    )
    config_bytes = native_bytes(config)
    require(
        sha(config_bytes) == request["definition_digest"]
        and author["schema_version"] == "frontier.task-author-release/v1"
        and author["task_version_id"] == task_id
        and author["source"]["sha256"] == identity
        and author["build"]["inputs_sha256"] == identity
        and author["build"]["platform"] == PLATFORM
        and author["definition"]
        == {"ref": request["definition_ref"], "digest": sha(config_bytes)},
        "TaskVersion author/definition/source binding differs",
    )
    staging_uri(
        request["definition_ref"],
        r"definitions/frontier-cs-2.0/[^/]+/sha256/"
        + sha(config_bytes)[7:]
        + r"\.json",
    )
    require(
        set(config["images"]) == {"agent_cpu_base", "verifier"}
        and set(author["images"]) == set(config["images"]),
        "CPU image roles differ",
    )
    for role, public_role in (
        ("agent_cpu_base", "agent_image"),
        ("verifier", "verifier_image"),
    ):
        image = config["images"][role]
        expected = dict(author["images"][role])
        if role == "agent_cpu_base":
            expected["platform"] = PLATFORM
        require(
            image == expected and image["digest"] == entry[public_role]["digest"],
            "TaskVersion image differs from published source",
        )
        staging_image(
            image["reference"],
            "agent" if role == "agent_cpu_base" else role,
            image["digest"],
        )
    require(set(author["datasets"]) == {"train", "eval"}, "dataset roles differ")
    for mode, pin in author["datasets"].items():
        body = read_regular(source / "datasets" / (mode + ".json"), MAX_MEMBER)
        keys(pin, {"body_sha256", "size_bytes", "s3_version_id", "uri"}, "dataset pin")
        require(
            pin["body_sha256"] == sha(body) == definition["datasets"][mode + "_sha256"]
            and pin["size_bytes"] == len(body)
            and isinstance(pin["s3_version_id"], str)
            and bool(pin["s3_version_id"])
            and pin["s3_version_id"] != "null",
            "dataset binding differs",
        )
        staging_uri(
            pin["uri"],
            r"datasets/frontier-cs-2.0/[^/]+/"
            + mode
            + "/sha256/"
            + sha(body)[7:]
            + r"\.json",
        )
    require(policy.get("gpu", GPU) == GPU, "unsupported GPU reference measurement")
    expected = {
        "schema_version": "frontier.task-config/v1",
        "task_version_id": task_id,
        "images": config["images"],
        "datasets": {
            mode: {"uri": pin["uri"], "version": pin["body_sha256"]}
            for mode, pin in author["datasets"].items()
        },
        "agent": policy["agent"],
        "network": policy["network"],
        "quota": {
            "wall_time_seconds": policy["quota"]["wall_time_seconds"],
            "agent_gpu": GPU["agent"],
            "verifier": {
                **GPU["verifier"],
                **{
                    key: policy["quota"][key]
                    for key in (
                        "max_train_calls",
                        "max_train_wall_seconds",
                        "eval_timeout_seconds",
                    )
                },
            },
        },
        "submission": {
            **policy["submission"],
            "media_type": "application/x-tar",
            "required_sha256": True,
        },
        "verifier": {
            **policy["verifier"],
            "result_path": "/frontier/output/result.json",
            "evidence_path": "/frontier/output/evidence.tar.zst",
        },
    }
    require(config == expected, "TaskVersion config changes frozen policy")
    jsonschema.validate(
        config,
        parse(
            read_regular(
                repository / "docs/contract/schemas/task-config-v1.schema.json",
                MAX_JSON,
            )
        ),
    )
    for key, value in {
        "task_version_id": task_id,
        "definition_ref": request["definition_ref"],
        "definition_digest": request["definition_digest"],
        "images": author["images"],
        "datasets": author["datasets"],
    }.items():
        require(
            publication.get(key) == value, "TaskVersion publication differs: " + key
        )
    return task_id, policy


def candidate_binding(docs, source, definition, entry, payload, launcher):
    request, receipt, binding = (
        docs[key]
        for key in ("candidate_create", "candidate_adoption", "candidate_binding")
    )
    config = request["driver_config"]
    candidate_id = typed_id(config["candidate_version_id"], "candver")
    reference = entry["reference"]
    identity = definition["source"]["source_set_sha256"]
    submission = definition["submission"]["entrypoint"]
    require(
        submission in ("payload/solution.patch", "payload/solution.json"),
        "unsupported reference payload entrypoint",
    )
    require(
        read_regular(
            source / ("judge/reference." + submission.rsplit(".", 1)[1]), MAX_MEMBER
        )
        == payload,
        "reference payload differs from frozen source",
    )
    require(
        sha(payload) == reference["payload_sha256"]
        and sha(launcher) == reference["launcher_sha256"],
        "published reference bytes differ",
    )
    image = {
        "reference": config["overlay_image"]["reference"],
        "digest": reference["digest"],
        "format": "frontier.oci-overlay/v1",
        "platform": PLATFORM,
    }
    staging_image(image["reference"], "candidate", image["digest"])
    expected_config = {
        "schema_version": "frontier.candidate-config/v1",
        "candidate_version_id": candidate_id,
        "overlay_image": image,
        "execution": REFERENCE_EXECUTION,
        "model_api": {"allowed_domains": [], "secret_refs": []},
    }
    require(
        config == expected_config,
        "Candidate changes image/guard/execution/model policy",
    )
    expected_metadata = {
        "source": "frontiercs2-preview-reference",
        "model_execution": "deterministic-reference-adopted",
        "profile": "preview-reference-v1",
        "problem_id": source.name,
        "source_sha256": identity,
        "launcher_sha256": sha(launcher),
        "payload_sha256": sha(payload),
        "public_image_reference": reference["repository"] + "@" + reference["digest"],
    }
    # Preserve the existing logical identities used by native adopted releases.
    # They are not an admission credential: typed versions and the entire receipt
    # and frozen config/image/source chain below must still match.
    # The native adopter also used the exact export basename (including '_').
    # Admit that one spelling; do not normalize arbitrary caller-supplied IDs.
    task_slug = source.name.replace("_", "-")
    logical_ids = {
        "frontiersmith2-preview-reference-" + task_slug,
        "frontiersmith2-reference-" + task_slug,
        "reference-" + task_slug + "-authoring",
        "frontiersmith2-reference-" + source.name,
    }
    require(
        request["metadata"] == expected_metadata
        and request["candidate_id"] in logical_ids,
        "Candidate is not the independent adopted reference",
    )
    require(
        request["definition_digest"] == sha(native_bytes(config)),
        "Candidate definition digest differs",
    )
    staging_uri(
        request["definition_ref"],
        r"definitions/(?:frontier-cs-2.0/)?reference-candidates/[^/]+/sha256/"
        + request["definition_digest"][7:]
        + r"\.json",
    )
    require(
        binding.get("schema_version") == "frontier.frontiersmith-reference-adoption/v1"
        and binding.get("profile") == "preview-reference-v1"
        and binding["source"]["sha256"] == identity
        and binding["request_sha256"] == sha(native_bytes(request)),
        "Candidate adoption packet differs",
    )
    for key, body in (("launcher", launcher), ("payload", payload)):
        require(
            binding[key]["sha256"] == sha(body)
            and binding[key]["size_bytes"] == len(body),
            "Candidate installed " + key + " differs",
        )
    require(
        binding["image_manifest"]["sha256"] == image["digest"]
        and binding["definition"]["uri"] == request["definition_ref"]
        and binding["definition"]["body_sha256"] == request["definition_digest"]
        and binding["definition"]["size_bytes"] == len(native_bytes(config))
        and binding["definition"].get("s3_version_id") not in (None, "", "null"),
        "Candidate immutable pins differ",
    )
    expected = {
        "schema_version": "frontier.frontiersmith-reference-adoption-receipt/v1",
        "candidate_version_id": candidate_id,
        "definition_digest": request["definition_digest"],
        "request_sha256": sha(native_bytes(request)),
        "binding_sha256": sha(native_bytes(binding)),
        "overlay_digest": image["digest"],
        "profile": "preview-reference-v1",
        "source_sha256": identity,
        "model_execution": "deterministic-reference-adopted",
        "installed_files": {
            "opt/frontier/reference-candidate": {
                "sha256": sha(launcher),
                "size_bytes": len(launcher),
                "mode": 0o555,
            },
            "opt/frontier/reference-payload/" + submission.split("/")[1]: {
                "sha256": sha(payload),
                "size_bytes": len(payload),
                "mode": 0o555,
            },
        },
    }
    require(
        all(receipt.get(key) == value for key, value in expected.items()),
        "Candidate adoption receipt differs",
    )
    catalog = receipt["catalog"]
    require(catalog.get("id") == candidate_id, "Candidate catalog ID differs")
    for key in (
        "candidate_id",
        "version",
        "definition_ref",
        "definition_digest",
        "driver_config",
        "metadata",
    ):
        require(
            catalog.get(key) == request.get(key), "Candidate catalog differs: " + key
        )
    return candidate_id


def archive_bytes(body):
    if body.startswith(b"\x28\xb5\x2f\xfd"):
        try:
            import zstandard
        except ImportError:
            raise ValueError(
                "measured baseline: install zstandard to read platform evidence"
            ) from None
        try:
            with zstandard.ZstdDecompressor().stream_reader(io.BytesIO(body)) as reader:
                body = reader.read(MAX_EXPANDED + 1)
        except zstandard.ZstdError as exc:
            raise ValueError("measured baseline: invalid zstd archive") from exc
    require(len(body) <= MAX_EXPANDED, "expanded archive exceeds bound")
    return body


def tar_members(body):
    body = archive_bytes(body)
    members, total, seen = {}, 0, set()
    try:
        with tarfile.open(fileobj=io.BytesIO(body), mode="r:") as archive:
            for count, item in enumerate(archive, 1):
                require(count <= 256, "archive has too many members")
                name = member_path(item.name.rstrip("/") if item.isdir() else item.name)
                require(
                    name not in seen
                    and (item.isfile() or item.isdir())
                    and not item.sparse,
                    "duplicate, linked or special archive member",
                )
                seen.add(name)
                require(
                    not item.pax_headers, "extended archive metadata is unsupported"
                )
                require(0 <= item.size <= MAX_MEMBER, "archive member exceeds bound")
                total += item.size
                require(total <= MAX_EXPANDED, "archive member total exceeds bound")
                if item.isfile():
                    data = archive.extractfile(item).read(MAX_MEMBER + 1)
                    require(len(data) == item.size, "truncated archive member")
                    members[name] = data
    except (tarfile.TarError, EOFError) as exc:
        raise ValueError("measured baseline: invalid tar archive") from exc
    require(bool(members), "empty archive")
    return members


def evidence_pin(body, pin, key):
    require(
        isinstance(pin, dict)
        and pin.get("sha256") == sha(body)
        and pin.get("size_bytes") == len(body)
        and pin.get("key") == key,
        "Driver artifact descriptor differs",
    )


def measurement_document(members, mode, dataset, payload_sha, *, problem_id=None):
    names = set(members) & {"metrics.json", "measurement.json"}
    require(len(names) == 1, "requires exactly one supported task measurement")
    name = next(iter(names))
    doc = parse(members[name])
    require(isinstance(doc, dict), "task measurement must be an object")
    require(
        doc.get("mode") == mode
        and digest(doc.get("dataset_sha256")) == dataset
        and digest(doc.get("payload_sha256")) == payload_sha,
        "measurement dataset/payload/mode differs",
    )
    if problem_id == "parallel_radix_sort":
        # verify.py project()/main() emit this exact successful audit shape.
        # Select it by the already source-bound export, never an artifact schema.
        # A generic metrics filename or DBSCAN schema must not bypass this guard.
        require(
            name == "measurement.json", "Radix measurement requires measurement.json"
        )
        keys(
            doc,
            {
                "mode",
                "dataset_sha256",
                "payload_sha256",
                "score",
                "score_unbounded",
                "metrics",
            },
            "Radix measurement",
        )
        score = doc["score"]
        require(
            finite(score)
            and 0 <= score <= 100
            # The evaluator returns the same score twice in this one JSON
            # body. Cross-transport adjacent-f64 tolerance does not apply.
            and finite(doc["score_unbounded"])
            and score == doc["score_unbounded"]
            and isinstance(doc["metrics"], dict),
            "invalid Radix measurement score/metrics",
        )
    elif name == "metrics.json":
        score = doc.get("raw_score")
    else:
        require(
            doc.get("schema") == "parallel-dbscan-evidence/v1",
            "unsupported measurement schema",
        )
        score = doc.get("measurement", {}).get("score")
    require(finite(score), "task measurement score is not finite")
    return score


def run_binding(docs, task_id, candidate_id):
    evaluation, run, create = docs["evaluation"], docs["run"], docs["evaluation_create"]
    evaluation_id, run_id = (
        typed_id(evaluation["id"], "eval"),
        typed_id(run["id"], "run"),
    )
    labels = create.get("labels", {})
    require(
        set(labels) == {"owner", "phase", "purpose"}
        and labels["purpose"] == "independent-authoring-reference"
        and labels["owner"] == docs["candidate_create"]["metadata"]["problem_id"]
        and isinstance(labels["phase"], str)
        and labels["phase"].startswith(
            ("current-reference-", "current-18000-reference-")
        ),
        "measurement must be independent authoring, never Preview parity",
    )
    require(
        create.get("task_version_ids") == [task_id]
        and create.get("candidate_version_id") == candidate_id
        and create.get("max_parallel_runs") == 1,
        "Evaluation create identities differ",
    )
    require(
        evaluation.get("candidate_version_id") == candidate_id
        and evaluation.get("labels") == labels
        and evaluation.get("status") == "COMPLETED"
        and all(
            evaluation.get(key) == value
            for key, value in {
                "total_runs": 1,
                "completed_runs": 1,
                "failed_runs": 0,
                "cancelled_runs": 0,
                "active_runs": 0,
                "queued_runs": 0,
            }.items()
        ),
        "Evaluation must have one completed successful Run",
    )
    items = evaluation.get("items")
    require(isinstance(items, list) and len(items) == 1, "Evaluation items differ")
    item = items[0]
    require(
        run.get("evaluation_id") == evaluation_id
        and run.get("task_version_id") == task_id
        and run.get("candidate_version_id") == candidate_id
        and run.get("state") == "COMPLETED"
        and run.get("outcome") == "SUCCEEDED"
        and run.get("finished_at")
        and finite(run.get("score"))
        and run.get("verdict") in ("PASS", "FAIL"),
        "Run is not complete, successful and scored",
    )
    require(
        item.get("run_id") == run_id
        and item.get("task_version_id") == task_id
        and item.get("run_state") == "COMPLETED"
        and item.get("outcome") == "SUCCEEDED"
        and same_score(item.get("score"), run["score"])
        and item.get("verdict") == run["verdict"],
        "Evaluation item differs from Run",
    )
    events = docs["events"]
    require(isinstance(events, list) and events, "missing actual Run events")
    require(
        all(
            event.get("evaluation_id") == evaluation_id
            and event.get("run_id") == run_id
            for event in events
        ),
        "event belongs to another Run/Evaluation",
    )
    require(len({event["id"] for event in events}) == len(events), "duplicate event")
    train = [
        event for event in events if event.get("kind") == "VERIFIER_TRAIN_COMPLETED"
    ]
    final = [event for event in events if event.get("kind") == "RESULT_ACCEPTED"]
    require(
        len(train) == len(final) == 1,
        "requires one actual train and accepted final result",
    )
    train_event, final_event = train[0], final[0]
    require(
        train_event.get("source") == final_event.get("source") == "DRIVER"
        and train_event["attempt_id"] == final_event["attempt_id"],
        "train/final Driver attempt differs",
    )
    typed_id(train_event["attempt_id"], "attempt")
    train, final = train_event["payload"], final_event["payload"]
    require(
        train.get("status") == "success"
        and finite(train.get("raw_score"))
        and train.get("failure_code") is None
        and train.get("failure_message") is None,
        "training did not succeed",
    )
    require(
        final.get("outcome") == "completed"
        and same_score(final.get("raw_score"), run["score"])
        and final.get("failure") is None
        and final.get("audit_warnings") == [],
        "final result is not clean and scored",
    )
    typed_id(train["verification_id"], "verification")
    typed_id(train["payload_id"], "payload")
    return evaluation_id, run_id, train, final


def artifact_binding(
    artifacts, run_id, train, final, definition, payload, *, problem_id=None
):
    outer = artifacts["platform_evidence"]
    evidence_pin(outer, final["evidence"], f"runs/{run_id}/final/evidence.tar.zst")
    train_id = train["verification_id"]
    train_key = f"runs/{run_id}/verifications/{train_id}/evidence.tar"
    evidence_pin(artifacts["train_evidence"], train["evidence"], train_key)
    members = tar_members(outer)
    require("manifest.json" in members, "missing platform manifest")
    manifest = parse(members["manifest.json"])
    require(
        manifest.get("schema_version") == "frontier.platform-evidence/v4"
        and manifest.get("run_id") == run_id
        and manifest.get("model_requests") == []
        and manifest.get("traffic_sessions") == 0,
        "platform evidence identity/model traffic differs",
    )
    declared = manifest.get("members")
    require(
        isinstance(declared, list) and len(declared) == len(members) - 1,
        "platform member list differs",
    )
    pins = {}
    for pin in declared:
        keys(pin, {"path", "size_bytes", "sha256"}, "platform member")
        name = member_path(pin["path"])
        require(
            name in members
            and name != "manifest.json"
            and name not in pins
            and pin["sha256"] == sha(members[name])
            and pin["size_bytes"] == len(members[name]),
            "platform member integrity differs",
        )
        pins[name] = pin
    results = {mode: [] for mode in ("train", "eval")}
    for name, body in members.items():
        match = re.fullmatch(
            r"verifications/(verification_[0-9A-HJKMNP-TV-Z]{26})/result.json", name
        )
        if match:
            result = parse(body)
            require(
                result.get("schema_version") == "frontier.verifier-result/v1"
                and result.get("status") == "passed"
                and result.get("verification_id") == match[1]
                and result.get("mode") in results
                and finite(result.get("raw_score")),
                "verifier result is not valid and successful",
            )
            results[result["mode"]].append(result)
    require(
        all(len(value) == 1 for value in results.values()),
        "requires one train and one final verifier result",
    )
    train_result, final_result = results["train"][0], results["eval"][0]
    require(
        train_result["verification_id"] == train_id
        and same_score(train_result["raw_score"], train["raw_score"]),
        "train verifier result differs from Driver",
    )
    for result, pin in (
        (train_result, train["evidence"]),
        (final_result, manifest["verifier_evidence"]),
    ):
        require(
            all(
                result.get("evidence", {}).get(key) == pin.get(key)
                for key in ("sha256", "size_bytes")
            ),
            "verifier artifact binding differs",
        )
    require(
        manifest["verifier_evidence"].get("key")
        == f"runs/{run_id}/verifications/{final_result['verification_id']}/evidence.tar",
        "final verifier evidence belongs to another Run",
    )
    require(
        same_score(final_result["raw_score"], final["raw_score"]),
        "final verifier score differs from Driver",
    )
    train_tar = archive_bytes(artifacts["train_evidence"])
    require(
        members.get(f"verifications/{train_id}/evidence.tar") == train_tar,
        "train archive bytes differ",
    )
    require("verifier/evidence.tar" in members, "missing final task evidence")
    evidence_pin(
        artifacts["final_evidence"],
        manifest["verifier_evidence"],
        f"runs/{run_id}/verifications/{final_result['verification_id']}/evidence.tar",
    )
    require(
        archive_bytes(artifacts["final_evidence"]) == members["verifier/evidence.tar"],
        "final archive bytes differ from its verifier descriptor",
    )
    payloads = {}
    for name, body in members.items():
        match = re.fullmatch(
            r"payloads/(payload_[0-9A-HJKMNP-TV-Z]{26})/metadata.json", name
        )
        if match:
            metadata = parse(body)
            purpose, payload_id = metadata.get("purpose"), match[1]
            require(
                purpose in ("train", "final")
                and purpose not in payloads
                and metadata.get("payload_id") == payload_id,
                "submission purpose/identity differs",
            )
            tar = members.get(f"payloads/{payload_id}/payload.tar")
            require(
                tar is not None
                and metadata.get("sha256") == sha(tar)
                and metadata.get("size_bytes") == len(tar)
                and metadata.get("object_key")
                == f"runs/{run_id}/payloads/{payload_id}/payload.tar",
                "submission archive binding differs",
            )
            submitted = tar_members(tar)
            require(
                submitted == {definition["submission"]["entrypoint"]: payload},
                "actual submitted payload differs from frozen reference",
            )
            payloads[purpose] = (payload_id, sha(tar))
    require(
        set(payloads) == {"train", "final"}
        and payloads["train"][0] == train["payload_id"],
        "train/final submissions missing",
    )
    for mode, body, purpose, result in (
        ("train", train_tar, "train", train_result),
        ("eval", members["verifier/evidence.tar"], "final", final_result),
    ):
        score = measurement_document(
            tar_members(body),
            mode,
            definition["datasets"][mode + "_sha256"],
            payloads[purpose][1],
            problem_id=problem_id,
        )
        require(
            same_score(score, result["raw_score"]),
            "task measurement differs from verifier score",
        )
    return train_result["raw_score"], final_result["raw_score"], payloads


def baseline_from_measurement(
    packet_path, source, definition, entry, payload_path, launcher_path, repository
):
    """Validate the closed offline packet and derive its single-sample baseline."""
    packet_path, source = Path(packet_path).absolute(), Path(source).absolute()
    require(
        not packet_path.resolve().is_relative_to(source.resolve()),
        "measurement packet must live outside export",
    )
    require(
        entry.get("reference") is not None and payload_path is not None,
        "measurement requires reference bytes",
    )
    packet_body = read_regular(packet_path, MAX_JSON)
    packet = parse(packet_body)
    keys(packet, {"schema_version", "documents", "artifacts"}, "packet")
    require(packet["schema_version"] == SCHEMA, "unsupported packet schema")
    keys(packet["documents"], DOCUMENTS, "document roles")
    keys(packet["artifacts"], ARTIFACTS, "artifact roles")
    docs = {
        key: parse(pinned_file(packet_path.parent, pin, MAX_JSON))
        for key, pin in packet["documents"].items()
    }
    artifacts = {
        key: pinned_file(packet_path.parent, pin, MAX_ARCHIVE)
        for key, pin in packet["artifacts"].items()
    }
    payload, launcher = (
        read_regular(payload_path, MAX_MEMBER),
        read_regular(launcher_path, MAX_MEMBER),
    )
    try:
        task_id, _ = task_binding(docs, source, definition, entry, repository)
        candidate_id = candidate_binding(
            docs, source, definition, entry, payload, launcher
        )
        evaluation_id, run_id, train, final = run_binding(docs, task_id, candidate_id)
        train_score, score, payloads = artifact_binding(
            artifacts, run_id, train, final, definition, payload, problem_id=source.name
        )
    except (KeyError, TypeError, IndexError, AttributeError) as exc:
        raise ValueError("measured baseline: incomplete or malformed evidence") from exc
    adapter = source / "verifier/fcs2_verifier.py"
    if not adapter.is_file():
        adapter = source / "verifier/verify.py"
    return {
        "agreement_rule": dict(RULE),
        "historical_scores": [score],
        "historical_mean": score,
        "historical_spread": 0.0,
        "datasets": {
            mode: definition["datasets"][mode + "_sha256"] for mode in ("train", "eval")
        },
        "verifier_adapter_sha256": sha(read_regular(adapter, MAX_MEMBER)),
        "evaluator_sha256": sha(
            read_regular(source / "judge/evaluator.py", MAX_MEMBER)
        ),
        "source_set_sha256": definition["source"]["source_set_sha256"],
        "reference_patch_sha256": sha(payload),
        "provenance": "measured",
        "comparable": True,
        "comparability_note": "Independent native authoring reference; one sample does not establish repeatability.",
        "measurement": {
            "schema_version": SCHEMA,
            "packet_sha256": sha(packet_body),
            "task_version_id": task_id,
            "candidate_version_id": candidate_id,
            "evaluation_id": evaluation_id,
            "run_id": run_id,
            "verdict": docs["run"]["verdict"],
            "management_score": docs["run"]["score"],
            "train_score": train_score,
            "submission_sha256": {mode: pin[1] for mode, pin in payloads.items()},
        },
    }
