#!/usr/bin/env python3
"""Model-free Candidate that submits a frozen reference through FCS2's Driver."""

from __future__ import annotations

import hashlib
import http.client
import json
import math
import os
from pathlib import Path, PurePosixPath
import stat
import sys
import tarfile
import time
from urllib.parse import quote, urlsplit


REFERENCE_PAYLOAD = Path("/opt/frontier/reference-payload")
SUBMISSION = Path("/workspace/frontier-reference.tar")
MAX_RESPONSE_BYTES = 1024 * 1024
MAX_TOKEN_BYTES = 16 * 1024
MAX_MEMBERS = 100_000
CHUNK = 1024 * 1024


class CandidateError(RuntimeError):
    pass


def _required(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise CandidateError(f"required environment variable {name} is missing")
    return value


def _driver_token() -> str:
    path = Path(os.environ.get("FRONTIER_DRIVER_TOKEN_FILE", "/var/run/frontier/token"))
    try:
        body = path.read_bytes()
    except OSError as error:
        raise CandidateError("Driver token file is unreadable") from error
    if not 0 < len(body) <= MAX_TOKEN_BYTES:
        raise CandidateError("Driver token file has an invalid size")
    try:
        token = body.decode("utf-8").strip()
    except UnicodeDecodeError as error:
        raise CandidateError("Driver token file is not UTF-8") from error
    if not token or any(character.isspace() for character in token):
        raise CandidateError("Driver token file contains an invalid token")
    return token


def _normalized(info: tarfile.TarInfo) -> tarfile.TarInfo:
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    info.mtime = 0
    info.mode &= 0o777
    return info


def _package_reference() -> tuple[int, str]:
    if not REFERENCE_PAYLOAD.is_dir():
        raise CandidateError("frozen reference payload is absent")
    paths = [REFERENCE_PAYLOAD]
    for path in sorted(REFERENCE_PAYLOAD.rglob("*")):
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not (
            stat.S_ISREG(metadata.st_mode) or stat.S_ISDIR(metadata.st_mode)
        ):
            raise CandidateError("frozen reference payload contains an unsafe member")
        paths.append(path)
        if len(paths) > MAX_MEMBERS:
            raise CandidateError("frozen reference payload has too many members")
    with tarfile.open(SUBMISSION, mode="w", format=tarfile.PAX_FORMAT) as archive:
        for path in paths:
            relative = path.relative_to(REFERENCE_PAYLOAD)
            name = PurePosixPath("payload", *relative.parts).as_posix()
            archive.add(path, arcname=name, recursive=False, filter=_normalized)
    size = SUBMISSION.stat().st_size
    digest = hashlib.sha256()
    with SUBMISSION.open("rb") as source:
        while chunk := source.read(CHUNK):
            digest.update(chunk)
    return size, "sha256:" + digest.hexdigest()


class DriverClient:
    def __init__(self, base_url: str, token: str):
        parsed = urlsplit(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise CandidateError("FRONTIER_DRIVER_API_URL is malformed")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise CandidateError("FRONTIER_DRIVER_API_URL contains forbidden components")
        self.scheme = parsed.scheme
        self.host = parsed.hostname
        self.port = parsed.port
        self.base_path = parsed.path.rstrip("/")
        self.token = token

    def _connection(self):
        cls = http.client.HTTPSConnection if self.scheme == "https" else http.client.HTTPConnection
        return cls(self.host, self.port, timeout=60)

    @staticmethod
    def _decode(response, expected: set[int]) -> dict:
        body = response.read(MAX_RESPONSE_BYTES + 1)
        if len(body) > MAX_RESPONSE_BYTES:
            raise CandidateError("Driver response exceeded its bound")
        if response.status not in expected:
            # An unexpected response body is not an approved feedback channel.
            raise CandidateError(f"Driver returned HTTP {response.status}")
        try:
            document = json.loads(body)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise CandidateError("Driver response was not JSON") from error
        if not isinstance(document, dict):
            raise CandidateError("Driver response was not an object")
        return document

    def upload(self, *, size: int, digest: str, purpose: str) -> str:
        connection = self._connection()
        try:
            connection.putrequest("POST", self.base_path + "/v1/payloads")
            headers = {
                "Authorization": "Bearer " + self.token,
                "Content-Type": "application/x-tar",
                "Content-Length": str(size),
                "X-Content-SHA256": digest,
                "X-Frontier-Purpose": purpose,
                "Idempotency-Key": f"reference-{purpose}-{digest.removeprefix('sha256:')}",
            }
            for name, value in headers.items():
                connection.putheader(name, value)
            connection.endheaders()
            with SUBMISSION.open("rb") as source:
                while chunk := source.read(CHUNK):
                    connection.send(chunk)
            result = self._decode(connection.getresponse(), {200, 201})
        finally:
            connection.close()
        payload_id = result.get("payload_id")
        if not isinstance(payload_id, str) or not payload_id:
            raise CandidateError("Driver payload response omitted payload_id")
        return payload_id

    def post_json(self, path: str, document: dict, expected: set[int], key: str) -> dict:
        body = json.dumps(document, separators=(",", ":")).encode()
        connection = self._connection()
        try:
            connection.request(
                "POST",
                self.base_path + path,
                body=body,
                headers={
                    "Authorization": "Bearer " + self.token,
                    "Content-Type": "application/json",
                    "Content-Length": str(len(body)),
                    "Idempotency-Key": key,
                },
            )
            return self._decode(connection.getresponse(), expected)
        finally:
            connection.close()

    def verification(self, verification_id: str) -> dict:
        connection = self._connection()
        try:
            connection.request(
                "GET",
                self.base_path + "/v1/verifications/" + quote(verification_id, safe=""),
                headers={"Authorization": "Bearer " + self.token},
            )
            return self._decode(connection.getresponse(), {200})
        finally:
            connection.close()


def _train_result_event(result: dict, verification_id: str) -> dict:
    if result.get("verification_id") != verification_id or result.get("mode") != "train":
        raise CandidateError("Driver returned a mismatched train verification")
    score = result.get("raw_score")
    if (isinstance(score, bool) or not isinstance(score, (int, float))
            or not math.isfinite(score) or score < 0):
        raise CandidateError("passed train verification omitted a finite nonnegative raw_score")
    metrics = result.get("metrics")
    if not isinstance(metrics, dict):
        raise CandidateError("passed train verification omitted its metrics object")
    try:
        # Copy only the Agent-facing train metrics. Do not retain the rest of the
        # response, environment, Driver credentials or frozen reference payload.
        copied = json.loads(json.dumps(metrics, allow_nan=False))
    except (TypeError, ValueError):
        raise CandidateError("train metrics are not finite JSON") from None
    return {"event": "reference_train_result", "verification_id": verification_id,
            "score": float(score), "metrics": copied}


def _wait_for_train(client: DriverClient, verification_id: str) -> dict:
    # The Driver's authored Run deadline owns the hard limit, including long GPU
    # verification. A reference overlay must not introduce a second private cap.
    while True:
        result = client.verification(verification_id)
        state = result.get("state")
        if state == "passed":
            return _train_result_event(result, verification_id)
        if state == "failed":
            raise CandidateError("reference train verification failed")
        if state not in {"queued", "running"}:
            raise CandidateError("reference train verification has an unknown state")
        time.sleep(2)


def main() -> int:
    size, digest = _package_reference()
    client = DriverClient(_required("FRONTIER_DRIVER_API_URL"), _driver_token())
    train_payload = client.upload(size=size, digest=digest, purpose="train")
    train = client.post_json(
        "/v1/verifier-train",
        {"payload_id": train_payload},
        {202},
        "reference-train-" + train_payload,
    )
    verification_id = train.get("verification_id")
    if not isinstance(verification_id, str) or not verification_id:
        raise CandidateError("Driver train response omitted verification_id")
    event = _wait_for_train(client, verification_id)
    print(json.dumps(event, sort_keys=True, separators=(",", ":"), allow_nan=False), flush=True)
    final_payload = client.upload(size=size, digest=digest, purpose="final")
    final = client.post_json(
        "/v1/submissions",
        {"payload_id": final_payload},
        {202},
        "reference-final-" + final_payload,
    )
    if not isinstance(final.get("verification_id"), str):
        raise CandidateError("Driver final response omitted verification_id")
    print("[frontier-reference] final submission accepted", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except CandidateError as error:
        print(f"[frontier-reference] error: {error}", file=sys.stderr, flush=True)
        raise SystemExit(1)
