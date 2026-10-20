"""Persist the pinned LiteLLM s3_v2 payload in Preview's shared object store."""

import asyncio
import hashlib
import os
from pathlib import Path
import re
import tempfile

from litellm.integrations.custom_logger import CustomLogger
from litellm.integrations.s3_v2 import S3Logger
from litellm._logging import verbose_logger
from litellm.litellm_core_utils.safe_json_dumps import safe_dumps


class PreviewAuditLogger(CustomLogger):
    # Reuse upstream object naming and payload construction without initializing
    # its AWS transport or an unbounded in-memory batch queue.
    create_s3_batch_logging_element = S3Logger.create_s3_batch_logging_element
    s3_path = "runs"
    s3_use_team_prefix = True
    s3_use_key_prefix = True
    s3_strip_base64_files = False

    def __init__(self):
        super().__init__()
        self.root = Path(os.environ["FRONTIER_MODEL_AUDIT_ROOT"])
        self.spool = self.root.parent.parent / ".model-audit-uploads"
        self.spool.mkdir(parents=True, exist_ok=True)

    def persist(self, element):
        parts = element.s3_object_key.split("/")
        if (
            len(parts) < 4 or parts[0] != "runs"
            or any(part in {"", ".", ".."} for part in parts)
            or any(not re.fullmatch(r"[A-Za-z0-9_.:-]+", part) for part in parts[:-1])
        ):
            raise ValueError("invalid model audit object key")
        # Responses IDs can contain Base64 padding. Preserve the payload and
        # routing prefix; only map unsafe upstream filenames to stable keys.
        if not re.fullmatch(r"[A-Za-z0-9_.:-]+", parts[-1]):
            parts[-1] = "sha256-" + hashlib.sha256(parts[-1].encode("utf-8")).hexdigest() + ".json"
        destination = self.root.joinpath(*parts)
        if not destination.parent.resolve().is_relative_to(self.root.resolve()):
            raise ValueError("model audit object escaped storage root")
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = safe_dumps(element.payload).encode("utf-8")
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=self.spool, prefix=".audit-", delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
                os.fchmod(stream.fileno(), 0o644)
            # Published objects are immutable. Never replace bytes observed by a
            # Driver collector; callbacks retried with identical bytes are safe.
            try:
                os.link(temporary, destination)
            except FileExistsError:
                if destination.is_symlink() or destination.read_bytes() != payload:
                    raise ValueError("conflicting model audit object")
            directory = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    async def _log(self, kwargs, start_time):
        try:
            element = self.create_s3_batch_logging_element(
                start_time=start_time,
                standard_logging_payload=kwargs.get("standard_logging_object"),
            )
            if element is not None:
                await asyncio.to_thread(self.persist, element)
        except Exception:
            # Never include provider payloads or credentials in process logs.
            verbose_logger.error("Preview model audit persistence failed")
            raise

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        await self._log(kwargs, start_time)

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        await self._log(kwargs, start_time)


preview_audit = PreviewAuditLogger()
