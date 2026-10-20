#!/usr/bin/env python3
"""Replay native reference overlays against seeded Preview tasks, resumably."""

from preview.acceptance import (
    CANCEL_ATTEMPTS,
    CLASSIFICATION_KEYS,
    ERROR_HISTORY_LIMIT,
    EXECUTION,
    RESPONSE_LIMIT,
    TERMINAL,
    ManagementError,
    ObservationDeadline,
    cancel_expired,
    candidate_document,
    management_json,
    observation_timeout,
    record_error,
    replay_task,
    save_receipt,
    successful_train,
    verdict,
)
from preview.acceptance import (
    reference_parity_main as main,
)

if __name__ == "__main__":
    raise SystemExit(main())
