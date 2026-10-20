"""Reference descriptors and baseline bindings in the publication catalog."""

from preview.acceptance import REFERENCE_EXECUTION, REFERENCE_READINESS_GUARD
from preview.infra import REPOSITORY_ROOT as ROOT
from preview.publication import (
    DIGEST,
    LAUNCHER,
    MANIFEST,
    PLACEHOLDER,
    RULE,
    file_sha,
    load_baselines,
    load_manifest,
    validate_baseline,
    validate_reference,
)
