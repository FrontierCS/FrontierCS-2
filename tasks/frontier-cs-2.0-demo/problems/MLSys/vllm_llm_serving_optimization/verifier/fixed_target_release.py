"""Immutable private release binding, required before ordinary candidate input."""
import json
from pathlib import Path

from cpu_reference_verdict import TrustedMeasurementError
from fixed_targets import canonical_bytes, load_targets
from runtime_config import execution_variants


class FixedTargetReleaseError(TrustedMeasurementError):
    def __init__(self):
        super().__init__(None, None, 'fixed_target_release_binding_unavailable')


def load_release(directory, config):
    """All paths/pins are image-local. No candidate/environment override exists."""
    try:
        directory = Path(directory)
        binding_path = directory/'fixed-target-release.json'
        manifest_path = directory/'fixed-targets.json'
        if binding_path.is_symlink() or manifest_path.is_symlink():
            raise ValueError('release files must be immutable regular image files')
        raw = binding_path.read_bytes()
        binding = json.loads(raw)
        if (not isinstance(binding, dict) or set(binding) != {
                'schema', 'variants', 'manifest_sha256', 'environment'}
                or binding['schema'] != 'frontier.vllm-fixed-target-release/v1'
                or raw != canonical_bytes(binding)
                or binding['variants'] != ['candidate']
                or execution_variants(config) != ('candidate',)):
            raise ValueError('ordinary release must bind candidate-only evaluation')
        return load_targets(manifest_path, expected_sha256=binding['manifest_sha256'],
                            expected_environment=binding['environment'])
    except Exception as error:
        raise FixedTargetReleaseError() from error
