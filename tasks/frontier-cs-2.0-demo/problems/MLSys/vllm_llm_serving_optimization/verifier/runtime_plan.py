"""Explicit publisher inputs for the native orchestrator; no resource defaults."""
from instance_manifest import expected_groups
from runtime_config import validate_projection, execution_variants
from sidecar_catalog import VARIANTS

SCHEMA = 'frontier.vllm-runtime-plan/v3'
FORMAL_ANSWER_SECONDS = 18000
SHORT_ANSWER_SECONDS = 1800
SWE_TEST_TIMEOUT_SECONDS = 1200  # Original canonical per-instance test allowance.
PHASE_KEYS = {'groups', 'cleanup_seconds', 'capture_seconds', 'startup_seconds', 'stop_seconds',
              'max_stdout_bytes', 'max_stderr_bytes'}
RELAY_KEYS = {'job_bytes', 'response_bytes', 'receipt_bytes', 'batch_timeout_seconds',
              'correctness_timeout_seconds', 'receipt_timeout_seconds'}
CORRECTNESS_KEYS = {'request_bytes', 'output_bytes'}


def group_operations(mode, group, *, variants=VARIANTS):
    """SSH operations one group parent needs: start/relay/stop per variant, then correctness."""
    operations = 3 * len(variants)
    if group.startswith('swe/'):
        operations += len(variants) * (1 + len(expected_groups(mode)['swe/public']))
    return operations


def phase_operations(mode, *, variants=VARIANTS):
    return sum(group_operations(mode, group, variants=variants) for group in expected_groups(mode))


def validate_transport(relay, correctness, config, phases):
    """Relay/correctness bounds against the SSH stream and each phase budget.

    ``phases`` maps a mode to its ``max_stderr_bytes`` and the seconds left for
    work after cleanup and capture reserves.
    """
    per_phase = ('batch_timeout_seconds', 'correctness_timeout_seconds')
    if (not isinstance(relay, dict) or set(relay) != RELAY_KEYS
            or any(not isinstance(relay[key], dict) or set(relay[key]) != {'train', 'eval'}
                   or any(type(n) is not int or n <= 0 for n in relay[key].values()) for key in per_phase)
            or any(type(n) is not int or n <= 0 for k, n in relay.items() if k not in per_phase)):
        raise ValueError('every relay byte and time bound must be explicit and positive')
    if (not isinstance(correctness, dict) or set(correctness) != CORRECTNESS_KEYS
            or any(type(n) is not int or n <= 0 for n in correctness.values())):
        raise ValueError('every correctness byte bound must be explicit and positive')
    if any(n < SWE_TEST_TIMEOUT_SECONDS for n in relay['correctness_timeout_seconds'].values()):
        raise ValueError('correctness relay cannot shorten the original 1200-second test allowance')
    if correctness['output_bytes'] > relay['receipt_bytes']:
        raise ValueError('a correctness receipt must fit one relay receipt fetch')
    stream = config['audit']['max_ssh_bytes_per_direction']
    for mode, bounds in phases.items():
        if (relay['job_bytes'] > stream
                or relay['response_bytes'] + bounds['max_stderr_bytes'] > stream
                or relay['receipt_bytes'] + bounds['max_stderr_bytes'] > stream
                or relay['batch_timeout_seconds'][mode] > bounds['budget_seconds']
                or relay['correctness_timeout_seconds'][mode] > bounds['budget_seconds']
                or relay['receipt_timeout_seconds'] > bounds['budget_seconds']):
            raise ValueError('relay bounds exceed the published phase/stream allocation')


def validate_plan(plan, config):
    validate_projection(config)
    keys = {'schema', 'max_payload_bytes', 'phases', 'relay', 'correctness'}
    if not isinstance(plan, dict) or set(plan) != keys or plan['schema'] != SCHEMA:
        raise ValueError('invalid closed publisher runtime plan')
    if type(plan['max_payload_bytes']) is not int or plan['max_payload_bytes'] <= 0:
        raise ValueError('explicit payload archive bound required')
    if not isinstance(plan['phases'], dict) or set(plan['phases']) != {'train', 'eval'}:
        raise ValueError('both independent phases required')
    relay, correctness = plan['relay'], plan['correctness']
    stream = config['audit']['max_ssh_bytes_per_direction']
    budgets = {}
    for mode, phase in plan['phases'].items():
        if (not isinstance(phase, dict) or set(phase) != PHASE_KEYS
                or any(type(n) is not int or n <= 0 for key, n in phase.items() if key != 'groups')
                or not isinstance(phase['groups'], dict)
                or set(phase['groups']) != set(expected_groups(mode))
                or any(not isinstance(cap, dict) or set(cap) != {'gpu_seconds'}
                       or type(cap['gpu_seconds']) is not int or cap['gpu_seconds'] <= 0
                       for cap in phase['groups'].values())):
            raise ValueError('every phase requires explicit positive lifetime and stream limits')
        seconds = config['verifier_quota']['max_train_wall_seconds' if mode == 'train' else 'eval_timeout_seconds']
        reserve = phase['cleanup_seconds'] + phase['capture_seconds']
        if (phase['cleanup_seconds'] >= seconds
                or max(cap['gpu_seconds'] for cap in phase['groups'].values()) > seconds - reserve
                or phase['startup_seconds'] + phase['stop_seconds'] >= seconds - reserve
                or phase['max_stdout_bytes'] + phase['max_stderr_bytes'] > stream):
            raise ValueError('runtime plan exceeds its published phase/stream allocation')
        if any(cap['gpu_seconds'] <= reserve + 1 for cap in phase['groups'].values()):
            raise ValueError('each group grant must fund cleanup, capture and clock margin')
        budgets[mode] = {'max_stderr_bytes': phase['max_stderr_bytes'], 'budget_seconds': seconds - reserve}
    validate_transport(relay, correctness, config, budgets)
    pools = config['audit']['ssh_session_budgets']
    if (pools['verifier_train'] < phase_operations('train', variants=execution_variants(config)) * config['verifier_quota']['max_train_calls']
            or pools['verifier_eval'] < phase_operations('eval', variants=execution_variants(config))):
        raise ValueError('SSH pools do not fund complete train allowance and independent final')
    if config['verifier_quota']['max_gpu_instances'] < len(expected_groups('eval')):
        raise ValueError('Verifier quota cannot create four concurrent workload parents')
    required_gpu = (sum(cap['gpu_seconds'] for cap in plan['phases']['train']['groups'].values())
                    * config['verifier_quota']['max_train_calls']
                    + sum(cap['gpu_seconds'] for cap in plan['phases']['eval']['groups'].values()))
    if config['verifier_quota']['max_gpu_seconds'] < required_gpu:
        raise ValueError('GPU allocation does not fund every allowed train and independent final')
    return plan
