"""Compatibility entrypoint for full four-group hidden-rate calibration.

Every group runs on its own parent with the declared scheduler and testbed
sidecars; the complete v3 group plan funds all four eval parents.
"""
import json
from pathlib import Path
import time

import four_group_diagnostic as diagnostic
from parallel_phase import run_parallel_phase
from public_calibration import execution_lock

HIDDEN_JPS = diagnostic.HIDDEN_JPS


def validate_calibration(plan, config):
    diagnostic.validate_diagnostic(plan, config, 'eval')
    return plan


def execute(plan, config, *, candidate_patch, output, environment, clock_path, resume=False):
    validate_calibration(plan, config)
    output = Path(output)
    with execution_lock(clock_path):
        output.mkdir(parents=True, exist_ok=resume)
        deadline = time.monotonic() + config['verifier_quota']['eval_timeout_seconds']
        if resume:
            return diagnostic.cleanup_only(config, plan, environment=environment,
                candidate_patch=candidate_patch, clocks=output/'parent-clocks', deadline=deadline)
        operator = run_parallel_phase(config, plan, 'eval', environment=environment,
            candidate_patch=candidate_patch, root=output, evidence=output/'operator-evidence',
            deadline=deadline, calibration=True, hidden_jps=HIDDEN_JPS)
        result = {'diagnostic_only':True, 'acceptance':False, 'score_publishable':False,
                  'formal_rates_approved':True, 'operator_result':operator}
        (output/'diagnostic-result.json').write_text(json.dumps(result, allow_nan=False)+'\n')
        return result


def platform_wrapper(plan, config, environment):
    return diagnostic.platform_wrapper(config, plan, environment)


if __name__ == '__main__': diagnostic.main()
