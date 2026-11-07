"""CPU ownership allocated before an exact original scheduler job is dispatched."""
import os
from pathlib import Path

from cpu_controller import CPUController
from controller_identity import ENVIRONMENT, load_limits
from instance_manifest import expected_groups
from terminal_contract import Slot


class PhaseControllerFactory:
    def __init__(self, session, groups, *, source_root, maximum_terminal_bytes,
                 maximum_manifest_bytes, environment=None, audit_plan=None):
        self.session, self.groups = session, groups
        self.source_root = Path(source_root)
        self.environment = dict(os.environ if environment is None else environment)
        # Missing Driver identity is a CPU release blocker before remote work.
        for variable in ENVIRONMENT.values():
            if variable not in self.environment:
                raise ValueError('CPU controller requires Driver-injected immutable identity')
        if self.environment['FRONTIER_VERIFICATION_ID'] != session.client.verification_id:
            raise ValueError('Driver grant and CPU verification identity differ')
        self.bounds = load_limits(self.source_root / 'controller-contract-limits.json')
        self.maximum_terminal_bytes = maximum_terminal_bytes
        self.maximum_manifest_bytes = maximum_manifest_bytes
        self.controllers = []
        self.audit_plan = audit_plan

    def __call__(self, kind, variant, fields, request_sha256, output, *, started, deadline):
        group = self.groups[variant]
        if kind not in {'batch', 'correctness', 'startup', 'shutdown', 'group_control'} or group.mode != self.session.mode:
            raise ValueError('CPU controller cannot allocate a foreign operation')
        name = group.workload + '/' + group.rate
        ids = expected_groups(group.mode)[name]
        if kind == 'correctness':
            prefailed = fields.get('prefailed_instance_ids', frozenset())
            requests = set(fields.get('requests', {}))
            if (not isinstance(prefailed, frozenset) or not prefailed.issubset(ids)
                    or requests & prefailed
                    or group.workload != 'swe' or requests | prefailed != set(ids)):
                raise ValueError('CPU correctness ownership differs from the frozen instance plan')
            name += '/correctness'
        elif kind in {'startup', 'shutdown', 'group_control'}:
            # A server that never becomes ready still owns the complete group.
            # Lifecycle ownership is separate from later per-instance terminals.
            name += '/' + kind
        slots = [Slot(self.environment['FRONTIER_RUN_ID'], self.environment['FRONTIER_ATTEMPT_ID'],
            self.environment['FRONTIER_VERIFICATION_ID'], self.session.runner_id,
            group.mode, name, variant, ident, group.model_source_digest, request_sha256)
            for ident in ids]
        controller = CPUController(slots, environment=self.environment,
            content_contract=self.source_root / 'controller-contract.json',
            source_root=self.source_root, evidence=Path(output) / 'cpu-controller',
            maximum_terminal_bytes=self.maximum_terminal_bytes,
            maximum_manifest_bytes=self.maximum_manifest_bytes,
            deadline=deadline, started=started, **self.bounds)
        if kind == 'batch' and group.workload == 'swe':
            from model_case_timing import freeze
            controller.timing_allocation = freeze(controller, source_root=self.source_root,
                contract_path=self.source_root / 'controller-contract.json', **self.bounds)
        if self.audit_plan is not None:
            self.audit_plan.register(kind, variant, request_sha256, group.model_source_digest,
                                     controller)
        self.controllers.append(controller)
        return controller
