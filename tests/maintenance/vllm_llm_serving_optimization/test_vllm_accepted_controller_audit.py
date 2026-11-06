"""Real CPU controller -> publisher archive -> Core401-shaped offline export.

These tests establish no live Management exporter or physical cgroup authority.
"""
from vllm_task import TASK_ROOT  # noqa: E402
import deployment_test_guard  # noqa: F401
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from cryptography.exceptions import InvalidSignature

TASK = TASK_ROOT
sys.path[:0] = [str(TASK / 'judge'), str(TASK / 'verifier')]
from accepted_controller_audit import consume_export, unique_json, allocation_exceeds_owner
from controller_audit_plan import PhaseAuditPlan, VARIANTS, controller_directory
from containment_policy import Boundary, child_failure_metadata
from controller_contract_build import emit
from controller_identity import ENVIRONMENT
from evidence import write_evidence
from instance_manifest import expected_groups
from phase_controller import PhaseControllerFactory
from terminal_contract import canonical, terminal


def sha(data):
    return 'sha256:' + hashlib.sha256(data).hexdigest()


class AcceptedControllerAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.variants = getattr(cls, 'variants', VARIANTS)
        cls.fixed_sha = getattr(cls, 'fixed_sha', None)
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        source = cls.root / 'source'
        source.mkdir()
        (source / 'fixture.py').write_text('"""Synthetic immutable source input."""\n')
        emit(source, [], ['fixture.py'], source / 'controller-contract.json',
             source / 'controller-contract-limits.json')
        cls.identity = dict(run_id='run_fixture', attempt_id='attempt_fixture',
            verification_id='verification_fixture', task_config_digest=sha(b'config'),
            executable_image_digest=sha(b'image'))
        environment = {name: cls.identity[key] for key, name in ENVIRONMENT.items()}
        cls.data = cls.root / 'data'
        phase_path = cls.data / 'phase'
        phase_path.mkdir(parents=True)
        phase = PhaseAuditPlan(phase_path, 'eval', environment=environment, source_root=source,
                              variants=cls.variants, fixed_targets_sha256=cls.fixed_sha)
        cls.content = phase.content
        cls.sources = {v: sha(v.encode()) for v in cls.variants}
        cls.controllers = {}
        for group, ids in expected_groups('eval').items():
            group_path = phase_path / group.replace('/', '-')
            group_path.mkdir()
            audit = phase.group(group, group_path)
            operation = 'operation_' + group.replace('/', '_')
            audit.allocated(operation)
            workload, rate = group.split('/')
            groups = {v: SimpleNamespace(mode='eval', workload=workload, rate=rate,
                       model_source_digest=cls.sources[v]) for v in cls.variants}
            session = SimpleNamespace(mode='eval', runner_id=operation,
                                      client=SimpleNamespace(verification_id=cls.identity['verification_id']))
            factory = PhaseControllerFactory(session, groups, source_root=source,
                maximum_terminal_bytes=524288, maximum_manifest_bytes=524288,
                environment=environment, audit_plan=audit)
            for variant in cls.variants:
                kinds = ['startup', 'batch', 'group_control', 'shutdown']
                if workload == 'swe':
                    kinds.append('correctness')
                for kind in kinds:
                    fields = {'requests': dict.fromkeys(ids)} if kind == 'correctness' else {}
                    output = group_path / controller_directory('eval', group, variant, kind)
                    output.parent.mkdir(parents=True, exist_ok=True)
                    # SWE batch timing needs production plan configuration; its
                    # separate boundary is already covered by timing-owner tests.
                    with patch('model_case_timing.freeze', return_value=object()):
                        controller = factory(kind, variant, fields, sha((variant + kind).encode()),
                            output.parent, started=1, deadline=100)
                    cls.controllers[controller.evidence.relative_to(cls.data).as_posix()] = controller
                    for slot in controller.slots:
                        record = terminal(slot, status='complete', cause='normal_completion', elapsed=1, cap=99,
                                          result={'fixture': True}, result_limit_bytes=1024, result_limit_depth=8)
                        controller.ledger.commit_once(slot, record)
                        controller._sign(slot, record)
            audit.complete()
        archive = cls.root / 'original.tar.zst'
        write_evidence(cls.data, archive, 8 * 1024 * 1024)
        cls.archive = archive.read_bytes()
        cls.binding = dict(schema_version='frontier.accepted-controller-export-binding/v1',
            source='current_accepted_original_selected_final', run_id=cls.identity['run_id'],
            attempt_id=cls.identity['attempt_id'], result_id='result_fixture',
            receipt_sha256=sha(b'receipt'), task_version_id='taskver_fixture',
            task_config_digest=cls.identity['task_config_digest'],
            cpu_verifier_image_digest=cls.identity['executable_image_digest'],
            outer_artifact=dict(bucket='fixture', key='original', version_id=None,
                                sha256=sha(b'outer'), size_bytes=5),
            selected_final=dict(payload_id='payload_fixture', payload_sha256=sha(b'payload'),
                verification_id=cls.identity['verification_id'], payload_size_bytes=7),
            nested_artifact=dict(member_path='verifications/verification_fixture/evidence.tar',
                                 sha256=sha(cls.archive), size_bytes=len(cls.archive)))
        cls.limits = dict(maximum_binding_bytes=16384, maximum_archive_bytes=8 * 1024 * 1024,
            maximum_expanded_bytes=128 * 1024 * 1024, maximum_plan_bytes=65536,
            maximum_manifest_bytes=524288, maximum_terminal_bytes=524288, maximum_members=20000)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def consume(self, *, archive=None, binding=None, expected=None, limits=None, mutate=None):
        blob = self.archive if archive is None else archive
        current = deepcopy(self.binding if binding is None else binding)
        if archive is not None:
            current['nested_artifact'].update(sha256=sha(blob), size_bytes=len(blob))
        with tempfile.TemporaryDirectory(dir=self.root) as temporary:
            directory = Path(temporary)
            for name, value in [('binding.json', canonical(current)), ('evidence.tar', blob)]:
                path = directory / name
                path.write_bytes(value)
                path.chmod(0o600)
            if mutate:
                mutate(directory)
            return consume_export(directory, expected_binding=current if expected is None else expected,
                expected_content=self.content, expected_sources=self.sources,
                limits=self.limits if limits is None else limits,
                expected_variants=self.variants, expected_fixed_targets_sha256=self.fixed_sha)

    def modified_archive(self, transform):
        import zstandard
        raw = zstandard.ZstdDecompressor().decompress(self.archive, max_output_size=128 * 1024 * 1024)
        stream = io.BytesIO()
        with tarfile.open(fileobj=io.BytesIO(raw), mode='r:') as original, tarfile.open(fileobj=stream, mode='w') as target:
            for member in original:
                data = original.extractfile(member).read()
                name, data = transform(member.name, data)
                if name is None:
                    continue
                item = tarfile.TarInfo(name)
                item.size = len(data)
                target.addfile(item, io.BytesIO(data))
        return zstandard.ZstdCompressor().compress(stream.getvalue())

    def test_owner_allocation_predicate_preserves_producer_float_precision(self):
        start, end, cap = 58869.388298165, 69669.388298165, 10800
        self.assertGreater(cap, end - start)
        self.assertEqual(start + cap, end)
        self.assertFalse(allocation_exceeds_owner(start, end, cap))
        self.assertTrue(allocation_exceeds_owner(start, end, cap + 0.001))

    def test_actual_cpu_complete_phase_archive_roundtrip(self):
        result = self.consume()
        self.assertEqual(result['status'], 'ACCEPTED_CONTROLLER_SIGNATURES_AND_COVERAGE_VERIFIED')
        self.assertEqual(result['controllers'], 54)
        self.assertFalse(result['scientific_write'])

    def test_independent_binding_and_actual_archive_digest_are_required(self):
        for key in ['attempt_id', 'receipt_sha256', 'task_config_digest', 'cpu_verifier_image_digest']:
            changed = deepcopy(self.binding)
            changed[key] = sha(b'foreign') if key.endswith(('sha256', 'digest')) else 'foreign'
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.consume(expected=changed)
        with self.assertRaises(ValueError):
            self.consume(mutate=lambda directory: (directory / 'evidence.tar').write_bytes(b'foreign'))

    def test_all_explicit_resource_limits_fail_closed(self):
        for key in self.limits:
            limits = dict(self.limits, **{key: 1})
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.consume(limits=limits)

    def test_plan_source_slot_key_and_terminal_substitutions_are_rejected(self):
        for suffix in ['controller-audit-plan.json', 'controller-audit-group.json', '/manifest.json', '.terminal.json']:
            changed = False
            def transform(name, data):
                nonlocal changed
                if not changed and name.endswith(suffix):
                    changed = True
                    value = json.loads(data)
                    if suffix == 'controller-audit-plan.json':
                        value['groups'].pop('bfcl/hidden')
                    elif suffix == 'controller-audit-group.json':
                        value['controllers'][0]['request_digest'] = sha(b'foreign')
                    elif suffix == '/manifest.json':
                        value['public_key'] = 'foreign'
                    else:
                        value['record']['elapsed_seconds'] = 0
                    data = canonical(value)
                return name, data
            with self.subTest(suffix=suffix), self.assertRaises((ValueError, InvalidSignature)):
                self.consume(archive=self.modified_archive(transform))

    def test_missing_terminal_and_unsafe_member_fail_closed(self):
        removed = False
        def remove(name, data):
            nonlocal removed
            if not removed and name.endswith('.terminal.json'):
                removed = True
                return None, data
            return name, data
        with self.assertRaises(ValueError):
            self.consume(archive=self.modified_archive(remove))
        with self.assertRaises(ValueError):
            self.consume(archive=self.modified_archive(lambda name, data: ('../' + name, data)))

    def test_explicit_preallocation_blocker_remains_unscored(self):
        def block(name, data):
            if name == 'phase/swe-public/controller-audit-group.json':
                value = json.loads(data)
                value.update(state='blocked', operation_id=None, controllers=[],
                    blocker=dict(stage='preallocation', error_type='OSError', disposition='unscored_trusted_blocker'))
                data = canonical(value)
            return name, data
        result = self.consume(archive=self.modified_archive(block))
        self.assertEqual(result['status'], 'UNSCORED_TRUSTED_BLOCKER')
        self.assertFalse(result['scientific_write'])

    def test_private_export_modes_and_duplicate_json_are_required(self):
        with self.assertRaises(ValueError):
            self.consume(mutate=lambda directory: (directory / 'binding.json').chmod(0o644))
        with self.assertRaises(ValueError):
            unique_json(b'{"run_id":1,"run_id":2}')

    def test_trusted_plan_fsync_and_memory_errors_never_create_science(self):
        environment = {name: self.identity[key] for key, name in ENVIRONMENT.items()}
        for error in (OSError('fsync unavailable'), MemoryError('CPU allocation failed')):
            with tempfile.TemporaryDirectory(dir=self.root) as temporary:
                with patch('controller_audit_plan.os.fsync', side_effect=error), self.assertRaises(type(error)):
                    PhaseAuditPlan(Path(temporary), 'eval', environment=environment,
                                   source_root=self.root / 'source')

    def test_complete_group_cannot_omit_original_variant(self):
        def omit(name, data):
            if name == 'phase/swe-public/controller-audit-group.json':
                value = json.loads(data)
                value['controllers'] = [entry for entry in value['controllers'] if entry['variant'] != 'reference']
                data = canonical(value)
            return name, data
        with self.assertRaises(ValueError):
            self.consume(archive=self.modified_archive(omit))

    def candidate_failure_archive(self, *, startup, omit_shutdown=False, variant='candidate'):
        prefix = 'phase/swe-public/'
        retained = {'startup'} if startup else {'startup', 'batch', 'group_control', 'shutdown'}
        if omit_shutdown:
            retained.discard('shutdown')
        all_paths = {kind: prefix + controller_directory('eval', 'swe/public', variant, kind)
                     for kind in ['startup', 'batch', 'group_control', 'shutdown', 'correctness']}
        rejected = [path + '/' for kind, path in all_paths.items() if kind not in retained]
        failed_path = all_paths['startup' if startup else 'group_control']
        controller = self.controllers[failed_path]
        def transform(name, data):
            if any(name.startswith(path) for path in rejected):
                return None, data
            if name == prefix + 'controller-audit-group.json':
                value = json.loads(data)
                value['controllers'] = [entry for entry in value['controllers']
                    if entry['variant'] != variant or entry['kind'] in retained]
                return name, canonical(value)
            if name.startswith(failed_path + '/') and name.endswith('.terminal.json'):
                value = json.loads(data)
                from terminal_contract import Slot
                slot = Slot(**value['record']['slot'])
                record = terminal(slot, status='runtime_error', cause='child_exit', elapsed=1, cap=99)
                record['candidate_input_dispatched'] = False
                record['failure_context'] = child_failure_metadata(slot, Boundary('candidate_service', 'execution'), False)
                return name, canonical(controller.signer.sign(record))
            return name, data
        return self.modified_archive(transform)

    def test_candidate_startup_failure_permits_only_original_startup_ownership(self):
        result = self.consume(archive=self.candidate_failure_archive(startup=True))
        self.assertEqual(result['status'], 'ACCEPTED_CONTROLLER_SIGNATURES_AND_COVERAGE_VERIFIED')
        self.assertEqual(result['controllers'], 50)

    def test_later_candidate_failure_preserves_complete_lifecycle(self):
        result = self.consume(archive=self.candidate_failure_archive(startup=False))
        self.assertEqual(result['controllers'], 53)
        with self.assertRaises(ValueError):
            self.consume(archive=self.candidate_failure_archive(startup=False, omit_shutdown=True))

    def test_wrong_group_source_and_missing_healthy_controller_are_rejected(self):
        for defect in ('source', 'missing_correctness'):
            def transform(name, data):
                if name == 'phase/swe-public/controller-audit-group.json':
                    value = json.loads(data)
                    if defect == 'source':
                        value['controllers'][0]['source_digest'] = sha(b'foreign')
                    else:
                        value['controllers'] = [entry for entry in value['controllers']
                            if (entry['variant'], entry['kind']) != ('reference', 'correctness')]
                    data = canonical(value)
                return name, data
            with self.subTest(defect=defect), self.assertRaises(ValueError):
                self.consume(archive=self.modified_archive(transform))

    def test_all_variants_startup_runtime_failure_preserves_signed_coverage(self):
        for variant in VARIANTS:
            with self.subTest(variant=variant):
                result = self.consume(archive=self.candidate_failure_archive(startup=True, variant=variant))
                self.assertEqual(result['status'], 'ACCEPTED_CONTROLLER_SIGNATURES_AND_COVERAGE_VERIFIED')
                self.assertEqual(result['controllers'], 50)

    def test_all_variants_later_runtime_failure_requires_full_lifecycle(self):
        for variant in VARIANTS:
            with self.subTest(variant=variant):
                result = self.consume(archive=self.candidate_failure_archive(startup=False, variant=variant))
                self.assertEqual(result['status'], 'ACCEPTED_CONTROLLER_SIGNATURES_AND_COVERAGE_VERIFIED')
                self.assertEqual(result['controllers'], 53)
                with self.assertRaises(ValueError):
                    self.consume(archive=self.candidate_failure_archive(startup=False, omit_shutdown=True, variant=variant))


if __name__ == '__main__':
    unittest.main()


class CandidateOnlyAcceptedControllerAuditTests(unittest.TestCase):
    variants = ('candidate',)
    fixed_sha = '9'*64
    setUpClass = classmethod(AcceptedControllerAuditTests.setUpClass.__func__)
    tearDownClass = classmethod(AcceptedControllerAuditTests.tearDownClass.__func__)
    consume = AcceptedControllerAuditTests.consume
    modified_archive = AcceptedControllerAuditTests.modified_archive

    def test_candidate_only_complete_signed_archive(self):
        result = self.consume()
        self.assertEqual(result['status'], 'ACCEPTED_CONTROLLER_SIGNATURES_AND_COVERAGE_VERIFIED')
        self.assertEqual(result['controllers'], 18)
        self.assertEqual(set(self.sources), {'candidate'})

    def test_fixed_target_pin_is_independent_of_archive(self):
        def changed(name, data):
            if name.endswith('controller-audit-plan.json'):
                document = unique_json(data)
                document['fixed_targets_sha256'] = '8'*64
                data = canonical(document)
            return name, data
        with self.assertRaises(ValueError):
            self.consume(archive=self.modified_archive(changed))

    def test_group_cannot_reintroduce_unplanned_baseline_variant(self):
        def changed(name, data):
            if name.endswith('controller-audit-group.json'):
                document = unique_json(data)
                document['variants'].insert(0, 'reference')
                data = canonical(document)
            return name, data
        with self.assertRaises(ValueError):
            self.consume(archive=self.modified_archive(changed))
