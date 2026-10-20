"""Native archive, selector, feedback and evidence tests; no provider traffic."""
import io
import json
import os
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).parents[1]/'verifier'))
from verify import dataset_selector,extract_patch,result_projection,write_evidence


class NativeAdapterTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)

    def archive(self,entries):
        path=self.root/'payload.tar'
        with tarfile.open(path,'w',format=tarfile.USTAR_FORMAT) as tar:
            for name,kind,data in entries:
                info=tarfile.TarInfo(name); info.type=kind; info.size=len(data)
                tar.addfile(info,io.BytesIO(data))
        return path

    def test_nonempty_patch_and_safe_tar_entry(self):
        path=self.archive([('payload/solution.patch',tarfile.REGTYPE,b'diff')])
        self.assertEqual(extract_patch(path,self.root/'patch',100000),b'diff')
        self.assertEqual((self.root/'patch').read_bytes(),b'diff')
        for entries in ([('../escape',tarfile.REGTYPE,b'x')],
                        [('payload/solution.patch',tarfile.SYMTYPE,b'')],
                        [('payload/solution.patch',tarfile.REGTYPE,b'')]*2):
            with self.assertRaises(ValueError): extract_patch(self.archive(entries),self.root/'patch',100000)

    def test_input_boundary_faults_are_invalid_submission(self):
        from verify import submission_input
        from unittest.mock import patch
        malformed = {
            'flat': [('solution.patch', tarfile.REGTYPE, b'diff')],
            'empty_archive': [],
            'whitespace_patch': [('payload/solution.patch', tarfile.REGTYPE, b' \n\t')],
            'unicode_whitespace_patch': [('payload/solution.patch', tarfile.REGTYPE, '\u2003'.encode())],
            'directory_patch': [('payload/solution.patch', tarfile.DIRTYPE, b'')],
            'extra': [('payload/solution.patch', tarfile.REGTYPE, b'diff'), ('extra', tarfile.REGTYPE, b'x')],
            'duplicate': [('payload/solution.patch', tarfile.REGTYPE, b'diff')] * 2,
            'traversal': [('../solution.patch', tarfile.REGTYPE, b'x')],
            'absolute': [('/payload/solution.patch', tarfile.REGTYPE, b'x')],
            'symlink': [('payload/solution.patch', tarfile.SYMTYPE, b'')],
            'hardlink': [('payload/solution.patch', tarfile.LNKTYPE, b'')],
            'oversize_patch': [('payload/solution.patch', tarfile.REGTYPE, b'x' * 1500001)],
            'non_utf8': [('payload/solution.patch', tarfile.REGTYPE, b'\xff')],
        }
        for name, entries in malformed.items():
            with self.subTest(name=name), patch('verify.inspect_submission') as inspector:
                candidate, result = submission_input(self.archive(entries), self.root/'patch', 2000000, self.root/'clean')
                self.assertIsNone(candidate)
                self.assertEqual(result['disposition'], 'invalid_submission')
                self.assertFalse(result['valid_submission'])
                inspector.assert_not_called()
        for name, raw in [('not_tar', b'bad'), ('zero_bytes', b''), ('truncated', b'x' * 512)]:
            with self.subTest(name=name):
                archive = self.root/'payload.tar'; archive.write_bytes(raw)
                candidate, result = submission_input(archive, self.root/'patch', 100000, self.root/'clean')
                self.assertIsNone(candidate)
                self.assertEqual(result['disposition'], 'invalid_submission')
        archive = self.archive([('payload/solution.patch', tarfile.REGTYPE, b'diff')])
        self.assertFalse(submission_input(archive, self.root/'patch', 1, self.root/'clean')[1]['valid_submission'])
        link = self.root/'link.tar'; link.symlink_to(archive)
        self.assertFalse(submission_input(link, self.root/'patch', 100000, self.root/'clean')[1]['valid_submission'])

    def test_input_boundary_contains_unknown_errors_but_not_typed_infrastructure(self):
        from verify import submission_input
        from submission import SubmissionInfrastructureError
        from unittest.mock import patch
        archive = self.archive([('payload/solution.patch', tarfile.REGTYPE, b'diff')])
        for target in ('extract_patch', 'inspect_submission'):
            with self.subTest(target=target), patch('verify.'+target, side_effect=ArithmeticError('private input')):
                candidate, result = submission_input(archive, self.root/'patch', 100000, self.root/'clean')
                self.assertIsNone(candidate)
                self.assertEqual(result['disposition'], 'invalid_submission')
                self.assertNotIn('private input', json.dumps(result))
        for invalid in (None, {}, {'valid_submission': 'yes'},
                        {'valid_submission': True, 'unexpected': object()},
                        {'valid_submission': False, 'metric': float('nan')}):
            with self.subTest(invalid=invalid), patch('verify.inspect_submission', return_value=invalid):
                self.assertEqual(submission_input(archive, self.root/'patch', 100000, self.root/'clean')[1]['disposition'], 'invalid_submission')
        with patch('verify.inspect_submission', side_effect=SubmissionInfrastructureError('clean source')):
            with self.assertRaises(SubmissionInfrastructureError):
                submission_input(archive, self.root/'patch', 100000, self.root/'clean')

    def test_native_invalid_input_is_scored_zero_without_phase(self):
        from unittest.mock import patch
        import verify
        for mode in ('train', 'eval'):
            for failure in ('flat', 'empty_archive', 'missing', 'unknown'):
                with self.subTest(mode=mode, failure=failure):
                    work=self.root/(mode+'-'+failure); work.mkdir()
                    (work/'config.json').write_text('{}'); (work/'plan.json').write_text('{}')
                    paths={'/opt/frontier/runtime-config.json':work/'config.json',
                           '/opt/frontier/runtime-plan.json':work/'plan.json'}
                    payload=self.archive([] if failure=='empty_archive' else [
                        ('solution.patch' if failure=='flat' else 'payload/solution.patch',
                         tarfile.REGTYPE, b'diff')])
                    if failure=='missing': payload=work/'absent.tar'
                    dataset=work/'dataset.json'; dataset.write_text('{}')
                    env={'FRONTIER_RESULT_PATH':str(work/'result.json'), 'FRONTIER_EVIDENCE_PATH':str(work/'evidence'),
                         'FRONTIER_VERIFICATION_MODE':mode, 'FRONTIER_DATASET_PATH':str(dataset),
                         'FRONTIER_PAYLOAD_PATH':str(payload), 'FRONTIER_MAX_EVIDENCE_BYTES':'100000'}
                    with patch.dict(os.environ,env), \
                         patch.object(verify,'Path',side_effect=lambda value: Path(paths.get(str(value),value))), \
                         patch.object(verify.tempfile,'mkdtemp',return_value=str(work/'scratch')), \
                         patch.object(verify,'validate_plan',return_value={'max_payload_bytes':100000}), \
                         patch.object(verify,'load_release',return_value=object()), \
                         patch.object(verify,'dataset_selector'), \
                         patch.object(verify,'inspect_submission',side_effect=LookupError('unknown candidate exception')), \
                         patch.object(verify,'write_evidence'), \
                         patch.object(verify,'run_parallel_phase') as phase:
                        (work/'scratch').mkdir()
                        self.assertEqual(verify.main(),0)
                    phase.assert_not_called()
                    result=json.loads((work/'result.json').read_text())
                    self.assertEqual(result['status'],'passed')
                    self.assertEqual(result['raw_score'],0.0)
                    self.assertNotIn('failure',result)
                    if mode=='eval': self.assertEqual(result['private_metrics'],{'beats_reference':0})
                    evidence=json.loads((work/'scratch/evidence/submission.json').read_text())
                    self.assertEqual(evidence['disposition'],'invalid_submission')
                    self.assertFalse((work/'scratch/evidence/failure.json').exists())

    def test_canonical_zero_byte_patch_enters_normal_complete_phase_and_scoring(self):
        from unittest.mock import patch
        import subprocess
        import verify
        import cpu_reference_verdict as scorer
        from instance_manifest import expected_groups
        from reference_verdict import InstanceObservation
        clean=self.root/'clean'; clean.mkdir()
        subprocess.run(['git', 'init', '-q', str(clean)], check=True)
        for mode in ('train', 'eval'):
            with self.subTest(mode=mode):
                work=self.root/(mode+'-legal-empty'); work.mkdir()
                (work/'scratch').mkdir()
                for name in ('config', 'plan', 'dataset'):
                    (work/(name+'.json')).write_text('{}')
                paths={'/opt/frontier/runtime-config.json':work/'config.json',
                       '/opt/frontier/runtime-plan.json':work/'plan.json',
                       '/opt/frontier/vllm-clean':clean}
                payload=self.archive([('payload/solution.patch', tarfile.REGTYPE, b'')])
                env={'FRONTIER_RESULT_PATH':str(work/'result.json'),
                     'FRONTIER_EVIDENCE_PATH':str(work/'evidence'),
                     'FRONTIER_VERIFICATION_MODE':mode,
                     'FRONTIER_DATASET_PATH':str(work/'dataset.json'),
                     'FRONTIER_PAYLOAD_PATH':str(payload),
                     'FRONTIER_MAX_EVIDENCE_BYTES':'100000'}
                plan={'max_payload_bytes':100000, 'phases':{mode:{'groups':{'g':{'gpu_seconds':1}}}}}
                def measured_phase(config, supplied_plan, phase, **kwargs):
                    self.assertEqual(phase, mode)
                    self.assertEqual(kwargs['candidate_patch'], b'')
                    groups=expected_groups(phase)
                    observed=lambda seconds: {group:[InstanceObservation(ident, seconds, True)
                        for ident in ids] for group,ids in groups.items()}
                    # Synthetic CPU measurements retain every original slot and
                    # pass through the production scorer; no GPU claim is made.
                    score=scorer.evaluate_train if phase=='train' else scorer.evaluate_final
                    return score(observed(12.), observed(9.), observed(10.))
                with patch.dict(os.environ,env), \
                     patch.object(verify,'Path',side_effect=lambda value: Path(paths.get(str(value),value))), \
                     patch.object(verify.tempfile,'mkdtemp',return_value=str(work/'scratch')), \
                     patch.object(verify,'validate_plan',return_value=plan), \
                     patch.object(verify,'load_release',return_value=object()), \
                     patch.object(verify,'dataset_selector'), \
                     patch.object(verify,'phase_limits',return_value={'timeout_seconds':1000}), \
                     patch.object(verify,'write_evidence'), \
                     patch.object(verify,'run_parallel_phase',side_effect=measured_phase) as phase:
                    self.assertEqual(verify.main(),0)
                phase.assert_called_once()
                result=json.loads((work/'result.json').read_text())
                self.assertEqual(result['status'],'passed')
                self.assertGreater(result['raw_score'],0)
                self.assertNotIn('failure',result)
                inspected=json.loads((work/'scratch/evidence/submission.json').read_text())
                self.assertTrue(inspected['valid_submission'])
                self.assertNotEqual(inspected.get('disposition'),'invalid_submission')
                self.assertEqual((work/'scratch/solution.patch').read_bytes(),b'')

    def test_dataset_requires_complete_original_phase_and_order(self):
        task=Path(__file__).parents[1]
        for mode in ('train','eval'):
            source=task/'datasets'/f'{mode}.json'
            dataset_selector(source,mode)
            value=json.loads(source.read_text()); value['swe_instance_ids'].pop()
            wrong=self.root/'dataset';wrong.write_text(json.dumps(value))
            with self.assertRaises(ValueError): dataset_selector(wrong,mode)
            with self.assertRaises(ValueError): dataset_selector(source,'eval' if mode=='train' else 'train')

    def test_private_reference_is_eval_only_and_no_operator_detail_leaks(self):
        private={'score':12.3,'pass':False,'reference_verdict':{'private_case':'secret-reference'},'continuous':{'hidden_rate':1.25}}
        train=result_projection(private,'train'); final=result_projection(private,'eval')
        self.assertNotIn('private_metrics',train)
        self.assertEqual(train['metrics'],{})
        self.assertEqual(final['private_metrics'],{'beats_reference':0})
        self.assertNotIn('secret-reference',json.dumps(final))
        for bad in (float('nan'),float('inf'),-1,101,True):
            with self.assertRaises(ValueError): result_projection({'score':bad,'pass':False},'eval')

    def test_real_zstd_evidence_roundtrip_and_overflow_preserves_prior_file(self):
        import zstandard
        directory=self.root/'evidence';directory.mkdir()
        (directory/'selection.json').write_text('{"payload_sha256":"synthetic"}')
        destination=self.root/'evidence.tar.zst'
        write_evidence(directory,destination,100000)
        before=destination.read_bytes()
        with zstandard.ZstdDecompressor().stream_reader(io.BytesIO(before)) as stream:
            with tarfile.open(fileobj=stream,mode='r|') as tar:
                member=tar.next()
                self.assertEqual(member.name,'selection.json')
                self.assertEqual(tar.extractfile(member).read(),(directory/'selection.json').read_bytes())
        (directory/'random').write_bytes(os.urandom(10000))
        with self.assertRaises(ValueError): write_evidence(directory,destination,100)
        self.assertEqual(destination.read_bytes(),before)
        self.assertFalse(destination.with_suffix('.pending').exists())

    def test_final_mode_runs_the_four_group_phase_with_configured_hidden_rates(self):
        from unittest.mock import patch
        import verify
        from batch_coordinator import FINAL_HIDDEN_JPS
        for mode,hidden,trusted_fault in (('eval',FINAL_HIDDEN_JPS,False),('train',None,False),
                                          ('eval',FINAL_HIDDEN_JPS,True),('train',None,True)):
            with self.subTest(mode=mode):
                work=self.root/(mode + ('-trusted-fault' if trusted_fault else '')); work.mkdir()
                (work/'config.json').write_text('{}'); (work/'plan.json').write_text('{}')
                paths={'/opt/frontier/runtime-config.json':work/'config.json',
                       '/opt/frontier/runtime-plan.json':work/'plan.json'}
                payload=self.archive([('payload/solution.patch',tarfile.REGTYPE,b'diff')])
                dataset=work/'dataset.json'; dataset.write_text('{}')
                env={'FRONTIER_RESULT_PATH':str(work/'result.json'),'FRONTIER_EVIDENCE_PATH':str(work/'evidence'),
                     'FRONTIER_VERIFICATION_MODE':mode,'FRONTIER_DATASET_PATH':str(dataset),
                     'FRONTIER_PAYLOAD_PATH':str(payload),'FRONTIER_MAX_EVIDENCE_BYTES':'100000'}
                calls=[]
                def phase(config,plan,phase_mode,**kwargs):
                    calls.append((phase_mode,kwargs.get('hidden_jps'),kwargs.get('calibration',False)))
                    if trusted_fault:
                        from cpu_reference_verdict import TrustedMeasurementError
                        raise TrustedMeasurementError('vanilla', 'swe/public', 'runtime_failure')
                    return {'score':0.0,'pass':False}
                plan={'max_payload_bytes':100000,'phases':{mode:{'groups':{'g':{'gpu_seconds':1}}}}}
                with patch.dict(os.environ,env), \
                     patch.object(verify,'Path',side_effect=lambda value: Path(paths.get(str(value),value))), \
                     patch.object(verify.tempfile,'mkdtemp',return_value=str(work/'scratch')), \
                     patch.object(verify,'validate_plan',return_value=plan), \
                     patch.object(verify,'load_release',return_value=object()), \
                     patch.object(verify,'dataset_selector'), \
                     patch.object(verify,'inspect_submission',return_value={'valid_submission':True}), \
                     patch.object(verify,'phase_limits',return_value={'timeout_seconds':10}), \
                     patch.object(verify,'write_evidence'), \
                     patch.object(verify,'run_parallel_phase',side_effect=phase):
                    (work/'scratch').mkdir()
                    self.assertEqual(verify.main(),0)
                self.assertEqual(calls,[(mode,hidden,False)])
                result = json.loads((work/'result.json').read_text())
                if trusted_fault:
                    self.assertEqual(result, {'status': 'failed', 'failure': {
                        'code': 'vllm_trusted_measurement_unavailable',
                        'message': 'Trusted verification measurements are unavailable.',
                        'retryable': True}})
                    self.assertNotIn('raw_score', result)
                    private = json.loads((work/'scratch/evidence/failure.json').read_text())
                    self.assertEqual(private['attribution'], 'infrastructure')
                    self.assertFalse(private['score_ready'])
                    self.assertEqual(private['variant'], 'vanilla')
                else:
                    self.assertFalse((work/'scratch/evidence/failure.json').exists())
                    self.assertEqual(result['status'],'passed')


if __name__=='__main__': unittest.main()
