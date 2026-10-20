"""Affected storage criterion checks only; no build or scientific simulation."""
import deployment_test_guard  # noqa: F401  # isort: skip

from pathlib import Path
from decimal import Decimal
import copy
import json
import sys
import unittest

ROOT=Path(__file__).resolve().parents[3]/"tasks/frontier-cs-2.0-demo/problems/Arch/runlts_predictor/judge"
from tests.maintenance.runlts_predictor.runtime_imports import MODULES
e = MODULES["evaluator"]
feedback = MODULES["beat_spec_feedback"]

class LogicalStorageTests(unittest.TestCase):
    def grade(self,bits,*,equal=False,final=True):
        rows=[{'id':'fixture-'+str(i)} for i in range(105 if final else 24)]
        reference={r['id']:e.Measurement(1000,20,200,2000,Decimal('20'),Decimal('200')) for r in rows}
        candidate=reference if equal else {r['id']:e.Measurement(1000,10,100,2000,Decimal('10'),Decimal('100')) for r in rows}
        return e.grade(rows,candidate,reference,final,logical_storage_bits=bits)
    def test_fixed_cap_accepts_equality_above_reference_size(self):
        result=self.grade(1572864)
        self.assertEqual(result[3]['beats_reference'],1)
        self.assertEqual(len(result[3]['beat_spec_workload_passes']),105)
        p=feedback.private_metrics(result[3],beat_spec_path=ROOT/'beat_spec.json')['beat_spec_progress']
        self.assertEqual(set(p),{'br_mis_pki','cyc_wp_pki','logical_storage_bits'})
        self.assertEqual((p['logical_storage_bits']['passed'],p['logical_storage_bits']['failed'],p['logical_storage_bits']['total']),(105,0,105))
    def test_prediction_ties_still_fail(self):
        result=self.grade(1570890,equal=True)
        self.assertEqual(result[3]['beats_reference'],0)
        p=feedback.private_metrics(result[3],beat_spec_path=ROOT/'beat_spec.json')['beat_spec_progress']
        self.assertEqual((p['br_mis_pki']['passed'],p['cyc_wp_pki']['passed'],p['logical_storage_bits']['passed']),(0,0,105))
    def test_over_cap_is_invalid(self):
        with self.assertRaises(e.CandidateError):self.grade(1572865)
    def test_invalid_or_missing_audit_fails_closed(self):
        for value in [None,True,False,float('nan'),float('inf'),1572864.0,-1,1517026]:
            with self.subTest(value=value),self.assertRaises(e.EvaluationError):self.grade(value)
    def test_component_sum_and_score_independence(self):
        a,b=self.grade(1517027),self.grade(1572864)
        self.assertEqual(a[:2],b[:2]);self.assertEqual(b[3]['logical_storage_accounting']['sr_bits'],55837)
    def test_development_progress_has_24_rows(self):
        result=self.grade(1572864,final=False)
        p=feedback.private_metrics(result[3],beat_spec_path=ROOT/'beat_spec.json')['beat_spec_progress']
        self.assertEqual(p['logical_storage_bits']['total'],24)
    def mixed(self,final=False):
        # 24 development rows: 9 win both comparisons, 5 only BrMisPKI, 4 only CycWpPKI, 6 neither.
        rows=[{'id':'fixture-'+str(i)} for i in range(105 if final else 24)]
        reference={r['id']:e.Measurement(1000,20,200,2000,Decimal('20'),Decimal('200')) for r in rows}
        def cell(i):
            b,w=(10,100) if i<9 else (10,200) if i<14 else (20,100) if i<18 else (20,300)
            return e.Measurement(1000,b,w,2000,Decimal(b),Decimal(w))
        candidate={r['id']:cell(i) for i,r in enumerate(rows)}
        return e.grade(rows,candidate,reference,final,logical_storage_bits=1572864)
    @staticmethod
    def counts(passed,total=24):
        return {'passed':passed,'failed':total-passed,'total':total,
                'pass_percent':round(100.0*passed/total,2),'fail_percent':round(100.0*(total-passed)/total,2)}
    def test_joint_count_is_over_the_two_prediction_comparisons(self):
        result=self.mixed()
        self.assertEqual(e.PREDICTION_METRICS,('br_mis_pki','cyc_wp_pki'))
        self.assertEqual(result[3]['beats_reference'],0)
        self.assertEqual(result[3]['all_metrics_progress'],self.counts(9))
        visible=feedback.public_metrics(result[3],beat_spec_path=ROOT/'beat_spec.json',
            configured_visible=e.CONFIG['evaluation']['solver_visible_metrics'])
        self.assertEqual(set(visible),{'train_br_mis_pki_mean','train_cyc_wp_pki_mean','all_metrics_progress','category_means','beat_spec_progress'})
        self.assertEqual(visible['all_metrics_progress'],self.counts(9))
        self.assertEqual(visible['beat_spec_progress'],{'br_mis_pki':self.counts(14),'cyc_wp_pki':self.counts(13),
            'logical_storage_bits':self.counts(24)})
        # The always-true storage row neither inflates the joint count nor is needed by it.
        rows=result[3]['beat_spec_workload_passes']
        self.assertEqual(sum(all(r['metrics'].values()) for r in rows),9)
        self.assertTrue(all(r['metrics']['logical_storage_bits'] for r in rows))
    def test_joint_count_does_not_depend_on_row_order_and_names_no_trace(self):
        result=self.mixed()
        encoded=json.dumps(result[3]['all_metrics_progress'])
        self.assertNotIn('fixture',encoded)
        self.assertEqual(set(result[3]['all_metrics_progress']),{'passed','failed','total','pass_percent','fail_percent'})
        self.assertEqual(self.grade(1572864,final=False)[3]['all_metrics_progress'],self.counts(24))
        self.assertEqual(self.grade(1572864,equal=True,final=False)[3]['all_metrics_progress'],self.counts(0))
    def test_final_result_has_no_joint_count(self):
        self.assertNotIn('all_metrics_progress',self.mixed(final=True)[3])
        self.assertNotIn('all_metrics_progress',feedback.private_metrics(self.mixed(final=True)[3],beat_spec_path=ROOT/'beat_spec.json'))
    def test_missing_storage_decision_rejected(self):
        result=self.grade(1572864);del result[3]['beat_spec_workload_passes'][0]['metrics']['logical_storage_bits']
        with self.assertRaises(ValueError):feedback.private_metrics(result[3],beat_spec_path=ROOT/'beat_spec.json')
    def test_false_storage_decision_cannot_claim_pass(self):
        result=self.grade(1572864);result[3]['beat_spec_workload_passes'][0]['metrics']['logical_storage_bits']=False
        with self.assertRaises(ValueError):feedback.private_metrics(result[3],beat_spec_path=ROOT/'beat_spec.json')

if __name__=='__main__':unittest.main()
