"""Regressions for observed scientific scoring defects; no model calls."""
import unittest
from unittest.mock import patch

import evaluator as ev


class ScientificContract(unittest.TestCase):
    def test_disjoint_languages_cannot_count_as_a_repair(self):
        safe = {"pattern": "b", "result": [], "error": None}
        with patch.object(ev, "_run_hunter", return_value=safe):
            result = ev._score_patterns(["a"], ["b"])
        self.assertEqual(result["well_formed_rate"], 1)
        self.assertEqual(result["jaccard"], 0)
        self.assertEqual(result["repair_rate"], 0)

    def test_unchanged_length_has_zero_rli(self):
        safe = {"pattern": "a", "result": [], "error": None}
        with patch.object(ev, "_run_hunter", return_value=safe):
            result = ev._score_patterns(["a"], ["a"])
        self.assertEqual(result["rli"], 0)
        self.assertEqual(result["jaccard"], 1)
        self.assertEqual(result["repair_rate"], 1)

    def test_missing_semantics_cannot_inherit_upstream_perfect_score(self):
        row = {"compilability": [True, True], "length_ratio": [1, 1],
               "normalized-edit-distance": [0, 0], "jaccard": [1, 1],
               "precision": [1, 1], "recall": [1, 1]}
        safe = {"result": [], "error": None}
        result = ev._aggregate_scores(row, [safe, safe], [True, False], [1.0, 0.0])
        for name in ("repair_rate", "jaccard", "precision", "recall", "fidelity"):
            self.assertEqual(result[name], 0.5)

    def test_empty_or_incomplete_semantic_samples_are_invalid(self):
        for texts, results in (([], []), (["a"] * 100, [True] * 99),
                               (["a"] * 100, [None] * 100)):
            self.assertFalse(ev._semantic_evidence_valid({"texts": texts, "results": results}))

    def test_malformed_detector_attack_record_is_unknown(self):
        raw = ("LRR reconstruction ReDoSHunter adapter\nregex: a\nmode: java/s; detector\n"
               "Is attack success: false\nAttack time: 0 (ms)\n"
               "Vulnerability Position: unknown\nAttack String: -\n"
               "Vulnerability Source: unknown\nVulnerability Degree: unknown\n")
        result = ev._parse_hunter_output("a", raw)
        self.assertIsNone(result["result"])
        self.assertTrue(result["error"])

    def test_completed_detector_without_attack_is_undetected(self):
        raw = "LRR reconstruction ReDoSHunter adapter\nregex: a\nmode: java/s; detector\n"
        result = ev._parse_hunter_output("a", raw)
        self.assertEqual(result["result"], [])
        self.assertIsNone(result["error"])


if __name__ == "__main__":
    unittest.main()
