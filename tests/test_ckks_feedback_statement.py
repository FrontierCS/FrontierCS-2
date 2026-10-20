from __future__ import annotations

import deployment_test_guard  # noqa: F401

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import validate_task_layout as layout


class CkksFeedbackStatementTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "ckks_composable_noise"
        self.review = json.loads((ROOT / "scripts/reference/ckks-feedback-statement-reviewed-v1.json").read_text())
        self.sentence = self.review["sentence"]
        self.visible = list(self.review["fields"])
        self.statement = (
            "beat_spec_progress passed failed total pass_percent fail_percent\n" + self.sentence
        )
        for name in self.review["bindings"]:
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(self.sentence if name.endswith("README.md") else "{}")
        self.review["bindings"] = {
            name: layout.digest(self.root / name) for name in self.review["bindings"]
        }
        self.review_path = Path(self.temp.name) / "review.json"
        self.review_path.write_text(json.dumps(self.review))
        mock = patch.dict(layout.FEEDBACK_STATEMENT_REVIEWS,
                          {self.root.name: self.review_path}, clear=True)
        mock.start()
        self.addCleanup(mock.stop)

    def test_bound_plain_language_description_is_accepted(self):
        layout.check_feedback_statement(self.root, self.statement, self.visible)

    def test_literal_feedback_names_keep_the_existing_contract(self):
        layout.check_feedback_statement(self.root.with_name("other"),
                                        self.statement + " " + " ".join(self.visible), self.visible)

    def test_missing_or_changed_scientific_qualifiers_are_rejected(self):
        for old, new in (("Development", "Final"), ("absolute", "relative"),
                         ("mean", "maximum"), ("candidate", "reference"),
                         ("two metrics", "one metric"), ("without private", "with private")):
            with self.subTest(old=old), self.assertRaisesRegex(layout.LayoutFailure, "absolute development"):
                layout.check_feedback_statement(self.root, self.statement.replace(old, new), self.visible)

    def test_missing_progress_field_and_different_visible_field_are_rejected(self):
        with self.assertRaisesRegex(layout.LayoutFailure, "beat-spec feedback"):
            layout.check_feedback_statement(self.root, self.statement.replace("pass_percent", ""), self.visible)
        for fields in (self.visible[:-1], self.visible + ["reference_psnr"], ["other"]):
            with self.subTest(fields=fields), self.assertRaisesRegex(layout.LayoutFailure, "field set"):
                layout.check_feedback_statement(self.root, self.statement, fields)

    def test_source_changes_and_same_prose_on_other_task_require_review(self):
        for name in self.review["bindings"]:
            path = self.root / name
            original = path.read_bytes()
            path.write_bytes(original + b"\n")
            with self.subTest(name=name), self.assertRaisesRegex(layout.LayoutFailure, "binding mismatch"):
                layout.check_feedback_statement(self.root, self.statement, self.visible)
            path.write_bytes(original)
        with self.assertRaisesRegex(layout.LayoutFailure, "beat-spec feedback"):
            layout.check_feedback_statement(self.root.with_name("other"), self.statement, self.visible)

    def test_review_identity_and_binding_inventory_are_required(self):
        for key, value in (("schema", "other"), ("problem_id", "other"), ("bindings", {})):
            changed = dict(self.review)
            changed[key] = value
            self.review_path.write_text(json.dumps(changed))
            with self.subTest(key=key), self.assertRaises(layout.LayoutFailure):
                layout.check_feedback_statement(self.root, self.statement, self.visible)


if __name__ == "__main__":
    unittest.main()
