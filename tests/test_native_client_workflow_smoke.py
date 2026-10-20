# Install repository test isolation before runtime imports.
import deployment_test_guard  # noqa: F401

import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class NativeClientWorkflowSmokeTests(unittest.TestCase):
    def test_source_only_proposal_requires_native_explicit_final(self):
        proposal = json.loads((ROOT / "smoke/client-workflow/proposal.json").read_text())
        self.assertEqual(proposal["status"], "SOURCE_ONLY_NOT_PUBLISHED_OR_REGISTERED")
        self.assertEqual(proposal["submission_authority"], "native-model-client")
        self.assertTrue(proposal["explicit_final_required"])
        self.assertIsNone(proposal["performance_target"])
        self.assertEqual(proposal["train_calls"], 1)
        for key in ("agent_source", "verifier_source", "datasets_source"):
            self.assertTrue((ROOT / proposal[key]).is_dir())

    def test_model_instructions_do_not_ship_submission_automation(self):
        source = ROOT / "smoke/client-workflow/agent-base/frontier/task"
        self.assertEqual([p.name for p in source.iterdir()], ["README.md"])
        instructions = (source / "README.md").read_text()
        for command in ("train", "status", "store-final", "final"):
            self.assertIn("frontier-submit " + command, instructions)
        self.assertIn("Do not optimize", instructions)
        self.assertIn("timeout fallback is insufficient", instructions)
        self.assertIn("Do not call the Driver API directly", instructions)


if __name__ == "__main__":
    unittest.main()
