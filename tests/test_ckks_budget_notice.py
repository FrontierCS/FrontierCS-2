"""The qualified CKKS public prose must preserve both exact timing boundaries."""
import deployment_test_guard  # noqa: F401
import tempfile
import unittest
from pathlib import Path
from test_answering_budgets import declares_answering_and_independent_final

ENTRY = """The task statement is at `/frontier/task/repo/README.md`.
The task's configured per-evaluation timeout is **600 seconds**, and the overall answering window is **18000 seconds**.
Agent work and train share that answering window; their limits are not additive.
Final evaluation has a separate **600-second** allowance; it does not consume train calls or the train wall-time budget.
"""
STATEMENT = """The inclusive answering window is 18000 seconds (five hours), including development verification.
The independent 600-second final evaluation starts after Agent exit and is outside that answering window.
"""

class CkksBudgetNoticeTests(unittest.TestCase):
    def test_exact_linked_contract_and_changed_limits(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / 'ckks_composable_noise'
            entry = root / 'agent-base/frontier/task/README.md'
            statement = entry.parent / 'repo/README.md'
            statement.parent.mkdir(parents=True)
            entry.write_text(ENTRY); statement.write_text(STATEMENT)
            self.assertTrue(declares_answering_and_independent_final(root, 600))
            self.assertFalse(declares_answering_and_independent_final(root, 601))
            for old,new in [('18000','1800'),('600','60'),('their limits are not additive','their limits are additive'),('/frontier/task/repo/README.md','other.md')]:
                entry.write_text(ENTRY.replace(old,new))
                with self.subTest(entry=old):self.assertFalse(declares_answering_and_independent_final(root,600))
            entry.write_text(ENTRY)
            for old,new in [('18000','1800'),('600','60'),('after Agent exit','before Agent exit'),('outside that answering window','inside that answering window')]:
                statement.write_text(STATEMENT.replace(old,new))
                with self.subTest(statement=old):self.assertFalse(declares_answering_and_independent_final(root,600))
            statement.write_text(STATEMENT);entry.write_text(ENTRY.replace('repo/README.md','README.md'))
            self.assertFalse(declares_answering_and_independent_final(root,600))
