"""The public catalog contains formal releases, never short acceptance copies."""

#
import deployment_test_guard  # noqa: F401  # isort: skip

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import render_task_definition as renderer


def declares_answering_and_independent_final(root, final_seconds):
    """Accept the same contract in the entry or its explicitly linked guide."""
    entry = (root / "agent-base/frontier/task/README.md").read_text()
    if all(
        part in " ".join(entry.replace("18,000 seconds", "18000 seconds").split())
        for part in (
            "18000 seconds (five hours)",
            "does not consume the answering window",
            f"{final_seconds}-second",
        )
    ):
        return True
    # This native contract states that final is measured independently after
    # the Agent exits, with the exact policy timeout. Comma punctuation does
    # not change the five-hour numeric answering budget.
    if (
        f"Then exit. Formal answering lasts 18,000 seconds and includes train waits; "
        f"final is independently measured with {final_seconds} seconds." in entry
    ):
        return True
    # Preserve the repaired Babble PR's wrapped native statement verbatim.
    if (
        f"Independent final evaluation starts after Agent/train cleanup, with its own "
        f"{final_seconds}-second budget. The formal answering budget is 18000 seconds "
        "(five hours)." in " ".join(entry.split())
    ):
        return True
    # Preserve the native Hacking statement, including its wrapped prose.
    # Both numeric budgets and the independent post-answering boundary matter.
    normalized_entry = " ".join(
        entry.replace("18,000 seconds", "18000 seconds").split()
    )
    if all(
        part in normalized_entry
        for part in (
            "You have 5 hours (18000 seconds) for answering.",
            f"Final verification has its own {final_seconds}-second budget after answering.",
            "The final budget is independent of the answering window;",
        )
    ):
        return True
    # 中文题面同样必须明确作答秒数、独立 final 秒数和退出后的评分边界。
    if root.name == "hacking_blind" and all(
        part in normalized_entry
        for part in (
            "作答时限 18,000 秒；",
            f"最终验证有独立的 {final_seconds} 秒预算。",
            "正式评分在 Agent 退出后开始。",
        )
    ):
        return True
    if root.name == "hacking_blind" and all(
        part in normalized_entry for part in (
            "作答18000秒，", f"独立 final{final_seconds}秒；",
            "唯一独立 final 在 Agent 退出后执行，结果不回流作答上下文。",
        )
    ):
        return True
    # QuarterPIR's qualified native entry explicitly names both clocks and
    # puts final scoring after exit. Preserve the measured Agent image bytes.
    if all(part in normalized_entry for part in (
        "The answering window is 18000 seconds, including development evaluations.",
        f"Final verification has its own independent {final_seconds:,}-second timeout.",
        "exactly once, and exit. The final score is produced after you exit;",
    )):
        return True
    # Opal states the same independent post-answering limit with Markdown
    # emphasis. Keep the full sentence, both exact budgets and all boundaries.
    if (
        f"The answering window is **5 hours (18000 seconds)**. "
        f"Independent final evaluation starts after the answering window "
        f"and has its own **{final_seconds}-second** limit." in normalized_entry
    ):
        return True
    # The MIPS native entry states both exact budgets and places independent
    # scoring after Agent cleanup. Preserve its already published source bytes.
    if (
        f"Formal answering lasts 18000 seconds (5 hours), including preparation "
        f"and train waits; independent final has a separate {final_seconds} seconds."
        in normalized_entry
        and "A successful explicit final locks the answer; exit afterward. "
        "Its score is computed independently after Agent cleanup and is not "
        "printed by the submission command."
        in normalized_entry
    ):
        return True
    if (
        f"The formal Agent answering budget is **5 hours (18000 seconds)**. "
        f"Independent final evaluation has a separate **{final_seconds}-second** "
        "budget after answering ends." in normalized_entry
    ):
        return True
    # MetaOpt uses comma-separated seconds and places independent final after
    # collection and cleanup. Preserve the native task text and exact limits.
    if (
        f"Formal answering is **18000 seconds (5 hours)**. Independent final receives "
        f"{final_seconds:,} seconds after collection and cleanup." in normalized_entry
    ):
        return True
    # FARGO and KAMI state the same numeric budgets and post-answering
    # final boundary in their native entry. Preserve those published bytes.
    if (
        "You have 18000 seconds of answering time," in normalized_entry
        and f"and a separate {final_seconds}-second independent final evaluation budget."
        in normalized_entry
        and "The final score is produced after you exit; the command does not print it."
        in normalized_entry
    ):
        return True
    if (
        "Formal answering lasts 18000 seconds (5 hours), including startup and train waits."
        in normalized_entry
        and f"Independent final has a separate {final_seconds:,} seconds beginning after Agent and train cleanup."
        in normalized_entry
        and "Its score is computed independently after Agent cleanup" in normalized_entry
    ):
        return True
    # LeJEPA spells out the answering budget and independent post-answering
    # timeout in separate sentences. Preserve the frozen native text.
    if (
        "The answering budget is 18000 seconds (5 hours)." in normalized_entry
        and f"Independent final evaluation has its own {final_seconds}-second timeout after answering ends."
        in normalized_entry
        and "The final score is produced after you exit; the command does not print it."
        in normalized_entry
    ):
        return True
    # Preserve Binary Fuse's frozen numeric clocks and independent final boundary.
    if all(part in normalized_entry for part in (
        "You have **100 train calls**, a **12,000-second cumulative train pool**, and "
        "**18000 seconds total answering time**, including development evaluations.",
        f"Each verification and the independent final evaluation have a {final_seconds}-second timeout.",
        "Final has its own clock after answering.",
    )):
        return True
    # CKKS's frozen native entry separates answering, cumulative train and final.
    # Follow its exact public statement link for the explicit post-exit boundary.
    ckks_statement = root / "agent-base/frontier/task/repo/README.md"
    if (
        root.name == "ckks_composable_noise"
        and "The task statement is at `/frontier/task/repo/README.md`." in normalized_entry
        and not ckks_statement.is_symlink()
        and not ckks_statement.parent.is_symlink()
        and ckks_statement.is_file()
        and all(part in normalized_entry for part in (
            "The task's configured per-evaluation timeout is **600 seconds**, and the overall answering window is **18000 seconds**.",
            "Agent work and train share that answering window; their limits are not additive.",
            "Final evaluation has a separate **600-second** allowance; it does not consume train calls or the train wall-time budget.",
        ))
        and final_seconds == 600
    ):
        linked = " ".join(ckks_statement.read_text().split())
        if all(part in linked for part in (
            "The inclusive answering window is 18000 seconds (five hours), including development verification.",
            "The independent 600-second final evaluation starts after Agent exit and is outside that answering window.",
        )):
            return True
    # Edlib declares the exact clocks in its linked immutable statement and
    # the post-exit scoring boundary in the submission entry.
    edlib_statement = root / "agent-base/frontier/task/repo/README.md"
    if (
        root.name == "edlib_exact_alignment"
        and "The task statement is at `/frontier/task/repo/README.md`." in normalized_entry
        and not edlib_statement.is_symlink()
        and not edlib_statement.parent.is_symlink()
        and edlib_statement.is_file()
        and final_seconds == 600
        and all(part in normalized_entry for part in (
            "The task's configured per-evaluation timeout is **600 seconds**, and the overall answering window is **18000 seconds**.",
            "Agent work and train share that answering window; their limits are not additive.",
            "Final evaluation has a separate **600-second** allowance; it does not consume train calls or the train wall-time budget.",
            "exactly once, and exit. The final score is produced after you exit;",
        ))
    ):
        linked = " ".join(edlib_statement.read_text().split())
        if all(part in linked for part in (
            "Answering lasts exactly 18,000 seconds (five hours), including train work.",
            "Final has its own independent 600-second allowance after answering; it does not add to the answering clock.",
        )):
            return True
    # Glenside links its existing public statement, which contains both exact
    # clocks and the independent post-exit boundary. Preserve measured bytes.
    statement = root / "agent-base/frontier/task/repo/README.md"
    if (
        "The task statement is at `/frontier/task/repo/README.md`." in normalized_entry
        and not statement.is_symlink()
        and not statement.parent.is_symlink()
        and statement.is_file()
    ):
        linked = " ".join(statement.read_text().split())
        if all(part in linked for part in (
            "| Answering allowance | 18000 seconds |",
            f"| Independent final evaluation allowance | {final_seconds} seconds |",
            f"Final evaluation has its own {final_seconds}-second timeout after the Agent exits, independent of remaining answering time.",
        )):
            return True
    # Native exports may put their complete budget/lifecycle description in
    # the linked usage guide. Do not rewrite an immutable measured source just
    # to move equivalent declarations into this test's preferred paragraph.
    guide = root / "agent-base/frontier/task/bin/README.md"
    if (
        "/frontier/task/bin/README.md" not in entry
        and "](bin/README.md)" not in entry
        or guide.is_symlink()
        or not guide.is_file()
    ):
        return False
    text = guide.read_text()
    normalized = " ".join(text.split())
    # Some native exports show the complete policy quota as JSON and state
    # the post-exit final boundary in the linked operational entry. Check
    # every authored quota value instead of rewriting that immutable guide.
    import re

    sections = re.findall(r"^### quota\s*\n+```json\n(.*?)\n```", text, re.M | re.S)
    if (
        len(sections) == 1
        and "These are the unchanged task-policy values, not new allowances."
        in normalized
        and "Time values are seconds;" in normalized
        and "The policy's total wall-time cap must not be interpreted as an additional answer-time allowance."
        in normalized
        and "The final score is produced after you exit; the command"
        in normalized_entry
    ):

        def unique_object(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError("duplicate quota field")
                result[key] = value
            return result

        try:
            quota = json.loads(sections[0], object_pairs_hook=unique_object)
            authored = json.loads((root / "policy.json").read_text())["quota"]
            if (
                quota == authored
                and type(quota.get("wall_time_seconds")) is int
                and quota["wall_time_seconds"] == 18000
                and type(quota.get("eval_timeout_seconds")) is int
                and quota["eval_timeout_seconds"] == final_seconds
            ):
                return True
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            pass
    if (
        "The formal answering budget is 18,000 seconds (5 hours)." in normalized
        and f"Independent final has its own {final_seconds}-second budget, beginning "
        "after Agent/train cleanup at verifier provisioning."
        in normalized
        and "Do not infer the final budget from remaining answering time." in normalized
    ):
        return True
    # Native exports also express the same limits as bullets or a table.
    # Require both exact numeric budgets and the independent cleanup boundary.
    if (
        (
            "Formal answering: 18000 seconds," in normalized
            and f"Independent final: {final_seconds} seconds after cleanup,"
            in normalized
        )
        or (
            "Formal answering lasts 18000 seconds;" in normalized
            and f"independent final has a separate {final_seconds}-second budget"
            in normalized
            and "cleanup before independently evaluating the selected archive."
            in normalized
        )
        or (
            "| Answering window | 18,000 seconds (5 hours) |" in normalized
            and (
                f"| Independent final evaluation | {final_seconds} seconds |"
                in normalized
                or (
                    final_seconds % 60 == 0
                    and f"| Independent final evaluation | {final_seconds:,} seconds ({final_seconds // 60} minutes) |"
                    in normalized
                )
                or (
                    final_seconds % 3600 == 0
                    and f"| Independent final evaluation | {final_seconds:,} seconds ({final_seconds // 3600} {'hour' if final_seconds == 3600 else 'hours'}) |"
                    in normalized
                )
            )
            and "Final evaluation has its own budget after collection/cleanup."
            in normalized
        )
    ):
        return True
    return all(
        part in text
        for part in (
            "formal answering 18000 seconds;",
            f"Independent final has {final_seconds} seconds.",
        )
    ) and any(
        part in text
        for part in (
            "Final runs independently after cleanup",
            "Final independently re-evaluates the three public workloads plus the three hidden workloads after cleanup.",
        )
    )



def declares_public_budget(task_name, root, policy):
    """Uniform five-hour answering, preserving Clusterwise train/final clocks."""
    quota = policy["quota"]
    if task_name == "lrr":
        expected = {"wall_time_seconds": 18000, "max_train_calls": 100,
                    "max_train_wall_seconds": 5400, "eval_timeout_seconds": 3600}
        notice = " ".join((root / "agent-base/frontier/task/bin/README.md").read_text().split())
        entry = (root / "agent-base/frontier/task/README.md").read_text()
        return (policy.get("problem_id") == task_name and quota == expected
                and "[the testbed guide](bin/README.md)" in entry
                and all(type(value) is int for value in quota.values())
                and all(part in notice for part in (
                    "**100 train calls**", "**5400 seconds**",
                    "overall answering window is **18000 seconds**.",
                    "Agent work and train share that answering window; their limits are not additive.",
                    "Final evaluation has a separate **3600-second** allowance; it does not consume train calls or the train wall-time budget.",
                    "The final score is produced after you exit;")))
    if task_name == "ckks_composable_noise":
        release = json.loads((root / "judge/judge/release.json").read_text())
        if release.get("mode") == "calibration":
            entry = (root / "agent-base/frontier/task/README.md").read_text()
            return (
                quota == {"wall_time_seconds": 600, "max_train_calls": 1,
                          "max_train_wall_seconds": 60, "eval_timeout_seconds": 7200}
                and release.get("formal_calibration") is None
                and "This private calibration release accepts only the exact reference payload" in entry
                and "it is not dispatched to model evaluations" in entry
            )
    if task_name != "clusterwise_spgemm":
        return quota["wall_time_seconds"] == 18000 and declares_answering_and_independent_final(
            root, quota["eval_timeout_seconds"]
        )
    expected = {
        "wall_time_seconds": 18000,
        "max_train_calls": 3,
        "max_train_wall_seconds": 21600,
        "eval_timeout_seconds": 7200,
    }
    if policy.get("problem_id") != task_name or quota != expected:
        return False
    if any(type(value) is not int for value in quota.values()):
        return False
    entry = root / "agent-base/frontier/task/README.md"
    notice = " ".join(entry.read_text().split())
    # The native entry links the actual solver-facing task statement. Require
    # its fixed path and the entry's exact numeric contract before following it.
    if "The task statement is at `/frontier/task/repo/README.md`." in notice:
        guide = entry.parent / "repo/README.md"
        if guide.is_symlink() or guide.parent.is_symlink() or not guide.is_file():
            return False
        if not all(part in notice for part in (
            "The train-call ceiling is **3 train calls**",
            "The separate cumulative train wall-time budget is **21600 seconds**.",
            "The task's configured per-evaluation timeout is **7200 seconds**, and the overall answering window is **18000 seconds (5 hours)**.",
            "Agent work and train share that answering window; their limits are not additive.",
            "Final evaluation has a separate **7200-second** allowance; it does not consume train calls or the train wall-time budget.",
        )):
            return False
        notice = " ".join(guide.read_text().split())
    return all(part in notice for part in (
        "The answering window is 18,000 seconds (5 hours), including preparation, editing, and waits for development submissions.",
        "At most 3 train calls may consume a cumulative 21,600 seconds within the remaining answering window; a train call cannot extend that window.",
        "Independent final evaluation begins after answering and Agent cleanup and has its own 7,200-second budget.",
        "The final budget does not consume or extend the answering window.",
    ))


class FormalAnsweringBudgetTests(unittest.TestCase):
    def test_linked_native_statement_preserves_exact_clocks_and_exit_boundary(self):
        import tempfile

        entry_text = "The task statement is at `/frontier/task/repo/README.md`."
        notice = (
            "| Answering allowance | 18000 seconds |\n"
            "| Independent final evaluation allowance | 300 seconds |\n"
            "Final evaluation has its own 300-second timeout after the Agent exits, "
            "independent of remaining answering time."
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = root / "agent-base/frontier/task/README.md"
            statement = entry.parent / "repo/README.md"
            statement.parent.mkdir(parents=True)
            entry.write_text(entry_text)
            statement.write_text(notice)
            self.assertTrue(declares_answering_and_independent_final(root, 300))
            self.assertFalse(declares_answering_and_independent_final(root, 301))
            for invalid in (
                notice.replace("18000", "1800"),
                notice.replace("300 seconds", "301 seconds"),
                notice.replace("after the Agent exits", "before the Agent exits"),
                notice.replace("independent of remaining", "bounded by remaining"),
            ):
                statement.write_text(invalid)
                self.assertFalse(declares_answering_and_independent_final(root, 300))
            statement.write_text(notice)
            entry.write_text("No statement link.")
            self.assertFalse(declares_answering_and_independent_final(root, 300))

    def test_fargo_and_kami_native_budgets_preserve_exact_limits_and_final_boundary(self):
        import tempfile

        notices = [
            (900, "You have 18,000 seconds of answering time, 50 train calls, and a "
             "separate 900-second independent final evaluation budget. "
             "The final score is produced after you exit; the command does not print it."),
            (1800, "Formal answering lasts 18,000 seconds (5 hours), including startup "
             "and train waits. Independent final has a separate 1,800 seconds beginning "
             "after Agent and train cleanup. Its score is computed independently after "
             "Agent cleanup and is not printed by the submission command."),
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = root / "agent-base/frontier/task/README.md"
            entry.parent.mkdir(parents=True)
            for final_seconds, notice in notices:
                for valid in (notice, " ".join(notice.split())):
                    entry.write_text(valid)
                    self.assertTrue(declares_answering_and_independent_final(root, final_seconds))
                for invalid in (
                    notice.replace("18,000", "1,800"),
                    notice.replace("900-second", "901-second").replace("1,800 seconds", "1,801 seconds"),
                    notice.replace("independent final", "shared final").replace("Independent final", "Shared final"),
                    notice.replace("after you exit", "while you answer").replace("after Agent and train cleanup", "during train"),
                ):
                    entry.write_text(invalid)
                    self.assertFalse(declares_answering_and_independent_final(root, final_seconds))
                entry.write_text(notice)
                self.assertFalse(declares_answering_and_independent_final(root, final_seconds + 1))

    def test_native_timeout_notice_requires_exact_limits_and_exit_boundary(self):
        import tempfile

        notice = (
            "The answering budget is 18000 seconds (5 hours). "
            "Independent final evaluation has its own 400-second timeout after answering ends. "
            "The final score is produced after you exit; the command does not print it."
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = root / "agent-base/frontier/task/README.md"
            entry.parent.mkdir(parents=True)
            for valid in (notice, notice.replace(" ", "\n")):
                entry.write_text(valid)
                self.assertTrue(declares_answering_and_independent_final(root, 400))
                self.assertFalse(declares_answering_and_independent_final(root, 401))
            for invalid in (
                notice.replace("18000", "1800"),
                notice.replace("5 hours", "7 hours"),
                notice.replace("400-second", "401-second"),
                notice.replace("Independent final", "Final"),
                notice.replace("after answering ends", "during answering"),
                notice.replace("after you exit", "before you exit"),
            ):
                entry.write_text(invalid)
                self.assertFalse(declares_answering_and_independent_final(root, 400))

    def test_native_table_budget_allows_exact_hours(self):
        import tempfile

        notice = (
            "| Answering window | 18,000 seconds (5 hours) |\n"
            "| Independent final evaluation | 3,600 seconds (1 hour) |\n"
            "Final evaluation has its own budget after collection/cleanup."
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = root / "agent-base/frontier/task/README.md"
            guide = root / "agent-base/frontier/task/bin/README.md"
            guide.parent.mkdir(parents=True)
            entry.write_text("Read /frontier/task/bin/README.md for time budgets.")
            for valid in (notice, " ".join(notice.split())):
                guide.write_text(valid)
                self.assertTrue(declares_answering_and_independent_final(root, 3600))
            for invalid in (
                notice.replace("18,000", "1,800"),
                notice.replace("3,600", "3,601"),
                notice.replace("1 hour", "2 hours"),
                notice.replace("collection/cleanup", "collection"),
            ):
                guide.write_text(invalid)
                self.assertFalse(declares_answering_and_independent_final(root, 3600))

    def test_native_json_quota_requires_exact_policy_and_post_exit_final(self):
        import tempfile

        quota = {
            "wall_time_seconds": 18000,
            "eval_timeout_seconds": 3600,
            "max_train_calls": 3,
            "max_train_wall_seconds": 10800,
        }
        notice = (
            "These are the unchanged task-policy values, not new allowances. "
            "Time values are seconds; size values are bytes. "
            "The policy's total wall-time cap must not be interpreted as an additional answer-time allowance.\n"
            "### quota\n\n```json\n" + json.dumps(quota) + "\n```\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = root / "agent-base/frontier/task/README.md"
            guide = root / "agent-base/frontier/task/bin/README.md"
            guide.parent.mkdir(parents=True)
            entry_text = (
                "Read [testbed usage](bin/README.md). "
                "The final score is produced after you exit; the command does not return the score."
            )
            entry.write_text(entry_text)
            (root / "policy.json").write_text(json.dumps({"quota": quota}))
            guide.write_text(notice)
            self.assertTrue(declares_answering_and_independent_final(root, 3600))
            for invalid in (
                notice.replace("18000", "1800"),
                notice.replace("3600", "900"),
                notice.replace("10800", "1080"),
                notice.replace(
                    '"wall_time_seconds": 18000',
                    '"wall_time_seconds": 18000, "wall_time_seconds": 18000',
                ),
                notice.replace("Time values are seconds;", "Time values are minutes;"),
                notice + notice,
            ):
                guide.write_text(invalid)
                self.assertFalse(declares_answering_and_independent_final(root, 3600))
            guide.write_text(notice)
            for invalid in (
                entry_text.replace("after you exit", "while you answer"),
                entry_text.replace("[testbed usage](bin/README.md)", "another guide"),
            ):
                entry.write_text(invalid)
                self.assertFalse(declares_answering_and_independent_final(root, 3600))

    def test_native_table_budget_allows_exact_comma_seconds_and_minutes(self):
        import tempfile

        notice = (
            "| Answering window | 18,000 seconds (5 hours) |\n"
            "| Independent final evaluation | 1,800 seconds (30 minutes) |\n"
            "Final evaluation has its own budget after collection/cleanup."
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = root / "agent-base/frontier/task/README.md"
            guide = root / "agent-base/frontier/task/bin/README.md"
            guide.parent.mkdir(parents=True)
            entry.write_text("Read /frontier/task/bin/README.md for time budgets.")
            for valid in (notice, " ".join(notice.split())):
                guide.write_text(valid)
                self.assertTrue(declares_answering_and_independent_final(root, 1800))
            for invalid in (
                notice.replace("18,000", "1,800"),
                notice.replace("1,800 seconds", "900 seconds"),
                notice.replace("30 minutes", "90 minutes"),
                notice.replace("after collection/cleanup", "during answering"),
            ):
                guide.write_text(invalid)
                self.assertFalse(declares_answering_and_independent_final(root, 1800))

    def test_metaopt_notice_requires_exact_limits_and_independent_cleanup(self):
        import tempfile

        notice = (
            "Formal answering is **18,000 seconds (5 hours)**. Independent final receives\n"
            "3,600 seconds after collection and cleanup."
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = root / "agent-base/frontier/task/README.md"
            entry.parent.mkdir(parents=True)
            for valid in (notice, " ".join(notice.split())):
                entry.write_text(valid)
                self.assertTrue(declares_answering_and_independent_final(root, 3600))
            for invalid in (
                notice.replace("18,000", "1,800"),
                notice.replace("5 hours", "6 hours"),
                notice.replace("3,600", "600"),
                notice.replace("Independent ", ""),
                notice.replace("after collection and cleanup", "during answering"),
            ):
                entry.write_text(invalid)
                self.assertFalse(declares_answering_and_independent_final(root, 3600))

    def test_babble_native_budget_wording_keeps_exact_limits_and_cleanup(self):
        import tempfile

        notice = (
            "Independent final evaluation starts after Agent/train cleanup, with its own\n"
            "600-second budget. The formal answering budget is 18000 seconds (five hours)."
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = root / "agent-base/frontier/task/README.md"
            entry.parent.mkdir(parents=True)
            entry.write_text(notice)
            self.assertTrue(declares_answering_and_independent_final(root, 600))
            for invalid in (
                notice.replace("18000", "1800"),
                notice.replace("600", "900"),
                notice.replace("after Agent/train cleanup", "during answering"),
            ):
                entry.write_text(invalid)
                self.assertFalse(declares_answering_and_independent_final(root, 600))

    def test_svg_native_budgets_require_both_limits_and_independence(self):
        import tempfile

        notice = (
            "The formal Agent answering budget is **5 hours (18000 seconds)**. "
            "Independent final evaluation has a separate **9000-second** "
            "budget after answering ends."
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = root / "agent-base/frontier/task/README.md"
            entry.parent.mkdir(parents=True)
            for valid in (
                notice,
                notice.replace("18000", "18,000"),
                notice.replace(". ", ".\n"),
            ):
                entry.write_text(valid)
                self.assertTrue(declares_answering_and_independent_final(root, 9000))
            for invalid in (
                notice.replace("18000", "1800"),
                notice.replace("5 hours", "6 hours"),
                notice.replace("9000", "900"),
                notice.replace("Independent ", ""),
                notice.replace("after answering ends", "during answering"),
            ):
                entry.write_text(invalid)
                self.assertFalse(declares_answering_and_independent_final(root, 9000))

    def test_every_public_task_preserves_its_approved_answering_and_final_contract(self):
        definitions = json.loads(renderer.DEFINITIONS.read_text())
        for name, root in renderer.catalog_problem_directories(
            renderer.catalog_task_names(), renderer.TASK_ROOT / "problems"
        ).items():
            with self.subTest(task=name):
                policy = json.loads((root / "policy.json").read_text())
                self.assertEqual(definitions[name]["quota"], policy["quota"])
                self.assertTrue(
                    declares_public_budget(name, root, policy)
                )


class LrrApprovedBudgetTests(unittest.TestCase):
    def test_exact_export_clocks_and_linked_public_guide(self):
        source = ROOT / "tasks/frontier-cs-2.0-demo/problems/Security/lrr"
        import copy
        import tempfile
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name in ("README.md", "bin/README.md"):
                target = root / "agent-base/frontier/task" / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes((source / "agent-base/frontier/task" / name).read_bytes())
            policy = json.loads((source / "policy.json").read_text())
            self.assertTrue(declares_public_budget("lrr", root, policy))
            for key in policy["quota"]:
                changed = copy.deepcopy(policy)
                changed["quota"][key] += 1
                self.assertFalse(declares_public_budget("lrr", root, changed))
            guide = root / "agent-base/frontier/task/bin/README.md"
            original = guide.read_text()
            for old, new in (("18000 seconds", "18001 seconds"),
                             ("5400 seconds", "5401 seconds"),
                             ("3600-second", "3601-second"),
                             ("100 train calls", "101 train calls"),
                             ("after you exit", "before you exit")):
                guide.write_text(original.replace(old, new))
                self.assertFalse(declares_public_budget("lrr", root, policy))


class NativeIndependentBudgetNoticeTests(unittest.TestCase):
    def test_edlib_linked_contract_requires_exact_clocks_and_final_boundary(self):
        import tempfile

        entry_text = (
            "The task statement is at `/frontier/task/repo/README.md`. "
            "The task's configured per-evaluation timeout is **600 seconds**, and the overall answering window is **18000 seconds**. "
            "Agent work and train share that answering window; their limits are not additive. "
            "Final evaluation has a separate **600-second** allowance; it does not consume train calls or the train wall-time budget. "
            "exactly once, and exit. The final score is produced after you exit;"
        )
        statement_text = (
            "Answering lasts exactly 18,000 seconds (five hours), including train work. "
            "Final has its own independent 600-second allowance after answering; "
            "it does not add to the answering clock."
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "edlib_exact_alignment"
            entry = root / "agent-base/frontier/task/README.md"
            statement = entry.parent / "repo/README.md"
            statement.parent.mkdir(parents=True)
            entry.write_text(entry_text)
            for valid in (statement_text, statement_text.replace(" ", "\n")):
                statement.write_text(valid)
                self.assertTrue(declares_answering_and_independent_final(root, 600))
                self.assertFalse(declares_answering_and_independent_final(root, 601))
            statement.write_text(statement_text)
            for before, after in (("18000", "18001"), ("600-second", "601-second"),
                                  ("after you exit", "before you exit"),
                                  ("`/frontier/task/repo/README.md`", "`/elsewhere`")):
                with self.subTest(entry_mutation=(before, after)):
                    entry.write_text(entry_text.replace(before, after))
                    self.assertFalse(declares_answering_and_independent_final(root, 600))
            entry.write_text(entry_text)
            for before, after in (("18,000", "18,001"), ("five hours", "six hours"),
                                  ("600-second", "601-second"), ("independent", "shared"),
                                  ("after answering", "during answering"),
                                  ("does not add", "adds")):
                with self.subTest(statement_mutation=(before, after)):
                    statement.write_text(statement_text.replace(before, after))
                    self.assertFalse(declares_answering_and_independent_final(root, 600))
            statement.unlink()
            self.assertFalse(declares_answering_and_independent_final(root, 600))
            outside = Path(directory) / "outside.md"
            outside.write_text(statement_text)
            statement.symlink_to(outside)
            self.assertFalse(declares_answering_and_independent_final(root, 600))

    def test_quarter_native_contract_requires_exact_clocks_and_exit(self):
        import tempfile

        original = (
            "The answering window is 18,000 seconds, including development evaluations. "
            "Final verification has its own independent 1,800-second timeout. "
            "exactly once, and exit. The final score is produced after you exit;"
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = root / "agent-base/frontier/task/README.md"
            entry.parent.mkdir(parents=True)
            entry.write_text(original)
            self.assertTrue(declares_answering_and_independent_final(root, 1800))
            for before, after in (("18,000", "18,001"), ("1,800", "1,801"),
                                  ("independent", "shared"), ("after you exit", "before you exit")):
                entry.write_text(original.replace(before, after))
                self.assertFalse(declares_answering_and_independent_final(root, 1800))

    def test_binary_fuse_native_contract_requires_exact_clocks_and_final_boundary(self):
        import tempfile

        notice = (
            "You have **100 train calls**, a **12,000-second cumulative train pool**, and\n"
            "**18,000 seconds total answering time**, including development evaluations.\n"
            "Each verification and the independent final evaluation have a 180-second timeout.\n"
            "Final has its own clock after answering."
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = root / "agent-base/frontier/task/README.md"
            entry.parent.mkdir(parents=True)
            for valid in (notice, " ".join(notice.split()), notice.replace("18,000", "18000")):
                entry.write_text(valid)
                self.assertTrue(declares_answering_and_independent_final(root, 180))
                self.assertFalse(declares_answering_and_independent_final(root, 181))
            for before, after in (("100 train", "101 train"), ("12,000", "12,001"),
                                  ("18,000", "18,001"), ("180-second", "181-second"),
                                  ("independent final", "shared final"),
                                  ("after answering", "during answering"),
                                  ("its own clock", "the train clock")):
                with self.subTest(mutation=(before, after)):
                    entry.write_text(notice.replace(before, after))
                    self.assertFalse(declares_answering_and_independent_final(root, 180))

    def test_jointree_linked_guide_preserves_exact_clocks_and_cleanup_boundary(self):
        import tempfile

        notice = (
            "The formal answering budget is 18,000 seconds (5 hours). "
            "Independent final has its own 900-second budget, beginning "
            "after Agent/train cleanup at verifier provisioning. "
            "Do not infer the final budget from remaining answering time."
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = root / "agent-base/frontier/task/README.md"
            guide = entry.parent / "bin/README.md"
            guide.parent.mkdir(parents=True)
            entry.write_text("Read the [testbed guide](bin/README.md).")
            guide.write_text(notice)
            self.assertTrue(declares_answering_and_independent_final(root, 900))
            self.assertFalse(declares_answering_and_independent_final(root, 600))
            for before, after in (
                ("18,000", "1,800"),
                ("5 hours", "6 hours"),
                ("after Agent/train cleanup", "before Agent/train cleanup"),
                ("Independent final", "Shared final"),
            ):
                guide.write_text(notice.replace(before, after))
                self.assertFalse(declares_answering_and_independent_final(root, 900))

    def test_mips_notice_keeps_budgets_train_waits_and_post_cleanup_scoring(self):
        import tempfile

        notice = (
            "A successful explicit final locks the answer; exit afterward. "
            "Its score is computed independently after\nAgent cleanup and "
            "is not printed by the submission command.\n\n"
            "Formal answering lasts 18,000 seconds (5 hours), including "
            "preparation and train\nwaits; independent final has a separate "
            "900 seconds. There are 100 train calls."
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = root / "agent-base/frontier/task/README.md"
            entry.parent.mkdir(parents=True)
            for valid in (
                notice,
                notice.replace("18,000", "18000"),
                " ".join(notice.split()),
            ):
                entry.write_text(valid)
                self.assertTrue(declares_answering_and_independent_final(root, 900))
                self.assertFalse(declares_answering_and_independent_final(root, 600))
            for bad in (
                notice.replace("18,000", "1,800"),
                notice.replace("5 hours", "6 hours"),
                notice.replace("900 seconds", "600 seconds"),
                notice.replace("has a separate", "shares"),
                notice.replace("independent final", "final"),
                notice.replace(
                    "including preparation and train\nwaits",
                    "excluding preparation and train waits",
                ),
                notice.replace(
                    "independently after\nAgent cleanup",
                    "independently during Agent cleanup",
                ),
                notice.replace(
                    "independently after\nAgent cleanup",
                    "independently after answering",
                ),
                notice.replace("exit afterward.", "continue answering."),
            ):
                with self.subTest(notice=bad):
                    entry.write_text(bad)
                    self.assertFalse(
                        declares_answering_and_independent_final(root, 900)
                    )

    def test_native_wrapped_notice_keeps_exact_budgets_and_independence(self):
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = root / "agent-base/frontier/task/README.md"
            entry.parent.mkdir(parents=True)
            notice = (
                "You have 5 hours (18,000 seconds) for answering. "
                "Final verification has its own\n120-second budget after answering. "
                "Training remains limited to 50 calls and\n3,600 seconds cumulatively. "
                "The final budget is independent of the answering\nwindow; "
                "plan the candidate's runtime accordingly."
            )
            for valid in (
                notice,
                notice.replace("18,000", "18000"),
                " ".join(notice.split()),
            ):
                entry.write_text(valid)
                self.assertTrue(declares_answering_and_independent_final(root, 120))
                self.assertFalse(declares_answering_and_independent_final(root, 600))
            for bad in (
                notice.replace("5 hours", "6 hours"),
                notice.replace("18,000", "1,800"),
                notice.replace("120-second", "600-second"),
                notice.replace("after answering.", "during answering."),
                notice.replace("has its own", "has a"),
                notice.replace("after answering.", "answering."),
                notice.replace("independent", ""),
                notice.replace("independent of", "part of"),
                notice.replace(
                    "The final budget is independent of the answering\nwindow;", ""
                ),
            ):
                with self.subTest(notice=bad):
                    entry.write_text(bad)
                    self.assertFalse(
                        declares_answering_and_independent_final(root, 120)
                    )

    def test_emphasized_opal_notice_requires_exact_budgets_and_independence(self):
        import tempfile

        notice = (
            "The answering window is **5 hours (18,000 seconds)**. "
            "Independent final evaluation\nstarts after the answering window "
            "and has its own **7200-second** limit."
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = root / "agent-base/frontier/task/README.md"
            entry.parent.mkdir(parents=True)
            for valid in (
                notice,
                notice.replace("18,000", "18000"),
                " ".join(notice.split()),
            ):
                entry.write_text(valid)
                self.assertTrue(declares_answering_and_independent_final(root, 7200))
                self.assertFalse(declares_answering_and_independent_final(root, 120))
            for invalid in (
                notice.replace("5 hours", "6 hours"),
                notice.replace("18,000", "1,800"),
                notice.replace("7200-second", "120-second"),
                notice.replace("Independent ", ""),
                notice.replace(
                    "after the answering window", "during the answering window"
                ),
                notice.replace("after the answering window", ""),
                notice.replace("has its own", "has a"),
                notice.replace("final evaluation", "train evaluation"),
            ):
                with self.subTest(invalid=invalid):
                    entry.write_text(invalid)
                    self.assertFalse(
                        declares_answering_and_independent_final(root, 7200)
                    )

    def test_native_post_exit_notice_requires_exact_budgets_and_independence(self):
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = root / "agent-base/frontier/task/README.md"
            entry.parent.mkdir(parents=True)
            notice = (
                "Then exit. Formal answering lasts 18,000 seconds and includes train waits; "
                "final is independently measured with 900 seconds."
            )
            entry.write_text(notice)
            self.assertTrue(declares_answering_and_independent_final(root, 900))
            for replacement in (
                notice.replace("18,000", "28,000"),
                notice.replace("900 seconds", "600 seconds"),
                notice.replace(
                    "independently measured", "measured in the answering window"
                ),
                notice.replace("Then exit. ", ""),
                "No budget notice.",
            ):
                with self.subTest(notice=replacement):
                    entry.write_text(replacement)
                    self.assertFalse(
                        declares_answering_and_independent_final(root, 900)
                    )
            entry.write_text(notice)
            self.assertFalse(declares_answering_and_independent_final(root, 600))


class LinkedBudgetGuideTests(unittest.TestCase):
    def test_linked_guide_requires_every_budget_and_lifecycle_declaration(self):
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = root / "agent-base/frontier/task/README.md"
            guide = entry.parent / "bin/README.md"
            guide.parent.mkdir(parents=True)
            entry.write_text("See `/frontier/task/bin/README.md` for budgets.")
            declarations = [
                "formal answering 18000 seconds;",
                "Independent final has 1800 seconds.",
                "Final runs independently after cleanup",
            ]
            guide.write_text("\n".join(declarations))
            self.assertTrue(declares_answering_and_independent_final(root, 1800))
            for removed in declarations:
                with self.subTest(missing=removed):
                    guide.write_text("\n".join(x for x in declarations if x != removed))
                    self.assertFalse(
                        declares_answering_and_independent_final(root, 1800)
                    )
            guide.write_text("\n".join(declarations))
            self.assertFalse(declares_answering_and_independent_final(root, 600))
            native_final = (
                "Final independently re-evaluates the three public workloads "
                "plus the three hidden workloads after cleanup."
            )
            native = declarations[:2] + [native_final]
            guide.write_text("\n".join(native))
            self.assertTrue(declares_answering_and_independent_final(root, 1800))
            for change in (
                "after cleanup",
                "independently",
                "plus the three hidden workloads",
            ):
                with self.subTest(native_missing=change):
                    guide.write_text("\n".join(native).replace(change, ""))
                    self.assertFalse(
                        declares_answering_and_independent_final(root, 1800)
                    )
            guide.write_text("\n".join(declarations))
            entry.write_text("See `/frontier/task/bin/OTHER.md` for budgets.")
            self.assertFalse(declares_answering_and_independent_final(root, 1800))
            entry.write_text("No link to the budget guide.")
            self.assertFalse(declares_answering_and_independent_final(root, 1800))
            entry.write_text("See `/frontier/task/bin/README.md` for budgets.")
            guide.unlink()
            self.assertFalse(declares_answering_and_independent_final(root, 1800))
            outside = root / "unlinked-guide.md"
            outside.write_text("\n".join(declarations))
            guide.symlink_to(outside)
            self.assertFalse(declares_answering_and_independent_final(root, 1800))
            entry.write_text(
                "18000 seconds (five hours); independent 1800-second "
                "final does not consume the answering window."
            )
            self.assertTrue(declares_answering_and_independent_final(root, 1800))


class StructuredBudgetGuideTests(unittest.TestCase):
    def test_bullets_table_and_wrapped_prose_keep_exact_budgets_and_cleanup(self):
        import tempfile

        forms = [
            "Formal answering: 18000 seconds, including bootstrap. Independent final: 120 seconds after cleanup, including provisioning.",
            "Formal answering lasts 18000 seconds; independent final has a separate 120-second budget. Driver checks cleanup before independently evaluating the selected archive.",
            "| Answering window | 18,000 seconds (5 hours) |\n| Independent final evaluation | 120 seconds |\nFinal evaluation has its own budget after collection/cleanup.",
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = root / "agent-base/frontier/task/README.md"
            guide = entry.parent / "bin/README.md"
            guide.parent.mkdir(parents=True)
            entry.write_text("See [platform guide](bin/README.md).")
            for form in forms:
                guide.write_text(form)
                self.assertTrue(declares_answering_and_independent_final(root, 120))
                self.assertFalse(declares_answering_and_independent_final(root, 600))
                for bad in (
                    form.replace("18000", "1800").replace("18,000", "1,800"),
                    form.replace("120", "600"),
                    form.replace("cleanup", "answering"),
                ):
                    guide.write_text(bad)
                    self.assertFalse(
                        declares_answering_and_independent_final(root, 120)
                    )



class ClusterwiseApprovedBudgetTests(unittest.TestCase):
    def test_exact_contract_and_counterexamples(self):
        import copy
        import tempfile
        notice = (
            "The answering window is 18,000 seconds (5 hours), including preparation, "
            "editing, and waits for development submissions. At most 3 train calls may "
            "consume a cumulative 21,600 seconds within the remaining answering window; "
            "a train call cannot extend that window. Independent final evaluation begins "
            "after answering and Agent cleanup and has its own 7,200-second budget. "
            "The final budget does not consume or extend the answering window."
        )
        policy = {"problem_id": "clusterwise_spgemm", "quota": {
            "wall_time_seconds": 18000, "max_train_calls": 3,
            "max_train_wall_seconds": 21600, "eval_timeout_seconds": 7200,
        }}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = root / "agent-base/frontier/task/README.md"
            entry.parent.mkdir(parents=True)
            for valid in (notice, notice.replace(". ", ".\n")):
                entry.write_text(valid)
                self.assertTrue(declares_public_budget("clusterwise_spgemm", root, policy))
            for key in policy["quota"]:
                for value in (0, policy["quota"][key] + 1, float(policy["quota"][key]), True):
                    wrong = copy.deepcopy(policy)
                    wrong["quota"][key] = value
                    with self.subTest(quota=key, value=value):
                        self.assertFalse(declares_public_budget("clusterwise_spgemm", root, wrong))
            for before, after in (
                ("18,000", "32,400"), ("5 hours", "9 hours"), ("At most 3", "At most 4"),
                ("21,600", "28,800"), ("7,200", "3,600"), ("cannot extend", "can extend"),
                ("within the remaining", "outside the remaining"), ("Independent final", "Final"),
                ("after answering and Agent cleanup", "during answering"),
                ("has its own", "has a"), ("does not consume or extend", "consumes"),
            ):
                with self.subTest(before=before):
                    entry.write_text(notice.replace(before, after))
                    self.assertFalse(declares_public_budget("clusterwise_spgemm", root, policy))
            entry.write_text(notice)
            self.assertFalse(declares_public_budget("another_task", root, policy))
            wrong = copy.deepcopy(policy)
            wrong["problem_id"] = "another_task"
            self.assertFalse(declares_public_budget("clusterwise_spgemm", root, wrong))
            wrong = copy.deepcopy(policy)
            wrong["quota"]["unreviewed_allowance"] = 1
            self.assertFalse(declares_public_budget("clusterwise_spgemm", root, wrong))
            # The structured contract still requires all separate-clock clauses.
            self.assertFalse(declares_answering_and_independent_final(root, 7200))


    def test_native_linked_statement_requires_exact_entry_and_real_guide(self):
        import tempfile

        policy = {"problem_id": "clusterwise_spgemm", "quota": {
            "wall_time_seconds": 18000, "max_train_calls": 3,
            "max_train_wall_seconds": 21600, "eval_timeout_seconds": 7200,
        }}
        entry_text = (
            "The task statement is at `/frontier/task/repo/README.md`. "
            "The train-call ceiling is **3 train calls**; every accepted request consumes one. "
            "The separate cumulative train wall-time budget is **21600 seconds**. "
            "The task's configured per-evaluation timeout is **7200 seconds**, and the overall answering window is **18000 seconds (5 hours)**. "
            "Agent work and train share that answering window; their limits are not additive. "
            "Final evaluation has a separate **7200-second** allowance; it does not consume train calls or the train wall-time budget."
        )
        statement = (
            "The answering window is 18,000 seconds (5 hours), including preparation, editing, and waits for development submissions. "
            "At most 3 train calls may consume a cumulative 21,600 seconds within the remaining answering window; a train call cannot extend that window. "
            "Independent final evaluation begins after answering and Agent cleanup and has its own 7,200-second budget. "
            "The final budget does not consume or extend the answering window."
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = root / "agent-base/frontier/task/README.md"
            guide = entry.parent / "repo/README.md"
            guide.parent.mkdir(parents=True)
            entry.write_text(entry_text)
            guide.write_text(statement)
            self.assertTrue(declares_public_budget("clusterwise_spgemm", root, policy))
            for old, new in (("/frontier/task/repo/README.md", "/elsewhere/README.md"),
                             ("18000", "32400"), ("7200", "600"), ("21600", "21601"),
                             ("**3 train calls**", "**4 train calls**"), ("not additive", "additive")):
                with self.subTest(entry=old):
                    entry.write_text(entry_text.replace(old, new))
                    self.assertFalse(declares_public_budget("clusterwise_spgemm", root, policy))
            entry.write_text(entry_text)
            for old, new in (("18,000", "18,001"), ("21,600", "21,601"), ("7,200", "7,201"),
                             ("Independent final", "Final"), ("Agent cleanup", "Agent execution"),
                             ("has its own", "has a"), ("does not consume or extend", "extends")):
                with self.subTest(statement=old):
                    guide.write_text(statement.replace(old, new))
                    self.assertFalse(declares_public_budget("clusterwise_spgemm", root, policy))
            guide.unlink()
            self.assertFalse(declares_public_budget("clusterwise_spgemm", root, policy))
            outside = root / "unlinked.md"
            outside.write_text(statement)
            guide.symlink_to(outside)
            self.assertFalse(declares_public_budget("clusterwise_spgemm", root, policy))


class ChineseHackingBudgetNoticeTests(unittest.TestCase):
    def test_requires_exact_clocks_and_post_exit_final(self):
        import tempfile

        notice = (
            "作答时限 18,000 秒；最多 50 次 train。\n"
            "最终验证有独立的 120 秒预算。\n"
            "正式评分在 Agent 退出后开始。"
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "hacking_blind"
            entry = root / "agent-base/frontier/task/README.md"
            entry.parent.mkdir(parents=True)
            entry.write_text(notice)
            self.assertTrue(declares_answering_and_independent_final(root, 120))
            self.assertFalse(declares_answering_and_independent_final(root, 121))
            for old, new in (
                ("18,000", "1,800"),
                ("120", "1200"),
                ("有独立的", "有共享的"),
                ("退出后", "退出前"),
            ):
                with self.subTest(statement=old):
                    entry.write_text(notice.replace(old, new))
                    self.assertFalse(declares_answering_and_independent_final(root, 120))

    def test_native_defense_exact_budgets_and_post_exit_boundary(self):
        import tempfile
        notice = "作答18000秒，最多100次 train；独立 final1800秒；唯一独立 final 在 Agent 退出后执行，结果不回流作答上下文。"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "hacking_blind"
            entry = root / "agent-base/frontier/task/README.md"
            entry.parent.mkdir(parents=True)
            entry.write_text(notice)
            self.assertTrue(declares_answering_and_independent_final(root, 1800))
            self.assertFalse(declares_answering_and_independent_final(root, 180))
            for a, b in (("18000", "1800"), ("独立 final", "共享 final"),
                         ("退出后", "退出前"), ("不回流", "回流")):
                entry.write_text(notice.replace(a, b))
                self.assertFalse(declares_answering_and_independent_final(root, 1800))


if __name__ == "__main__":
    unittest.main()
