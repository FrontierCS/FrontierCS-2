"""tasks/template/Theory/template_task stays a valid, working task.

The template is what new tasks copy, so a format or contract change that breaks it must fail
here: it passes the layout validator, and its judge, run through the shared verifier exactly as
the Verifier image runs it, scores the reference at about 50 and a wrong answer at 0 in both
modes, with the private final metrics present.
"""

from __future__ import annotations

import deployment_test_guard  # noqa: F401  # isort: skip

import difflib
import io
import json
import shutil
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "tasks/shared/verifier"))

import task_manifest  # noqa: E402
import validate_task_layout  # noqa: E402

TEMPLATE = REPO / "tasks/template/Theory/template_task"
WRONG = """\
--- a/src/primes.c
+++ b/src/primes.c
@@ -15,5 +15,5 @@ uint64_t count_primes(uint32_t n) {
     uint64_t count = 0;
     for (uint32_t x = 2; x <= n && x != 0; x++)
         count += (uint64_t)is_prime(x);
-    return count;
+    return count + 1;
 }
"""


def payload(patch: str, directory: Path) -> Path:
    body = patch.encode()
    path = directory / "payload.tar"
    with tarfile.open(path, "w", format=tarfile.USTAR_FORMAT) as archive:
        folder = tarfile.TarInfo("payload")
        folder.type, folder.mode = tarfile.DIRTYPE, 0o755
        archive.addfile(folder)
        entry = tarfile.TarInfo("payload/solution.patch")
        entry.size, entry.mode = len(body), 0o644
        archive.addfile(entry, io.BytesIO(body))
    return path


@unittest.skipUnless(shutil.which("cc") and shutil.which("git"), "needs cc and git")
class TemplateTask(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the shared modules from tasks/shared, not whatever copy an earlier test of
        # another task left in sys.modules: the Verifier is a fresh process in production.
        for name in ("fcs2_verifier", "beat_spec_feedback", "edit_scope"):
            sys.modules.pop(name, None)
        sys.path.insert(0, str(REPO / "tasks/shared/verifier"))
        import fcs2_verifier

        cls.verifier = fcs2_verifier
        cls.scratch = Path(tempfile.mkdtemp(prefix="template-task-test-"))
        # The Verifier image's /opt/frontier/task: judge/, the generated config and the
        # starter at pristine/.
        cls.bundle = cls.scratch / "task"
        shutil.copytree(task_manifest.judge_bundle(TEMPLATE), cls.bundle)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.scratch, ignore_errors=True)

    def run_verifier(self, patch: str, mode: str) -> dict:
        work = Path(tempfile.mkdtemp(dir=self.scratch))
        result = work / "result.json"
        self.verifier.verify(
            bundle=self.bundle,
            entrypoint="payload/solution.patch",
            payload=payload(patch, work),
            dataset=TEMPLATE / "datasets" / f"{mode}.json",
            mode=mode,
            result_path=result,
            evidence_path=work / "evidence.tar.zst",
        )
        outcome = json.loads(result.read_text())
        self.assertEqual(outcome["status"], "passed", outcome.get("summary"))
        return outcome

    def test_layout_is_valid(self):
        self.assertEqual(validate_task_layout.validate_task(TEMPLATE), "patch-workspace")

    def test_agent_image_cannot_carry_the_definition_or_the_judge(self):
        # The Agent must never read task.toml, its generated effective definition, or anything
        # of the judge: the layout check refuses an Agent Dockerfile that copies them.
        for source in ("task.toml", ".frontier-generated/effective-task.toml",
                       ".frontier-generated/judge/config.yaml", "judge", "docker"):
            with self.subTest(source=source):
                root = Path(tempfile.mkdtemp(dir=self.scratch)) / "Theory" / "template_task"
                shutil.copytree(TEMPLATE, root)
                recipe = root / "docker/agent.Dockerfile"
                text = recipe.read_text()
                self.assertIn("COPY repo /frontier/task/repo", text)
                recipe.write_text(text.replace(
                    "COPY repo /frontier/task/repo",
                    f"COPY repo /frontier/task/repo\nCOPY {source} /frontier/task/leak"))
                with self.assertRaises(validate_task_layout.LayoutFailure):
                    validate_task_layout.check_native_dockerfile(root, "agent-base")

    def test_statement_may_not_restate_platform_numbers(self):
        for sentence in ("You have 18000 seconds.", "There are 20 train calls.",
                         "The archive may hold 262,144 bytes."):
            with self.subTest(sentence=sentence):
                root = Path(tempfile.mkdtemp(dir=self.scratch)) / "Theory" / "template_task"
                shutil.copytree(TEMPLATE, root)
                statement = root / "repo/README.md"
                statement.write_text(statement.read_text() + "\n" + sentence + "\n")
                with self.assertRaises(validate_task_layout.LayoutFailure):
                    validate_task_layout.check_no_platform_numbers(root)

    def test_definition_round_trips(self):
        policy = task_manifest.load_policy(TEMPLATE)
        self.assertEqual(policy["problem_id"], "template_task")
        generated = task_manifest.generated_files(TEMPLATE)
        config = json.loads(generated[".frontier-generated/judge/config.yaml"])
        self.assertEqual(config["edit_scope"]["allowed_paths"], ["src/primes.c"])
        agent = json.loads(generated[".frontier-generated/agent/submission-policy.json"])
        self.assertNotIn("forbidden_added", agent["edit_scope"])  # content rules stay judge-only

    def test_reference_scores_about_fifty(self):
        reference = (TEMPLATE / "judge/reference.patch").read_text()
        train = self.run_verifier(reference, "train")
        self.assertEqual(train["status"], "passed")
        self.assertAlmostEqual(train["raw_score"], 50.0, delta=15.0)
        self.assertEqual(train["metrics"]["beat_spec_progress"]["wall_seconds"]["total"], 3)
        final = self.run_verifier(reference, "eval")
        self.assertEqual(final["private_metrics"]["beat_spec_progress"]["wall_seconds"]["total"], 6)
        self.assertNotIn("beat_spec_progress", final["metrics"])

    def test_out_of_scope_patch_is_a_shared_rejection_before_the_judge_runs(self):
        before = (TEMPLATE / "repo/bench/main.c").read_text().splitlines(keepends=True)
        outside = "".join(difflib.unified_diff(before, ["/* x */\n", *before],
                                               "a/bench/main.c", "b/bench/main.c"))
        for mode in ("train", "eval"):
            result = self.run_verifier(outside, mode)
            self.assertEqual((result["status"], result["raw_score"]), ("passed", 0.0), mode)
            self.assertEqual(result["metrics"]["rejected"], "out_of_scope", mode)
            self.assertIn("bench/main.c", result["summary"])

    def test_wrong_answer_is_a_scored_zero_in_both_modes(self):
        for mode in ("train", "eval"):
            result = self.run_verifier(WRONG, mode)
            self.assertEqual((result["status"], result["raw_score"]), ("passed", 0.0), mode)
        self.assertEqual(result["private_metrics"]["beats_reference"], 0)


    def copy_template(self) -> Path:
        root = Path(tempfile.mkdtemp(dir=self.scratch)) / "Theory" / "template_task"
        shutil.copytree(TEMPLATE, root)
        return root

    def test_feedback_table_is_checked(self):
        # task.toml [feedback]: lower_snake_case names, never private or shared names, each a
        # non-empty sentence.
        for table in ('"Bad-Name" = "x"', 'reference_seconds = "x"', 'margin_total = "x"',
                      'beats_count = "x"', 'score_unbounded = "x"', 'beat_spec_progress = "x"',
                      'beat_spec_workload_passes = "x"', 'rejected = "x"', 'solved = ""',
                      'solved = 3', 'solved = "  "'):
            with self.subTest(table=table):
                root = self.copy_template()
                manifest = root / "task.toml"
                manifest.write_text(manifest.read_text() + f"\n[feedback]\n{table}\n")
                with self.assertRaises(task_manifest.ManifestError):
                    task_manifest.read_task(root)

    def test_feedback_names_reach_config_readme_and_definition_in_order(self):
        root = self.copy_template()
        manifest = root / "task.toml"
        manifest.write_text(manifest.read_text() + (
            '\n[feedback]\nwall_seconds_replicates_by_workload = "Candidate and reference wall times of every replicate."\n'
            'wall_seconds_candidate_by_workload = "Candidate wall time per workload, in seconds."\n'))
        names = ["wall_seconds_replicates_by_workload", "wall_seconds_candidate_by_workload"]
        task = task_manifest.read_task(root)
        self.assertEqual(task_manifest.judge_config(task)["evaluation"]["solver_visible_metrics"],
                         names)
        policy = task_manifest.load_policy(root)
        self.assertEqual(policy["metadata"]["evaluation"]["solver_visible_metrics"], names)
        readme = task_manifest.agent_readme(root)
        self.assertIn("- `wall_seconds_candidate_by_workload`: Candidate wall time per workload, in seconds.",
                      readme)
        self.assertNotIn("No task-specific metrics", readme)
        self.assertIn("No task-specific metrics", task_manifest.agent_readme(TEMPLATE))
        self.assertIn('wall_seconds_candidate_by_workload = "Candidate wall time per workload, in seconds."',
                      task_manifest.effective_toml(root))

    def test_a_task_toml_task_has_no_beat_spec(self):
        root = self.copy_template()
        (root / "judge/beat_spec.json").write_text('{"solver_feedback": []}\n')
        with self.assertRaisesRegex(validate_task_layout.LayoutFailure,
                                    "beat_spec.json is removed"):
            validate_task_layout.check_task_feedback(
                root, task_manifest.load_policy(root)["metadata"]["evaluation"])

    def test_feedback_metric_must_be_named_by_the_evaluator(self):
        root = self.copy_template()
        manifest = root / "task.toml"
        manifest.write_text(manifest.read_text() + '\n[feedback]\nsolved_count = "Solved."\n')
        with self.assertRaisesRegex(validate_task_layout.LayoutFailure, "solved_count"):
            validate_task_layout.check_task_feedback(
                root, task_manifest.load_policy(root)["metadata"]["evaluation"])

    def test_evaluator_needs_its_scoring_note(self):
        root = self.copy_template()
        evaluator = root / "judge/evaluator.py"
        text = evaluator.read_text()
        evaluator.write_text(text[text.index('"""', 3) + 3:])
        with self.assertRaisesRegex(validate_task_layout.LayoutFailure, "scoring note"):
            validate_task_layout.check_task_feedback(
                root, task_manifest.load_policy(root)["metadata"]["evaluation"])

    def test_feedback_probe_rejects_a_leaking_shared_helper(self):
        shared = REPO / "tasks/shared/verifier/beat_spec_feedback.py"
        validate_task_layout.check_shared_feedback(shared, ["solved_count"])
        leaky = self.scratch / "leaky_feedback.py"
        leaky.write_text(shared.read_text() + """

_strict_public_metrics = public_metrics


def public_metrics(metrics, *, visible):
    result = _strict_public_metrics(metrics, visible=visible)
    result.update({k: v for k, v in metrics.items() if k == "private_probe_detail"})
    return result
""")
        with self.assertRaisesRegex(validate_task_layout.LayoutFailure, "private fields"):
            validate_task_layout.check_shared_feedback(leaky, ["solved_count"])

    def test_train_returns_exactly_the_config_allowlist(self):
        # The shared verifier returns the evaluator metrics config.yaml
        # evaluation.solver_visible_metrics (task.toml [feedback]) names, never a private one,
        # and nothing else besides score_unbounded and beat_spec_progress.
        reference = (TEMPLATE / "judge/reference.patch").read_text()
        default = self.run_verifier(reference, "train")
        self.assertEqual(set(default["metrics"]), {"score_unbounded", "beat_spec_progress"})
        self.assertEqual(self.with_allowlist(
            ["wall_seconds_total", "reference_score", "margin"],
            lambda: set(self.run_verifier(reference, "train")["metrics"])),
            {"score_unbounded", "beat_spec_progress", "wall_seconds_total"})
        # A named value that carries private workload identities fails closed.
        with self.assertRaises(self.verifier.VerifierFailure):
            self.with_allowlist(["wall_seconds_candidate_by_workload"],
                                lambda: self.run_verifier(reference, "train"))

    def with_allowlist(self, names, action):
        config_path = self.bundle / "config.yaml"
        original = config_path.read_text()
        try:
            config = json.loads(original)
            config["evaluation"]["solver_visible_metrics"] = names
            config_path.write_text(json.dumps(config))
            return action()
        finally:
            config_path.write_text(original)

    def test_missing_allowlist_is_a_configuration_failure(self):
        config_path = self.bundle / "config.yaml"
        original = config_path.read_text()
        try:
            config = json.loads(original)
            del config["evaluation"]["solver_visible_metrics"]
            config_path.write_text(json.dumps(config))
            with self.assertRaises(self.verifier.VerifierFailure):
                self.run_verifier(WRONG, "train")
        finally:
            config_path.write_text(original)

    def test_metrics_cap_comes_from_task_toml(self):
        task = task_manifest.read_task(TEMPLATE)
        self.assertEqual(task["verifier"]["max_metrics_bytes"], 524288)  # explicit default
        self.assertEqual(task_manifest.judge_config(task)["evaluation"]["max_metrics_bytes"],
                         524288)
        # The shared verifier's own cap, not a platform field of the definition.
        self.assertNotIn("max_metrics_bytes", task_manifest.load_policy(TEMPLATE)["verifier"])
        root = self.copy_template()
        manifest = root / "task.toml"
        manifest.write_text(manifest.read_text().replace(
            "[verifier]\n", "[verifier]\nmax_metrics_bytes = 3145728\n", 1))
        config = json.loads(task_manifest.generated_files(root)[
            ".frontier-generated/judge/config.yaml"])
        self.assertEqual(config["evaluation"]["max_metrics_bytes"], 3145728)
        manifest.write_text(manifest.read_text().replace(
            "max_metrics_bytes = 3145728", "max_metrics_bytes = 0"))
        with self.assertRaises(task_manifest.ManifestError):
            task_manifest.read_task(root)

    def test_verifier_enforces_the_configured_metrics_cap(self):
        reference = (TEMPLATE / "judge/reference.patch").read_text()
        config_path = self.bundle / "config.yaml"
        original = config_path.read_text()
        try:
            config = json.loads(original)
            config["evaluation"]["max_metrics_bytes"] = 64
            config_path.write_text(json.dumps(config))
            with self.assertRaisesRegex(self.verifier.VerifierFailure, "metrics exceed"):
                self.run_verifier(reference, "train")
        finally:
            config_path.write_text(original)

    def test_only_verifier_recipes_copy_from_digest_pinned_private_images(self):
        pinned = "docker.io/joyemang33/fcs2-private-traces@sha256:" + "0" * 64
        cases = (
            ("verifier", f"COPY --from={pinned} /traces /opt/frontier/task/traces", True),
            ("verifier", "COPY --from=docker.io/joyemang33/fcs2-private-traces:v1 /t /t", False),
            ("verifier", "COPY --from=fcs2-private-traces@sha256:" + "0" * 64 + " /t /t", False),
            ("agent-base", f"COPY --from={pinned} /traces /frontier/task/traces", False),
        )
        for role, line, allowed in cases:
            with self.subTest(role=role, line=line):
                root = self.copy_template()
                recipe = task_manifest.layout(root).dockerfile(role)
                recipe.write_text(recipe.read_text() + line + "\n")
                if allowed:
                    validate_task_layout.check_native_dockerfile(root, role)
                else:
                    with self.assertRaises(validate_task_layout.LayoutFailure):
                        validate_task_layout.check_native_dockerfile(root, role)

    def test_final_zero_without_beat_spec_fields_records_beats_reference_zero(self):
        # A validity failure scored zero by an evaluator that omits the beat-spec fields still
        # records beats_reference = 0 in the final result, as a rejection does.
        bundle = Path(tempfile.mkdtemp(dir=self.scratch)) / "task"
        shutil.copytree(self.bundle, bundle)
        (bundle / "evaluator.py").write_text(
            '"""Stub."""\n\ndef evaluate(tree):\n'
            '    return 0.0, 0.0, "invalid output", {"invalid_rows": 1}\n')
        work = Path(tempfile.mkdtemp(dir=self.scratch))
        reference = (TEMPLATE / "judge/reference.patch").read_text()
        for mode, expected in (("eval", {"beats_reference": 0}),):
            result_path = work / f"{mode}.json"
            self.verifier.verify(bundle=bundle, entrypoint="payload/solution.patch",
                                 payload=payload(reference, work),
                                 dataset=TEMPLATE / "datasets" / f"{mode}.json", mode=mode,
                                 result_path=result_path,
                                 evidence_path=work / f"{mode}.tar.zst")
            result = json.loads(result_path.read_text())
            self.assertEqual((result["status"], result["raw_score"]), ("passed", 0.0))
            self.assertEqual(result.get("private_metrics"), expected, mode)

    def test_statement_follows_the_statement_template(self):
        validate_task_layout.check_statement_structure(TEMPLATE)
        statement = (TEMPLATE / "repo/README.md").read_text()
        broken = {
            "an extra section": statement + "\n## Calibration\nText.\n",
            "a second title": statement + "\n# Another title\n",
            "a missing section": statement.replace("## Correctness\n", ""),
            "sections out of order": statement.replace("## Task\n", "## Interface\n", 1)
                                                .replace("## Interface\n`src", "## Task\n`src"),
            "optional sections out of order": statement + "\n## Running locally\nAgain.\n",
            "too many words": statement + "\n## Notes\n" + "word " * 1600 + "\n",
            "no beats-reference paragraph": statement.replace("**Beats the reference**",
                                                              "Beats the reference"),
        }
        for reason, text in broken.items():
            with self.subTest(reason=reason):
                root = self.copy_template()
                (root / "repo/README.md").write_text(text)
                with self.assertRaises(validate_task_layout.LayoutFailure):
                    validate_task_layout.check_statement_structure(root)
        # A '# ' line inside a code block is not a title.
        root = self.copy_template()
        (root / "repo/README.md").write_text(statement + "\n## Notes\n```sh\n# a comment\n```\n")
        validate_task_layout.check_statement_structure(root)

    def test_task_readme_has_no_scoring_or_objective_section(self):
        for title in ("Scoring", "Objective", "Optimization priority"):
            with self.subTest(title=title):
                root = self.copy_template()
                readme = root / "README.md"
                readme.write_text(readme.read_text() + f"\n## {title}\nText.\n")
                with self.assertRaises(validate_task_layout.LayoutFailure):
                    validate_task_layout.check_statement_structure(root)

    def test_generated_readme_must_name_every_feedback_metric(self):
        from unittest import mock
        root = self.copy_template()
        manifest = root / "task.toml"
        manifest.write_text(manifest.read_text()
                            + '\n[feedback]\nwall_seconds_total = "Sum of candidate times."\n')
        evaluation = task_manifest.load_policy(root)["metadata"]["evaluation"]
        validate_task_layout.check_task_feedback(root, evaluation)
        real = task_manifest.agent_readme
        with mock.patch.object(task_manifest, "agent_readme",
                               lambda r, t=None: real(r, t).replace("`wall_seconds_total`", "")):
            with self.assertRaisesRegex(validate_task_layout.LayoutFailure, "wall_seconds_total"):
                validate_task_layout.check_task_feedback(root, evaluation)

    def test_generated_readme_states_the_file_rules(self):
        readme = task_manifest.agent_readme(TEMPLATE)
        self.assertIn("The patch may change at most 50 files. It may not delete files.\n", readme)
        self.assertNotIn("add new files", readme)
        self.assertIn("The judge evaluates your submission with 2 CPU cores and 4 GiB of\n  memory.",
                      readme)
        root = self.copy_template()
        manifest = root / "task.toml"
        manifest.write_text(manifest.read_text().replace(
            'allowed_paths = ["src/primes.c"]',
            'allowed_paths = ["src/primes.c"]\nmax_changed_files = 1\nallow_delete = true\n'
            'allow_new_files = false', 1))
        readme = task_manifest.agent_readme(root)
        self.assertIn("The patch may change at most 1 file. It may not add new files.\n", readme)
        self.assertNotIn("delete files", readme)
        self.assertNotIn("rename", readme)
        self.assertNotIn("allow_rename",
                         task_manifest.generated_files(root)[
                             f"{task_manifest.GENERATED}/agent/submission-policy.json"].decode())

    def test_allow_rename_reaches_the_pre_check_the_judge_and_the_readme(self):
        root = self.copy_template()
        manifest = root / "task.toml"
        manifest.write_text(manifest.read_text().replace(
            'allowed_paths = ["src/primes.c"]',
            'allowed_paths = ["src/primes.c"]\nallow_rename = true', 1))
        task = task_manifest.read_task(root)
        self.assertTrue(task_manifest.edit_scope_rules(task, agent=True)["allow_rename"])
        self.assertTrue(task_manifest.judge_config(task)["edit_scope"]["allow_rename"])
        self.assertIn("It may not delete files. It may rename a file when both the old and the "
                      "new path are editable.", " ".join(task_manifest.agent_readme(root).split()))

    def test_agent_gpu_recipe_copies_only_from_the_tasks_own_agent_images(self):
        from unittest import mock
        digest = "@sha256:" + "0" * 64
        own = "docker.io/joyemang33/fcs2-task-template_task-agent"
        cases = (
            ("agent-gpu", own + digest, True),
            ("agent-gpu", own + "-gpu" + digest, True),
            ("agent-gpu", "docker.io/joyemang33/fcs2-task-verifier-template_task" + digest, False),
            ("agent-gpu", "docker.io/joyemang33/fcs2-task-other_task-agent" + digest, False),
            ("agent-gpu", own + ":v1", False),
            ("agent-base", own + digest, False),
        )
        for role, stage, allowed in cases:
            with self.subTest(role=role, stage=stage):
                root = self.copy_template()
                recipe = root / "docker" / f"{role.replace('-base', '')}.Dockerfile"
                if role == "agent-gpu":
                    recipe.write_text((root / "docker/agent.Dockerfile").read_text())
                recipe.write_text(recipe.read_text() + f"COPY --from={stage} /data /data\n")
                with mock.patch.object(validate_task_layout, "copy_source",
                                       lambda root, source, role: source):
                    if allowed:
                        validate_task_layout.check_native_dockerfile(root, role)
                    else:
                        with self.assertRaises(validate_task_layout.LayoutFailure):
                            validate_task_layout.check_native_dockerfile(root, role)

    def test_patch_cap_defaults_to_the_archive_cap_and_may_be_smaller(self):
        task = task_manifest.read_task(TEMPLATE)
        self.assertEqual(task["submission"]["max_patch_bytes"], 262144)
        self.assertIn("default: submission.max_bytes", task_manifest.effective_toml(TEMPLATE))
        self.assertIn("`payload/solution.patch`,\n  at most 262,144 bytes;",
                      task_manifest.agent_readme(TEMPLATE))
        root = self.copy_template()
        manifest = root / "task.toml"
        manifest.write_text(manifest.read_text().replace(
            "max_bytes = 262144 ", "max_bytes = 2097152\nmax_patch_bytes = 2000000 ", 1))
        task = task_manifest.read_task(root)
        files = task_manifest.generated_files(root)
        for name in ("judge/config.yaml", "agent/submission-policy.json"):
            rules = json.loads(files[f".frontier-generated/{name}"])["edit_scope"]
            self.assertEqual(rules["max_patch_bytes"], 2000000, name)
        self.assertEqual(task_manifest.load_policy(root)["submission"]["max_bytes"], 2097152)
        self.assertIn("the patch at most 2,000,000 bytes, the tar at most 2,097,152 bytes;",
                      task_manifest.agent_readme(root))
        manifest.write_text(manifest.read_text().replace("max_patch_bytes = 2000000",
                                                         "max_patch_bytes = 3000000"))
        with self.assertRaises(task_manifest.ManifestError):
            task_manifest.read_task(root)

    def test_gpu_section_only_with_an_agent_gpu_budget(self):
        root = self.copy_template()
        manifest = root / "task.toml"
        base = manifest.read_text()
        for instances, expected in ((0, False), (1, True)):
            with self.subTest(max_instances=instances):
                manifest.write_text(base + (
                    "\n[gpu.agent]\n"
                    f'allowed_types = ["H100"]\nmax_instances = {instances}\n'
                    f"max_instance_seconds = {instances * 3600}\nmax_gpu_seconds = {instances * 3600}\n"))
                gpu_text = (REPO / "tasks/shared/agent/README.gpu.md").read_text().splitlines()[0]
                self.assertEqual(gpu_text in task_manifest.agent_readme(root), expected)

if __name__ == "__main__":
    unittest.main()
