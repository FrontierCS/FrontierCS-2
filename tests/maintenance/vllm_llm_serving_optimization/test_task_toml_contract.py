"""The task.toml migration (#757) keeps what the platform, the Agent and the judge saw:
the expanded test sandboxes equal the previous policy's sidecars byte for byte, every previous
platform value is unchanged, the Agent's pre-check scope mirrors the judge's patch policy, and
every judge rejection maps to one shared category."""
from vllm_task import REPO_ROOT, TASK_ROOT  # noqa: E402
import fnmatch
import itertools
import json
import random
import sys
import tempfile
import unittest
from pathlib import Path

sys.path[:0] = [str(REPO_ROOT / 'scripts'), str(REPO_ROOT / 'tasks/shared/lib')]
import edit_scope  # noqa: E402
import task_manifest  # noqa: E402
import patch_policy  # noqa: E402
from submission import REJECTION_CATEGORIES, rejection_category  # noqa: E402
import verify  # noqa: E402
from beat_spec_feedback import progress as shared_progress  # noqa: E402
from instance_manifest import expected_groups  # noqa: E402

OLD = json.loads((REPO_ROOT / 'docs/task-authoring/vllm_llm_serving_optimization/provenance/'
                  'policy.json').read_text())
TASK = task_manifest.read_task(TASK_ROOT)
POLICY = task_manifest.load_policy(TASK_ROOT)


class PlatformDefinitionTests(unittest.TestCase):
    def test_test_sandboxes_expand_to_the_previous_sidecars_byte_for_byte(self):
        self.assertEqual(json.dumps(POLICY['sidecars']), json.dumps(OLD['sidecars']))

    def test_every_previous_platform_value_is_unchanged(self):
        for key in ('agent', 'quota', 'submission', 'verifier', 'network', 'gpu',
                    'gpu_parent_resources', 'display_name', 'version', 'problem_id'):
            with self.subTest(key=key):
                self.assertEqual(POLICY[key], OLD[key])
        for key in ('accelerator', 'datasets'):
            self.assertEqual(POLICY['metadata'][key], OLD['metadata'][key])
        evaluation = OLD['metadata']['evaluation']
        self.assertEqual(POLICY['metadata']['evaluation']['solver_visible_metrics'],
                         evaluation['solver_visible_metrics'])
        self.assertEqual(list(TASK['feedback']), evaluation['solver_visible_metrics'])
        self.assertEqual(list(verify.FEEDBACK), evaluation['solver_visible_metrics'])

    def test_each_instance_image_keeps_its_previous_base_and_slot(self):
        images = TASK['sidecar_images']
        for name, profile in OLD['sidecars'].items():
            if name == 'scheduler':
                continue
            dockerfile, arguments = images[profile['image']]
            self.assertEqual(dockerfile, 'docker/test-sandbox.Dockerfile')
            self.assertIn(arguments['TEST_SANDBOX_INSTANCE'], profile['args'])


class SubmissionScopeTests(unittest.TestCase):
    """The shared Agent pre-check (task.toml [submission]) and the judge's own patch policy
    (judge/patch_policy.py) describe the same editable surface."""

    def test_limits_match(self):
        submission = TASK['submission']
        self.assertEqual(submission['max_patch_bytes'], patch_policy.MAX_PATCH_BYTES)
        self.assertEqual(submission['max_changed_files'], patch_policy.MAX_CHANGED_FILES)
        self.assertEqual(tuple(submission['allowed_suffixes']),
                         patch_policy.ALLOWED_SOURCE_EXTENSIONS)
        self.assertEqual(submission['allowed_paths'],
                         [*patch_policy.STRONGLY_ALLOWED_PATTERNS,
                          *patch_policy.CONDITIONALLY_ALLOWED_PATTERNS])
        self.assertFalse(submission['allow_delete'])
        self.assertTrue(submission['allow_rename'])
        self.assertTrue(submission['allow_empty'])
        self.assertTrue(submission['allow_new_files'])
        self.assertEqual(submission['max_bytes'], OLD['submission']['max_bytes'])

    def test_path_rules_agree_on_every_starter_path_and_new_files(self):
        rules = edit_scope.Rules.from_mapping(task_manifest.edit_scope_rules(TASK))
        starter = [str(path.relative_to(TASK_ROOT / 'repo')) for path in
                   (TASK_ROOT / 'repo').rglob('*') if path.is_file()]
        extra = ['vllm/v1/core/sched/new_policy.py', 'vllm/v1/core/x/y/z.pyi',
                 'vllm/v1/engine/new.txt', 'vllm/config/scheduler.pyc', 'vllm/config/new.py',
                 'vllm/v1/request.py.bak', 'csrc/new.py', 'tests/new.py']
        for path in starter + extra:
            judge = patch_policy._validate_patch_path(path, {})[0]
            agent = (edit_scope.matches(path, rules.allowed_paths)
                     and not edit_scope.matches(path, rules.deny_paths)
                     and path.endswith(tuple(rules.allowed_suffixes)))
            self.assertEqual(agent, judge, path)


class RenameTests(unittest.TestCase):
    """A rename + edit: the judge's patch policy accepts it when both paths are allowlisted
    and rejects it otherwise; the Agent pre-check (allow_rename) decides the same."""

    def test_rename_scope_matches_the_judge(self):
        rules = edit_scope.Rules.from_mapping(task_manifest.edit_scope_rules(TASK, agent=True))
        for old, new, ok in (('vllm/v1/core/sched/a.py', 'vllm/v1/core/sched/b.py', True),
                             ('vllm/v1/core/sched/a.py', 'vllm/lora/b.py', False),
                             ('vllm/lora/a.py', 'vllm/v1/core/sched/b.py', False),
                             ('vllm/v1/core/sched/a.py', 'vllm/v1/core/sched/b.txt', False)):
            with self.subTest(old=old, new=new), tempfile.TemporaryDirectory() as scratch:
                body = ''.join(f'line {i}\n' for i in range(40))
                pristine = Path(scratch) / 'pristine'
                (pristine / old).parent.mkdir(parents=True)
                (pristine / old).write_text(body)
                patch = (f'diff --git a/{old} b/{new}\nsimilarity index 97%\nrename from {old}\n'
                         f'rename to {new}\n--- a/{old}\n+++ b/{new}\n@@ -40,0 +41 @@\n'
                         '+extra\n')
                path = Path(scratch) / 'solution.patch'
                path.write_text(patch)
                self.assertEqual(patch_policy.validate_patch(path)[0], ok)
                result = edit_scope.check(patch.encode(), pristine, rules, build_tree=False,
                                          content_rules=False)
                self.assertEqual(result.ok, ok, result.message)
                self.assertEqual(result.metrics['changed_files'], 1)  # one change, as the judge counts


class RejectionCategoryTests(unittest.TestCase):
    def patch(self, text: str) -> Path:
        directory = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__('shutil').rmtree(directory))
        path = directory / 'solution.patch'
        path.write_text(text)
        return path

    def test_every_patch_policy_rejection_has_a_shared_category(self):
        add = 'diff --git a/{0} b/{0}\n--- a/{0}\n+++ b/{0}\n@@ -1 +1 @@\n-a\n+{1}\n'
        cases = {
            'too_large': 'x' * (patch_policy.MAX_PATCH_BYTES + 1),
            'forbidden_file_type': 'diff --git a/vllm/v1/request.py b/vllm/v1/request.py\n'
                                   'GIT binary patch\n',
            'too_many_files': ''.join(add.format(f'vllm/v1/core/f{i}.py', 'x')
                                      for i in range(patch_policy.MAX_CHANGED_FILES + 1)),
            'out_of_scope': add.format('tests/a.py', 'x'),
            'content_policy': add.format('vllm/v1/request.py', 'SWE-bench'),
        }
        deletion = ('diff --git a/vllm/v1/request.py b/vllm/v1/request.py\n'
                    '--- a/vllm/v1/request.py\n+++ /dev/null\n@@ -1 +0,0 @@\n-a\n')
        suffix = add.format('vllm/v1/core/a.txt', 'x')
        secret = add.format('vllm/v1/request.py', 'os.environ["HF_TOKEN"]')
        for category, text in [*cases.items(), ('out_of_scope', deletion),
                               ('out_of_scope', suffix), ('content_policy', secret)]:
            with self.subTest(category=category):
                valid, message, _ = patch_policy.validate_patch(self.patch(text))
                self.assertFalse(valid)
                self.assertEqual(rejection_category(message), category, message)
        categories = {category for _, category in REJECTION_CATEGORIES}
        self.assertLessEqual(categories, set(edit_scope.CATEGORIES))
        with self.assertRaises(ValueError):
            rejection_category('a message nobody wrote')


class ProgressTests(unittest.TestCase):
    """verify.beat_spec_progress counts the same as the shared beat_spec_feedback.progress over
    one row per group (it is kept because it also validates the operator result)."""

    def test_counts_equal_the_shared_helper(self):
        generator = random.Random(757)
        metrics = ('paired_timing', 'p90', 'p95', 'p99', 'correctness')
        for mode, _ in itertools.product(('train', 'eval'), range(50)):
            groups = {}
            for key, ids in expected_groups(mode).items():
                comparisons = {name: generator.random() < 0.7 for name in metrics}
                groups[key] = {'complete': True, 'expected_instances': len(ids),
                               'comparisons': comparisons, 'pass': all(comparisons.values())}
            passed = all(group['pass'] for group in groups.values())
            result = {'bfcl_flip_gate_enabled': False, 'valid': True, 'pass': passed,
                      'reference_verdict': {'pass': passed, 'groups': groups},
                      'validity': {'pass': True, 'groups': {}}}
            rows = [{'workload': key, 'metrics': group['comparisons']}
                    for key, group in groups.items()]
            self.assertEqual(verify.beat_spec_progress(result, mode),
                             shared_progress({'beat_spec_workload_passes': rows,
                                              'beats_reference': int(passed)}))


if __name__ == '__main__':
    unittest.main()
