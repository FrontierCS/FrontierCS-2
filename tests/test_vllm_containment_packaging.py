"""Source/import closure and cache boundaries of the actual shipped roles."""
import deployment_test_guard  # noqa: F401
import ast
from pathlib import Path
import unittest

TASK = Path(__file__).resolve().parents[1] / 'tasks/frontier-cs-2.0-demo/problems/MLSys/vllm_llm_serving_optimization'
JUDGE = TASK / 'judge'


def sources(dockerfile):
    return {word for line in dockerfile.splitlines() if line.startswith('COPY ')
            for word in line.split()[1:-1] if word.startswith('judge/')}


def local_module(name):
    direct = JUDGE / (name.replace('.', '/') + '.py')
    package = JUDGE / name.replace('.', '/') / '__init__.py'
    return direct if direct.is_file() else package if package.is_file() else None


def imports(name):
    path = local_module(name)
    package = name.split('.') if path.name == '__init__.py' else name.split('.')[:-1]
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            candidates = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            base = '.'.join(package[:len(package) - node.level + 1]) if node.level else ''
            module = '.'.join(part for part in (base, node.module) if part)
            candidates = [module] + [module + '.' + alias.name for alias in node.names]
        else:
            continue
        yield from (candidate for candidate in candidates if local_module(candidate))


class ContainmentPackagingTests(unittest.TestCase):
    def assert_closure(self, role, entrypoints):
        dockerfile = (TASK / role / 'Dockerfile').read_text()
        projected = sources(dockerfile)
        pending, observed = list(entrypoints), set()
        while pending:
            module = pending.pop()
            if module in observed:
                continue
            observed.add(module)
            source = 'judge/' + str(local_module(module).relative_to(JUDGE))
            self.assertTrue(source in projected or any(
                source.startswith(prefix) for prefix in projected if prefix.endswith('/')),
                (role, module, source))
            pending.extend(imports(module))
        return projected

    def test_scheduler_complete_broker_dependency_closure_excludes_cpu_authority(self):
        projected = self.assert_closure('sidecar-scheduler', [
            'scheduler_service', 'episode_worker', 'http_request_worker'])
        for private in ['cpu_controller', 'controller_envelope', 'controller_identity',
                        'controller_contract_build', 'remote_terminal']:
            self.assertNotIn('judge/' + private + '.py', projected)

    def test_gpu_relay_is_stdlib_only_plus_bounded_stream_format(self):
        projected = self.assert_closure('verifier-gpu', ['scheduler_relay', 'scheduler_stream'])
        self.assertEqual(projected, {'judge/variant_batch.py', 'judge/scheduler_relay.py',
                                     'judge/scheduler_stream.py'})

    def test_cpu_contract_follows_final_source_copy_but_git_prep_precedes_mutable_source(self):
        text = (TASK / 'verifier/Dockerfile').read_text()
        self.assertLess(text.index('RUN git -C /opt/frontier/vllm-clean'), text.index('COPY judge/'))
        self.assertGreater(text.index('RUN PYTHONPATH='), text.index('COPY verifier/'))
        self.assertIn('--source-tree judge --source-tree verifier', text)
        self.assertNotIn('controller-contract', text.split('COPY judge/')[0])
        self.assertIn('chmod -R a-w /opt/frontier/vllm-clean /opt/frontier/python',
                      text.split('COPY judge/')[0])
        self.assertNotIn('chmod -R a-w /opt/frontier\n', text)

    def test_all_48_swe_runtime_layers_are_checked_before_final_and_remain_readonly(self):
        roles = sorted(TASK.glob('sidecar-swe-*/Dockerfile'))
        self.assertEqual(len(roles), 48)
        for path in roles:
            with self.subTest(role=path.parent.name):
                text = path.read_text()
                self.assertLess(text.index('COPY --chmod=0444 judge/serving_eval/swe_case_failure.py'), text.index('--phase verify'))
                self.assertLess(text.index('chown -R 10001:10001 /testbed'), text.index('COPY --chmod=0444 judge/serving_eval/swe_tool_service.py'))
                self.assertIn('FROM prepared AS runtime\n', text)
                self.assertIn('FROM runtime AS offline-check\n', text)
                self.assertIn('FROM runtime\n', text)
                runtime = text.split('FROM prepared AS runtime\n')[1].split('FROM runtime AS offline-check\n')[0]
                self.assertIn('swe_test_worker.py', runtime)
                final = text.rsplit('FROM runtime\n', 1)[1]
                self.assertIn('COPY --from=offline-check', final)
                self.assertIn('import serving_eval.swe_tool_service', runtime)
                self.assertNotIn('pip install', runtime)
                self.assertNotIn('chown -R', runtime)
