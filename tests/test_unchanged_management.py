import deployment_test_guard  # noqa: F401
import copy
import unittest
import json
import tempfile
from contextlib import ExitStack
from pathlib import Path
from unittest import mock
from preview import cli, infra
from preview.unchanged_management import (
    unchanged_native_management, require_driver_only_patches,
)
import test_retention_runtime_upgrade as retention
from test_retention_runtime_upgrade import apply_patch


class UnchangedManagementTests(unittest.TestCase):
    def fixture(self, *, modal_gpu=False):
        config, images, cm, dep = retention.RetentionUpgradeTests().fixture(modal_gpu=modal_gpu)
        source = config['images']['management']
        commit = infra.runtime_source_commit(config, 'management')
        repo = config['network']['registry_cluster_ip'] + ':5000/frontier/runtime/management'
        images['management'] = {'tag': repo + ':git-' + commit[:20],
            'reference': repo + '@' + source['digest'], 'digest': source['digest'],
            'source_sha256': source['digest']}
        ref = images['management']['reference']
        dep['spec']['template']['spec']['containers'][0]['image'] = ref
        cm['data']['MANAGEMENT_LITELLM_KEY_WORKER_IMAGE'] = ref
        dep['spec']['template']['spec']['initContainers'] = [infra.gateway_artifact_retention_init(ref)]
        return config, images, cm, dep

    def test_preserves_exact_management_with_new_driver_and_persisted_state(self):
        config, images, cm, dep = self.fixture()
        before = copy.deepcopy(images['management'])
        self.assertEqual(unchanged_native_management(config, cm, dep, images), 'frontier/runtime/management')
        self.assertEqual(images['management'], before)
        self.assertEqual(infra.validate_runtime_images(images, config)['management'], before)
        cp, dp, data = cli.runtime_upgrade_patches(cm, dep, images,
            litellm_cidr=config['network']['litellm_cluster_ip']+'/32',
            gateway_artifact_retention=True, large_verifier_results=True)
        require_driver_only_patches(cm, dep, data, dp)
        result = apply_patch(dep, dp)
        self.assertEqual(result['spec']['template']['spec']['containers'][0]['image'], before['reference'])
        self.assertEqual(result['spec']['template']['spec']['initContainers'][0]['image'], before['reference'])
        self.assertEqual(data['MANAGEMENT_LITELLM_KEY_WORKER_IMAGE'], before['reference'])

    def test_tuple_overrides_and_missing_init_fail_closed(self):
        for change in ['digest', 'source_digest', 'tag', 'commit', 'unknown_registry', 'key_worker', 'env', 'init_image', 'missing_init']:
            with self.subTest(change=change):
                config, images, cm, dep = self.fixture()
                pod = dep['spec']['template']['spec']
                if change == 'digest': images['management']['digest'] = 'sha256:'+'0'*64
                elif change == 'source_digest': images['management']['source_sha256'] = 'sha256:'+'0'*64
                elif change == 'tag': images['management']['tag'] += '-unknown'
                elif change == 'commit': config['application']['runtime_source_commits']['management']='0'*40
                elif change == 'unknown_registry': pod['containers'][0]['image']='elsewhere.example/management@'+config['images']['management']['digest']
                elif change == 'key_worker': cm['data']['MANAGEMENT_LITELLM_KEY_WORKER_IMAGE']='bad'
                elif change == 'env': pod['containers'][0].setdefault('env',[]).append({'name':'MANAGEMENT_DRIVER_IMAGE','value':'bad'})
                elif change == 'init_image': pod['initContainers'][0]['image']='bad'
                elif change == 'missing_init': pod['initContainers']=[]
                before=copy.deepcopy(images)
                with self.assertRaises(infra.DeployError): unchanged_native_management(config,cm,dep,images)
                self.assertEqual(images,before)

    def test_persisted_alternate_path_requires_source_bound_tuple(self):
        for field in ['tag','digest','source_sha256']:
            config,images,cm,dep=self.fixture()
            images['management'][field]='sha256:'+'f'*64 if field!='tag' else images['management']['tag']+'x'
            with self.assertRaises(infra.DeployError): unchanged_native_management(config,cm,dep,images)

    def test_changed_management_follows_existing_path(self):
        config,images,cm,dep=self.fixture()
        dep['spec']['template']['spec']['containers'][0]['image']='172.20.0.50:5000/frontier/runtime/management@sha256:'+'f'*64
        self.assertIsNone(unchanged_native_management(config,cm,dep,images))

    def test_extra_effective_patch_rejected_but_two_exact_defaults_allowed(self):
        config,images,cm,dep=self.fixture()
        data=copy.deepcopy(cm['data'])
        data['MANAGEMENT_DRIVER_IMAGE']='new-driver'
        for key in ['MANAGEMENT_LITELLM_KEY_WORKER_IMAGE','unrelated']:
            bad={**data,key:'unexpected'}
            with self.assertRaises(infra.DeployError): require_driver_only_patches(cm,dep,bad,[])
        image_path='/spec/template/spec/containers/0/image'
        with self.assertRaises(infra.DeployError): require_driver_only_patches(cm,dep,data,[{'op':'replace','path':image_path,'value':'different'}])
        init=dep['spec']['template']['spec']['initContainers'][0]
        init.update(terminationMessagePath='/dev/termination-log',terminationMessagePolicy='File')
        clean=copy.deepcopy(init);clean.pop('terminationMessagePath');clean.pop('terminationMessagePolicy')
        patch=[{'op':'add','path':'/spec/template/spec/initContainers','value':[clean]}]
        require_driver_only_patches(cm,dep,data,patch)
        init['terminationMessagePolicy']='FallbackToLogsOnError'
        with self.assertRaises(infra.DeployError): require_driver_only_patches(cm,dep,data,patch)



    def run_command_case(self, mutation=None, *, modal_gpu=True, changed=False):
        config, images, cm, dep = self.fixture(modal_gpu=modal_gpu)
        before_management = copy.deepcopy(images['management'])
        pod = dep['spec']['template']['spec']
        cm['data']['MANAGEMENT_DRIVER_IMAGE'] = 'old-driver@sha256:' + 'a' * 64
        if changed:
            pod['containers'][0]['image'] = 'old-management@sha256:' + 'f' * 64
            pod['initContainers'][0]['image'] = pod['containers'][0]['image']
        role = {'metadata': {'uid': 'role-uid', 'resourceVersion': '9'},
                'rules': [{'apiGroups': [''], 'resources': ['nodes'], 'verbs': ['get', 'list']}]}
        observed = {'configmap': cm, 'deployment': dep, 'clusterrole': role,
                    'networkpolicy': {'spec': {}}}
        for item in infra.modal_observer_rbac(config):
            observed[item['kind']] = copy.deepcopy(item)
        if mutation == 'role': role['rules'] = []
        if mutation == 'observer': observed['Role']['rules'][0]['verbs'].append('list')
        if mutation == 'config': cm['data']['MANAGEMENT_LITELLM_SERVICE_CIDR'] = 'different'
        if mutation == 'deployment': pod['containers'][0]['resources']['limits']['memory'] = '1Gi'
        if mutation == 'env': pod['containers'][0]['env'] = [{'name': 'MANAGEMENT_LITELLM_KEY_WORKER_IMAGE', 'value': 'bad'}]
        calls, cache_calls = [], []
        def discover(*args, receipt_path=None):
            if receipt_path:
                receipt_path.write_text('{"cidrs":["172.20.0.1/32"]}')
            return ['172.20.0.1/32']
        def cached(port, name, digest):
            cache_calls.append((name, digest))
            if name == 'frontier/runtime/management':
                return None if mutation == 'cache-missing' else 'sha256:'+'0'*64 if mutation == 'cache-wrong' else digest
            return None if sum(n == name for n, d in cache_calls) == 1 else digest
        def command(argv, **kwargs):
            calls.append(argv)
            if 'get' in argv:
                kind = argv[argv.index('get') + 1]
                result = copy.deepcopy(observed[kind])
                if mutation == 'observer-race' and kind == 'Role' and sum('get' in c and 'Role' in c for c in calls) > 1:
                    result['metadata']['resourceVersion'] = 'changed'
                return json.dumps(result)
            if 'patch' in argv and '--dry-run=server' not in argv:
                kind = argv[argv.index('patch') + 1]
                patches = json.loads(Path(argv[argv.index('--patch-file') + 1]).read_text())
                observed[kind] = apply_patch(observed[kind], patches)
            return ''
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            mocks = {
                'load_task_selection': {}, 'api_cidrs': {'side_effect': discover},
                'verify_api_routes': {}, 'verify_api_route_reader_rbac': {},
                'api_policy_patch': {'return_value': ([], {})},
                'read_gateway_snapshot': {'return_value': {}},
                'load_runtime_images': {'return_value': copy.deepcopy(images)},
                'executable': {'side_effect': lambda x: x}, 'kubernetes_environment': {'return_value': {}},
                'port_forward': {'side_effect': lambda *a: mock.MagicMock(__enter__=lambda s: 1234, __exit__=lambda *a: None)},
                'require_idle_runtime': {}, 'cached_registry_image': {'side_effect': cached},
                'run_command': {'side_effect': command},
                'platform_outputs': {'return_value': retention.Fixture.outputs(config)},
                'api_json': {'return_value': {}}, 'state_directory': {'return_value': root},
            }
            for name, options in mocks.items(): stack.enter_context(mock.patch.object(cli, name, **options))
            if mutation:
                with self.assertRaises(infra.DeployError): cli.upgrade_runtime(config, root, root/'kubeconfig')
                self.assertFalse(any(any(v in c for v in ('apply', 'patch')) and '--dry-run=server' not in c for c in calls))
                self.assertFalse((root/'runtime-images.json').exists())
                if mutation in ('cache-missing', 'cache-wrong', 'env'):
                    self.assertFalse(any('copy' in c for c in calls))
                return
            cli.upgrade_runtime(config, root, root/'kubeconfig')
            persisted = json.loads((root/'runtime-images.json').read_text())
            receipt = json.loads(next(root.glob('runtime-upgrade-*/accepted.json')).read_text())
            self.assertEqual(infra.validate_runtime_images(persisted, config), persisted)
            if not changed:
                self.assertEqual(persisted['management'], before_management)
                self.assertEqual(receipt['images']['management'], before_management)
                self.assertEqual(observed['deployment']['spec']['template']['spec']['containers'][0]['image'], before_management['reference'])
                self.assertEqual(observed['deployment']['spec']['template']['spec']['initContainers'][0]['image'], before_management['reference'])
                self.assertEqual(observed['configmap']['data']['MANAGEMENT_LITELLM_KEY_WORKER_IMAGE'], before_management['reference'])
                self.assertEqual(cache_calls[0], ('frontier/runtime/management', before_management['digest']))
                self.assertFalse(any(n == 'frontier/management' for n, d in cache_calls))
                self.assertFalse(any('apply' in c and any('modal-observer' in a for a in c) for c in calls))
            copies = [c for c in calls if 'copy' in c]
            self.assertEqual(len(copies), 2 if changed else 1)
            self.assertEqual(copies[-1][-2], config['images']['driver']['reference'])
            self.assertEqual(observed['configmap']['data']['MANAGEMENT_DRIVER_IMAGE'], persisted['driver']['reference'])

    def test_command_preserves_native_management_and_mirrors_only_driver(self):
        for modal in (False, True):
            with self.subTest(modal_gpu=modal): self.run_command_case(modal_gpu=modal)

    def test_command_cache_drift_and_extra_changes_fail_before_apply(self):
        for mutation in ('cache-missing', 'cache-wrong', 'role', 'observer', 'observer-race', 'config', 'deployment', 'env'):
            with self.subTest(mutation=mutation): self.run_command_case(mutation)

    def test_command_changed_management_still_mirrors_both_roles(self):
        self.run_command_case(changed=True)


if __name__ == '__main__': unittest.main()
