import deployment_test_guard  # noqa: F401
import copy
import json
import unittest

from preview import cli, infra
from test_frontier_deploy import Fixture
import test_retention_runtime_upgrade as retention
from test_retention_runtime_upgrade import apply_patch


class InfrastructureUpgradeTests(unittest.TestCase):
    def test_default_install_observes_nodes_without_enabling_capacity(self):
        config, images, _, _ = retention.RetentionUpgradeTests().fixture()
        for simulator in (False, True):
            items = json.loads(infra.render_runtime_manifest(
                config, Fixture.outputs(config), images, ['172.20.0.1/32'],
                simulator_bootstrap=simulator))['items']
            cm = next(x for x in items if x['kind'] == 'ConfigMap'
                      and x['metadata']['name'] == 'frontier-management-config')['data']
            role = next(x for x in items if x['kind'] == 'ClusterRole'
                        and x['metadata']['name'] == 'frontier-management-preview')
            self.assertEqual('MANAGEMENT_EKS_STATUS_CLUSTER' in cm, not simulator)
            if not simulator:
                self.assertEqual(cm['MANAGEMENT_EKS_STATUS_CLUSTER'], Fixture.outputs(config)['cluster_name'])
                self.assertEqual(cm['MANAGEMENT_EKS_STATUS_REGION'], config['site']['aws_region'])
            self.assertNotIn('MANAGEMENT_PREVIEW_CAPACITY', cm)
            self.assertEqual(any(r['resources'] == ['nodes'] for r in role['rules']), not simulator)
            self.assertFalse(any('metrics.k8s.io' in r['apiGroups'] for r in role['rules']))

    def test_upgrade_removes_legacy_fields_and_preserves_unrelated_configuration(self):
        _, images, cm, dep = retention.RetentionUpgradeTests().fixture()
        legacy = ['MANAGEMENT_EKS_STATUS_PATH', 'MANAGEMENT_EKS_STATUS_NODEGROUPS',
                  'MANAGEMENT_EKS_STATUS_MAX_AGE_SECONDS']
        cm['data'].update(dict.fromkeys(legacy, 'old'))
        cm['data']['MANAGEMENT_PREVIEW_CAPACITY'] = 'existing-explicit-binding'
        before = copy.deepcopy(cm)
        cp, dp, data = cli.runtime_upgrade_patches(
            cm, dep, images, cluster_name='current-cluster', region='us-west-1')
        self.assertEqual(cm, before)
        self.assertEqual(apply_patch(cm, cp)['data'], data)
        self.assertFalse(any(k in data for k in legacy))
        self.assertEqual(data['MANAGEMENT_EKS_STATUS_CLUSTER'], 'current-cluster')
        self.assertEqual(data['MANAGEMENT_EKS_STATUS_REGION'], 'us-west-1')
        for key, value in before['data'].items():
            if key not in {*legacy, 'MANAGEMENT_DRIVER_IMAGE', 'MANAGEMENT_EKS_STATUS_CLUSTER', 'MANAGEMENT_EKS_STATUS_REGION'}:
                self.assertEqual(data[key], value)
        self.assertEqual(dp[-1]['value']['frontier.dev/config-sha256'], cli.canonical_object_sha256(data))

    def test_direct_overrides_partial_identity_and_simulator_fail_before_patch(self):
        _, images, cm, dep = retention.RetentionUpgradeTests().fixture()
        for key in ['MANAGEMENT_EKS_STATUS_CLUSTER', 'MANAGEMENT_EKS_STATUS_REGION',
                    'MANAGEMENT_EKS_STATUS_PATH', 'MANAGEMENT_EKS_STATUS_NODEGROUPS',
                    'MANAGEMENT_EKS_STATUS_MAX_AGE_SECONDS']:
            changed = copy.deepcopy(dep)
            changed['spec']['template']['spec']['containers'][0]['env'] = [{'name': key, 'value': 'old'}]
            with self.assertRaisesRegex(cli.DeployError, 'environment overrides'):
                cli.runtime_upgrade_patches(cm, changed, images, cluster_name='cluster', region='us-west-1')
        for cluster, region in [('cluster', None), (None, 'us-west-1'), ('bad/cluster', 'us-west-1')]:
            with self.assertRaises(cli.DeployError):
                cli.runtime_upgrade_patches(cm, dep, images, cluster_name=cluster, region=region)
        cm['data']['MANAGEMENT_DRIVER_MODE'] = 'simulator'
        with self.assertRaisesRegex(cli.DeployError, 'kubernetes mode'):
            cli.runtime_upgrade_patches(cm, dep, images, cluster_name='cluster', region='us-west-1')

    def test_role_patch_preserves_rules_and_rejects_replaced_or_changed_role(self):
        role = {'metadata': {'uid': 'u', 'resourceVersion': 'r'}, 'rules': [
            {'apiGroups': [''], 'resources': ['pods'], 'verbs': ['list']}]}
        original = copy.deepcopy(role)
        patches, rules = cli.infrastructure_role_patch(role)
        result = apply_patch(role, patches)
        self.assertEqual(role, original)
        self.assertEqual(rules[:-1], original['rules'])
        self.assertEqual(rules[-1], {'apiGroups': [''], 'resources': ['nodes'], 'verbs': ['get', 'list']})
        self.assertEqual(cli.infrastructure_role_patch(result)[1], rules)
        for key in ['uid', 'resourceVersion']:
            changed = copy.deepcopy(role)
            changed['metadata'][key] = 'other'
            with self.assertRaises(AssertionError):
                apply_patch(changed, patches)


if __name__ == '__main__':
    unittest.main()
