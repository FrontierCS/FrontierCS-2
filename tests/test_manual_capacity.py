from __future__ import annotations

import deployment_test_guard  # noqa: F401  # isort: skip

import copy
import json
import unittest

from test_frontier_deploy import Fixture, ROOT, frontier_deploy
from preview.manual_capacity import capacity_settings


class ManualCapacityTests(unittest.TestCase):
    def config(self):
        config = json.loads((ROOT / 'config/site.example.json').read_text())
        config['compute']['cpu'].update(instance_types=['c6i.4xlarge'], min_size=1, desired_size=2, max_size=2)
        pool = lambda name: {
            'node_group_arn': f'arn:aws:eks:us-west-1:123456789012:nodegroup/frontiercs2-preview/{name}/id',
            'labels': {'frontier.dev/pool': 'cpu', 'frontier.dev/site': 'preview'},
            'tags': {'frontier:owner': 'operator'}, 'hourly_usd': 1.0}
        config['management']['manual_capacity'] = {
            'm6i': pool('preview-parallel'), 'c6i': pool('frontiercs2-preview-cpu'),
            'price_observed_at': '2026-10-05T00:00:00Z'}
        return frontier_deploy.validate_config(config)

    def test_opt_in_adds_only_management_credentials_and_read_only_observation(self):
        config = self.config()
        data = json.loads(frontier_deploy.render_runtime_manifest(config, Fixture.outputs(config), Fixture.images(config), ['172.20.0.1/32']))
        cm = next(r for r in data['items'] if r['kind'] == 'ConfigMap' and r['metadata']['name'] == 'frontier-management-config')['data']
        self.assertEqual(json.loads(cm['MANAGEMENT_PREVIEW_CAPACITY']), capacity_settings(config))
        role = next(r for r in data['items'] if r['kind'] == 'ClusterRole' and r['metadata']['name'] == 'frontier-management-infrastructure-observer')
        self.assertEqual(role['rules'], [
            {'apiGroups':[''], 'resources':['nodes'], 'verbs':['get','list']},
            {'apiGroups':['metrics.k8s.io'], 'resources':['nodes'], 'verbs':['get','list']}])
        binding = next(r for r in data['items'] if r['kind'] == 'ClusterRoleBinding' and r['metadata']['name'] == role['metadata']['name'])
        self.assertEqual(binding['subjects'], [{'kind':'ServiceAccount','name':'frontier-management','namespace':'frontier-system'}])

    def test_default_configuration_has_no_capacity_authority(self):
        config = frontier_deploy.validate_config(json.loads((ROOT/'config/site.example.json').read_text()))
        self.assertIsNone(capacity_settings(config))
        rendered = frontier_deploy.render_runtime_manifest(config, Fixture.outputs(config), Fixture.images(config), ['172.20.0.1/32'])
        self.assertNotIn(b'MANAGEMENT_PREVIEW_CAPACITY', rendered)

    def test_wrong_account_wildcards_duplicate_groups_and_terraform_min_drift_fail(self):
        config = self.config()
        for mutate in [
            lambda c: c['management']['manual_capacity']['m6i'].update(node_group_arn='arn:aws:eks:us-west-1:999999999999:nodegroup/frontiercs2-preview/a/id'),
            lambda c: c['management']['manual_capacity']['m6i'].update(node_group_arn='arn:aws:eks:us-west-1:123456789012:nodegroup/frontiercs2-preview/*/id'),
            lambda c: c['management']['manual_capacity'].update(m6i=copy.deepcopy(c['management']['manual_capacity']['c6i'])),
            lambda c: c['compute']['cpu'].update(min_size=2),
        ]:
            changed=copy.deepcopy(config); mutate(changed)
            with self.assertRaises(frontier_deploy.DeployError): frontier_deploy.validate_config(changed)

    def test_system_and_wrong_cpu_group_are_not_rendered(self):
        config=self.config(); outputs=Fixture.outputs(config)
        outputs['cpu_node_group_name']='different'
        with self.assertRaises(frontier_deploy.DeployError):
            frontier_deploy.render_runtime_manifest(config, outputs, Fixture.images(config), ['172.20.0.1/32'])

    def test_iam_mutation_has_only_two_exact_group_arns(self):
        source=(ROOT/'terraform/platform/main.tf').read_text()
        block=source.split('sid       = "ApplyManualWorkerCapacity"',1)[1].split('}',1)[0]
        self.assertIn('["eks:UpdateNodegroupConfig"]',block)
        self.assertIn('resources = var.manual_capacity_nodegroup_arns',block)
        self.assertNotIn('autoscaling:',source)


if __name__ == '__main__': unittest.main()
