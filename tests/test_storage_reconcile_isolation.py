import deployment_test_guard  # noqa: F401
import unittest
from unittest import mock
from tests.test_frontier_deploy import load_storage_controller


class StorageReconcileIsolationTests(unittest.TestCase):
    def test_terminating_namespace_causes_no_api_calls(self):
        controller = load_storage_controller()
        with mock.patch.object(controller, 'api') as api:
            controller.reconcile_namespace({'metadata': {'name': 'frontier-run-old',
                'deletionTimestamp': '2026-10-08T00:00:00Z'}})
        api.assert_not_called()

    def test_one_namespace_failure_does_not_starve_later_namespaces_or_collect_live_pvs(self):
        controller = load_storage_controller()
        namespaces = [{'metadata': {'name': name}} for name in ('terminating', 'broken', 'healthy')]
        namespaces[0]['metadata']['deletionTimestamp'] = '2026-10-08T00:00:00Z'
        with (
            mock.patch.object(controller, 'api', return_value={'items': namespaces}),
            mock.patch.object(controller, 'reconcile_namespace', side_effect=[None, RuntimeError('403'), None]) as reconcile,
            mock.patch.object(controller, 'ownership', return_value=('run', 'attempt')),
            mock.patch.object(controller, 'cleanup_orphan_pvs') as cleanup,
            mock.patch('builtins.print'),
        ):
            controller.reconcile_once()
        self.assertEqual(reconcile.call_args_list, [mock.call(n) for n in namespaces])
        cleanup.assert_called_once_with({'terminating', 'broken', 'healthy'})

    def test_failed_namespace_list_never_runs_orphan_cleanup(self):
        controller = load_storage_controller()
        with mock.patch.object(controller, 'api', side_effect=RuntimeError('network')), mock.patch.object(controller, 'cleanup_orphan_pvs') as cleanup:
            with self.assertRaises(RuntimeError): controller.reconcile_once()
        cleanup.assert_not_called()
