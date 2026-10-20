import deployment_test_guard  # noqa: F401  # isort: skip
import copy
import json
import unittest

from preview import cli, infra, config as config_module
from preview.runtime_reference_compat import (
    REVIEWED_COMMIT, REVIEWED_DIGESTS, REVIEWED_OLD_INIT, REVIEWED_REFERENCES,
    preserve_reviewed_public_images,
)
import test_retention_runtime_upgrade as retention
from test_retention_runtime_upgrade import apply_patch


class PublicReferenceRepairTests(unittest.TestCase):
    def fixture(self, *, modal_gpu=True):
        # This repair is bound to its historical release, independent of current pins.
        raw = json.loads((retention.ROOT / "config/site.example.json").read_text())
        raw["application"].pop("runtime_source_commits", None)
        raw["application"]["commit"] = REVIEWED_COMMIT
        for name in ("management", "driver"):
            raw["images"][name].update(tag="sha-" + REVIEWED_COMMIT,
                                       digest=REVIEWED_DIGESTS[name])
        config, images, cm, dep = retention.RetentionUpgradeTests().fixture(
            modal_gpu=modal_gpu, raw_config=raw)
        for name in ("management", "driver"):
            source = config["images"][name]
            self.assertEqual(source["digest"], REVIEWED_DIGESTS[name])
            images[name].update(digest=source["digest"], source_sha256=source["digest"],
                reference=f"{config['network']['registry_cluster_ip']}:5000/frontier/{name}@{source['digest']}")
        management = dep["spec"]["template"]["spec"]["containers"][0]
        management["image"] = REVIEWED_REFERENCES["management"]
        cm["data"]["MANAGEMENT_DRIVER_IMAGE"] = REVIEWED_REFERENCES["driver"]
        cm["data"]["MANAGEMENT_LITELLM_KEY_WORKER_IMAGE"] = REVIEWED_REFERENCES["management"]
        init = infra.gateway_artifact_retention_init(REVIEWED_OLD_INIT)
        init.update(terminationMessagePath="/dev/termination-log", terminationMessagePolicy="File")
        dep["spec"]["template"]["spec"]["initContainers"] = [init]
        return config, images, cm, dep

    def repair(self, config, images, cm, dep):
        self.assertTrue(preserve_reviewed_public_images(config, cm, dep, images))
        return cli.runtime_upgrade_patches(cm, dep, images,
            litellm_cidr=config['network']['litellm_cluster_ip']+'/32',
            gateway_artifact_retention=True, large_verifier_results=True,
            kubernetes_api_cidrs=['10.42.0.79/32','10.42.1.143/32','172.20.0.1/32'],
            reviewed_public_repair=True)

    def test_effective_change_is_only_routes_init_and_derived_annotation(self):
        config, images, cm, dep = self.fixture()
        before = copy.deepcopy((cm, dep))
        cp, dp, data = self.repair(config, images, cm, dep)
        after_cm, after_dep = apply_patch(cm, cp), apply_patch(dep, dp)
        expected_cm, expected_dep = copy.deepcopy(before)
        expected_cm['data']['MANAGEMENT_KUBERNETES_API_CIDRS'] = data['MANAGEMENT_KUBERNETES_API_CIDRS']
        expected_dep['spec']['template']['metadata']['annotations']['frontier.dev/config-sha256'] = infra.canonical_object_sha256(data)
        expected_dep['spec']['template']['spec']['initContainers'] = [infra.gateway_artifact_retention_init(REVIEWED_REFERENCES['management'])]
        self.assertEqual(after_cm, expected_cm)
        self.assertEqual(after_dep, expected_dep)
        self.assertEqual((cm, dep), before)
        self.assertEqual(images['management']['reference'], REVIEWED_REFERENCES['management'])
        self.assertEqual(images['driver']['reference'], REVIEWED_REFERENCES['driver'])

    def test_partial_tuple_unknown_repository_missing_worker_and_direct_env_rejected(self):
        for mutation in ('mixed','unknown-management','unknown-driver','missing-worker','unknown-worker','env'):
            with self.subTest(mutation=mutation):
                config, images, cm, dep = self.fixture();management=dep['spec']['template']['spec']['containers'][0]
                if mutation=='mixed':cm['data']['MANAGEMENT_DRIVER_IMAGE']=images['driver']['reference']
                if mutation=='unknown-management':management['image']='unknown.example/repo@'+REVIEWED_DIGESTS['management']
                if mutation=='unknown-driver':cm['data']['MANAGEMENT_DRIVER_IMAGE']='unknown.example/repo@'+REVIEWED_DIGESTS['driver']
                if mutation=='missing-worker':cm['data'].pop('MANAGEMENT_LITELLM_KEY_WORKER_IMAGE')
                if mutation=='unknown-worker':cm['data']['MANAGEMENT_LITELLM_KEY_WORKER_IMAGE']='unknown.example/repo@'+REVIEWED_DIGESTS['management']
                if mutation=='env':management.setdefault('env',[]).append({'name':'MANAGEMENT_DRIVER_IMAGE','value':'override'})
                old=copy.deepcopy(images)
                with self.assertRaises(infra.DeployError):self.repair(config,images,cm,dep)
                self.assertEqual(images,old)

    def test_commit_tag_and_target_digest_interlocks_remain_required(self):
        for field in ('commit','tag','repository','reference'):
            with self.subTest(field=field):
                config,images,cm,dep=self.fixture()
                if field=='commit':config['application']['commit']='0'*40
                else:config['images']['management'][field]='unreviewed'
                with self.assertRaises(infra.DeployError):self.repair(config,images,cm,dep)

    def test_initializer_unknown_image_and_every_nonimage_spec_change_rejected(self):
        for field in ('image','command','securityContext','resources','volumeMounts','terminationMessagePolicy','duplicate'):
            with self.subTest(field=field):
                config,images,cm,dep=self.fixture();init=dep['spec']['template']['spec']['initContainers'][0]
                if field=='duplicate':dep['spec']['template']['spec']['initContainers'].append(copy.deepcopy(init))
                elif field=='image':init[field]='unknown.example/management@'+REVIEWED_OLD_INIT.split('@')[1]
                elif field=='terminationMessagePolicy':init[field]='FallbackToLogsOnError'
                else:init[field]={} if isinstance(init[field],dict) else []
                with self.assertRaisesRegex(infra.DeployError,'existing init differs'):self.repair(config,images,cm,dep)

    def test_extra_config_resources_mounts_and_missing_init_rejected(self):
        for mutation in ('worker-cidr','resources','tmp','mount','missing-init'):
            with self.subTest(mutation=mutation):
                config,images,cm,dep=self.fixture();pod=dep['spec']['template']['spec']
                if mutation=='worker-cidr':cm['data']['MANAGEMENT_LITELLM_SERVICE_CIDR']='172.20.0.61/32'
                if mutation=='resources':pod['containers'][0]['resources']['limits']['memory']='1Gi'
                if mutation=='tmp':next(v for v in pod['volumes']if v['name']=='tmp')['emptyDir']['sizeLimit']='1Gi'
                if mutation=='mount':pod['containers'][0]['volumeMounts']=[m for m in pod['containers'][0]['volumeMounts']if m.get('mountPath')!='/var/lib/frontier-local/artifacts']
                if mutation=='missing-init':pod.pop('initContainers')
                with self.assertRaises(infra.DeployError):self.repair(config,images,cm,dep)

    def test_already_public_initializer_is_idempotent(self):
        config,images,cm,dep=self.fixture();dep['spec']['template']['spec']['initContainers'][0]['image']=REVIEWED_REFERENCES['management']
        self.repair(config,images,cm,dep)

    def test_different_digest_upgrade_uses_existing_path(self):
        config,images,cm,dep=self.fixture()
        for roles in (('management',), ('driver',), ('management', 'driver')):
            with self.subTest(roles=roles):
                changed = copy.deepcopy(config)
                for name in roles:
                    changed['images'][name]['digest'] = 'sha256:' + '1' * 64
                self.assertFalse(preserve_reviewed_public_images(changed, cm, dep, images))

    def test_full_guarded_upgrade_preserves_public_references_in_readback_and_receipt(self):
        runner=retention.RetentionUpgradeTests()
        runner.fixture=self.fixture
        runner.check_supported_upgrade(True, retain_existing_init=True)

    def test_real_config_validation_rejects_malformed_digest_and_commit_tag_mismatch(self):
        raw = json.loads((retention.ROOT / "config/site.example.json").read_text())
        for field, value in (("digest", "unreviewed"), ("tag", "sha-" + "0" * 40)):
            with self.subTest(field=field):
                changed = copy.deepcopy(raw)
                changed["images"]["management"][field] = value
                with self.assertRaises(infra.DeployError):
                    config_module.validate_config(changed)

    def test_runtime_record_digest_mismatch_rejected(self):
        config, images, cm, dep = self.fixture()
        images["management"]["digest"] = "sha256:" + "1" * 64
        with self.assertRaises(infra.DeployError):
            self.repair(config, images, cm, dep)

    def test_public_runtime_records_roundtrip_and_reject_partial_or_unknown_tuple(self):
        config, images, cm, dep = self.fixture()
        self.repair(config, images, cm, dep)
        self.assertEqual(infra.validate_runtime_images(images, config), images)
        for field in ("tag", "digest", "reference", "source_sha256"):
            changed = copy.deepcopy(images)
            changed["management"][field] = "unknown"
            with self.subTest(field=field), self.assertRaises(infra.DeployError):
                infra.validate_runtime_images(changed, config)
        changed = copy.deepcopy(images)
        changed["driver"] = retention.Fixture.images(config)["driver"]
        with self.assertRaises(infra.DeployError):
            infra.validate_runtime_images(changed, config)

    def test_invalid_live_reference_or_spec_rejected_before_registry_or_site_write(self):
        for mutation in ("reference", "resources"):
            with self.subTest(mutation=mutation):
                runner = retention.RetentionUpgradeTests()
                def invalid_fixture(*, modal_gpu):
                    config, images, cm, dep = self.fixture(modal_gpu=modal_gpu)
                    management = dep["spec"]["template"]["spec"]["containers"][0]
                    if mutation == "reference":
                        management["image"] = "unknown.example/management@" + REVIEWED_DIGESTS["management"]
                    else:
                        management["resources"]["limits"]["memory"] = "1Gi"
                    return config, images, cm, dep
                runner.fixture = invalid_fixture
                runner.check_supported_upgrade(True, retain_existing_init=True, expected_failure=True)
