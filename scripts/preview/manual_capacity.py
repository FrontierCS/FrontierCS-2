"""Typed opt-in bindings for Preview's two manually controlled worker pools."""
from datetime import datetime
import math
import re


def capacity_settings(config):
    raw = config['management'].get('manual_capacity')
    if raw is None:
        return None
    if not isinstance(raw, dict) or set(raw) != {'m6i', 'c6i', 'price_observed_at'}:
        raise ValueError('management.manual_capacity requires m6i, c6i and price_observed_at')
    site = config['site']
    if site['name'] != 'preview':
        raise ValueError('manual capacity is scoped to the Preview site')
    prefix = f"arn:aws:eks:{site['aws_region']}:{site['expected_account_id']}"
    cluster = f"frontiercs2-{site['name']}"
    pattern = re.compile(re.escape(f'{prefix}:nodegroup/{cluster}/') + r'[A-Za-z0-9_-]+/[A-Za-z0-9_-]+')
    result = {'cluster_arn': f'{prefix}:cluster/{cluster}', 'region': site['aws_region'], **raw}
    datetime.fromisoformat(raw['price_observed_at'].replace('Z', '+00:00'))
    for role in ('m6i', 'c6i'):
        binding = raw[role]
        if not isinstance(binding, dict) or set(binding) != {'node_group_arn', 'labels', 'tags', 'hourly_usd'}:
            raise ValueError(f'manual capacity {role} requires exact ARN, labels, tags and hourly_usd')
        if not isinstance(binding['node_group_arn'], str) or not pattern.fullmatch(binding['node_group_arn']):
            raise ValueError(f'manual capacity {role} ARN must be an exact nodegroup in this cluster')
        if binding['labels'] != {'frontier.dev/pool': 'cpu', 'frontier.dev/site': 'preview'}:
            raise ValueError('manual capacity requires stable Preview CPU labels')
        tags = binding['tags']
        if not isinstance(tags, dict) or not tags or any(not isinstance(k, str) or not isinstance(v, str) or not k or not v for k, v in tags.items()):
            raise ValueError('manual capacity requires explicit ownership tags')
        price = binding['hourly_usd']
        if type(price) not in (int, float) or not math.isfinite(price) or price <= 0:
            raise ValueError('manual capacity hourly_usd must be positive and finite')
    if raw['m6i']['node_group_arn'] == raw['c6i']['node_group_arn']:
        raise ValueError('manual worker pools must be distinct')
    cpu = config['compute']['cpu']
    if cpu['instance_types'] != ['c6i.4xlarge'] or cpu['min_size'] != 1 or cpu['max_size'] != 2:
        raise ValueError('opt-in manual capacity requires compute.cpu c6i.4xlarge min=1 max=2; desired remains explicit')
    return result


def observer_rbac():
    name = 'frontier-management-infrastructure-observer'
    return [
        {'apiVersion': 'rbac.authorization.k8s.io/v1', 'kind': 'ClusterRole',
         'metadata': {'name': name}, 'rules': [
             {'apiGroups': [''], 'resources': ['nodes'], 'verbs': ['get', 'list']},
             {'apiGroups': ['metrics.k8s.io'], 'resources': ['nodes'], 'verbs': ['get', 'list']}]},
        {'apiVersion': 'rbac.authorization.k8s.io/v1', 'kind': 'ClusterRoleBinding',
         'metadata': {'name': name}, 'roleRef': {'apiGroup': 'rbac.authorization.k8s.io', 'kind': 'ClusterRole', 'name': name},
         'subjects': [{'kind': 'ServiceAccount', 'name': 'frontier-management', 'namespace': 'frontier-system'}]},
    ]
