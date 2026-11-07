import math

exp = math.exp

config = {
    'kvmerger': {
        'algorithm': 'kvmerger',
        'top_k_ratio': 0.0,
        'c2_method': 'merge',
        'beta_method': 'zero',
    },
    'kvmerger_on-policy': {
        'algorithm': 'kvmerger',
        'top_k_ratio': 0.0,
        'c2_method': 'merge',
        'beta_method': 'zero',
        'on_policy': True,
    },
    'kvmerger+AM': {
        'algorithm': 'kvmerger',
        'top_k_ratio': 0.0,
        'c2_method': 'lsq',
        'beta_method': 'nnls',
        'nnls_iters': 2,
        'nnls_lower_bound': exp(-3),
        'nnls_upper_bound': exp(3),
    },
    'kvmerger+AM_on-policy': {
        'algorithm': 'kvmerger',
        'top_k_ratio': 0.0,
        'c2_method': 'lsq',
        'beta_method': 'nnls',
        'nnls_iters': 2,
        'nnls_lower_bound': exp(-3),
        'nnls_upper_bound': exp(3),
        'on_policy': True,
    },
}
