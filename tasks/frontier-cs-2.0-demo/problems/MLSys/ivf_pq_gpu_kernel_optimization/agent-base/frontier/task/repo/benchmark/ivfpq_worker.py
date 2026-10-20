"""Paired IVF-PQ measurement shared by trusted evaluation and public GPU development."""
from __future__ import annotations

import hashlib
import importlib
import json
import math
from pathlib import Path
import statistics
import sys
import time


def verified_data_path(data: dict) -> Path:
    path = Path(data['path'])
    expected = data.get('sha256')
    if not isinstance(expected, str) or len(expected) != 71 or not expected.startswith('sha256:'):
        raise RuntimeError('SIFT data identity is not configured; a verified artifact is required')
    if not path.is_file():
        raise RuntimeError('The verified SIFT artifact is missing from the GPU image')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    if 'sha256:' + digest.hexdigest() != expected:
        raise RuntimeError('SIFT artifact checksum does not match the task definition')
    return path


def validate_output(workload: dict, output, database_size: int) -> None:
    import torch
    if not isinstance(output, (tuple, list)) or len(output) != 2:
        raise ValueError('search must return (vals, ids)')
    vals, ids = output
    shape = (int(workload['Q']), int(workload['k']))
    if not torch.is_tensor(vals) or not torch.is_tensor(ids) or tuple(vals.shape) != shape or tuple(ids.shape) != shape:
        raise ValueError('search output shape is invalid')
    if vals.dtype != torch.float32 or ids.dtype != torch.int64:
        raise ValueError('search must return float32 distances and int64 IDs')
    if vals.device != ids.device:
        raise ValueError('search output tensors must use the same device')
    valid = ids >= 0
    if bool(((ids < -1) | (ids >= database_size)).any()):
        raise ValueError('search IDs are out of range')
    if not bool(torch.isfinite(vals[valid]).all()):
        raise ValueError('valid-neighbor distances must be finite')
    if not bool(torch.isposinf(vals[~valid]).all()):
        raise ValueError('padding must use ID -1 with distance +inf')
    ordered = ids.sort(dim=1).values
    if bool(((ordered[:, 1:] == ordered[:, :-1]) & (ordered[:, 1:] >= 0)).any()):
        raise ValueError('valid neighbor IDs must not repeat within a query')


def exact_nearest_distances(base, queries, block=256):
    """Squared L2 to each query's exact nearest database vector (float32).

    Brute force over the uncompressed SIFT base on the actual (bf16-rounded)
    queries; the argmin is recomputed with the direct difference formula so it
    matches nn_recall's distance arithmetic.
    """
    import torch
    best = torch.empty(queries.shape[0], dtype=torch.float32, device=queries.device)
    for start in range(0, queries.shape[0], block):
        q = queries[start:start + block].to(torch.float32)
        nearest = torch.cdist(q, base).argmin(1)
        best[start:start + block] = ((q - base.index_select(0, nearest)) ** 2).sum(1)
    return best


def nn_recall(base, queries, ids, nearest, block=256) -> float:
    """Recall@k of the IVFADC evaluation: share of queries whose exact nearest
    neighbor is among the k returned vectors. A returned vector at the nearest
    distance (an exact duplicate) also counts; padding never counts."""
    import torch
    hits = 0
    for start in range(0, queries.shape[0], block):
        chunk = ids[start:start + block]
        valid = chunk >= 0
        vectors = base.index_select(0, chunk.clamp_min(0).reshape(-1)).reshape(chunk.shape[0], chunk.shape[1], -1)
        q = queries[start:start + block].to(torch.float32)
        distances = ((q[:, None, :] - vectors) ** 2).sum(-1)
        distances = torch.where(valid, distances, torch.full_like(distances, float('inf')))
        bound = nearest[start:start + block] * (1 + 1e-6) + 1e-6
        hits += int((distances.min(1).values <= bound).sum().item())
    return hits / queries.shape[0]


def iso_recall(agent_ids, naive_ids) -> float:
    # Output validation rejects duplicate valid IDs. Padding never counts as a hit.
    hit = ((agent_ids.unsqueeze(2) == naive_ids.unsqueeze(1))
           & (agent_ids.unsqueeze(2) >= 0) & (naive_ids.unsqueeze(1) >= 0)).any(2)
    return float(hit.sum().item()) / naive_ids.numel()


def validate_distance_config(config: dict, *, allow_calibration: bool = False) -> None:
    if config.get('mode') == 'calibration' and allow_calibration:
        return
    if config.get('mode') != 'enforce':
        raise RuntimeError('Distance consistency enforcement must be configured')
    for name in ('atol', 'rtol'):
        value = config.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise RuntimeError('Distance tolerance is pending user confirmation; no final score is available')


def adc_distances(index, queries, ids):
    """Frozen float32 residual ADC, with the naive dsub-then-m reduction order."""
    import torch
    positions = torch.empty(index.M, dtype=torch.int64, device=ids.device)
    positions[index.ids] = torch.arange(index.M, dtype=torch.int64, device=ids.device)
    result = torch.full(ids.shape, float('inf'), dtype=torch.float32, device=ids.device)
    rows, columns = (ids >= 0).nonzero(as_tuple=True)
    for start in range(0, rows.numel(), 8192):
        r, c = rows[start:start + 8192], columns[start:start + 8192]
        pos = positions[ids[r, c]]
        cells = torch.searchsorted(index.list_offsets, pos, right=True) - 1
        query = queries[r].to(torch.float32)
        if query.shape[1] < index.Dp:
            query = torch.nn.functional.pad(query, (0, index.Dp - query.shape[1]))
        if index.by_residual:
            query = query - index.centroids[cells].to(torch.float32)
        codes = index.codes[pos].long()
        subspaces = torch.arange(index.m, device=ids.device)[None, :]
        reconstructed = index.pq_codebooks[subspaces, codes].to(torch.float32)
        squared = (query.reshape(-1, index.m, index.dsub) - reconstructed) ** 2
        result[r, c] = squared.sum(-1).sum(-1)
    return result


def distance_errors(vals, ids, expected, config: dict) -> dict:
    import torch
    validate_distance_config(config, allow_calibration=True)
    valid = ids >= 0
    truth = expected[valid]
    if not bool(torch.isfinite(truth).all()):
        raise RuntimeError('Frozen ADC oracle produced a non-finite valid-neighbor distance')
    # Oracle arithmetic stays float32; promotion only preserves error-report precision.
    truth = truth.detach().to(device='cpu', dtype=torch.float64)
    actual = vals[valid].detach().to(device='cpu', dtype=torch.float64)
    absolute = (actual - truth).abs()
    nonzero = truth != 0
    relative = absolute[nonzero] / truth[nonzero].abs()

    def distribution(values):
        if values.numel() == 0:
            return {'count': 0, 'min': None, 'p50': None, 'p90': None, 'p99': None, 'max': None}
        numbers = torch.quantile(values, torch.tensor([0.0, 0.5, 0.9, 0.99, 1.0], dtype=torch.float64)).tolist()
        return dict(zip(('min', 'p50', 'p90', 'p99', 'max'), numbers), count=values.numel())

    gate = None
    violations = None
    if config.get('mode') == 'enforce':
        rejected = absolute > float(config['atol']) + float(config['rtol']) * truth.abs()
        violations = int(rejected.sum().item())
        gate = violations == 0
    return {
        'absolute_error': distribution(absolute),
        'relative_error_nonzero_oracle': distribution(relative),
        'zero_oracle_count': int((~nonzero).sum().item()),
        'zero_oracle_nonzero_error_count': int(((~nonzero) & (absolute != 0)).sum().item()),
        'padding_count': int((~valid).sum().item()),
        'distance_gate_passed': gate,
        'distance_gate_violation_count': violations,
        'mode': config['mode'],
    }


def fresh_index(index):
    import torch
    result = object.__new__(type(index))
    for name, value in vars(index).items():
        if torch.is_tensor(value):
            value = value.clone()
            if name in {'centroids', 'pq_codebooks'} and value.is_floating_point():
                value = value.to(torch.bfloat16).to(value.dtype)
        result.__dict__[name] = value
    return result


def summarize_samples(workload_id: str, samples: list[dict]) -> dict:
    if not samples:
        raise ValueError('no timed samples')
    for row in samples:
        if any(not math.isfinite(row[key]) or row[key] <= 0 for key in ('naive_ms', 'candidate_ms')):
            raise ValueError('timing samples must be positive and finite')
        if not math.isfinite(row['recall']) or not 0 <= row['recall'] <= 1:
            raise ValueError('recall is invalid')
        if not math.isfinite(row['nn_recall']) or not 0 <= row['nn_recall'] <= 1:
            raise ValueError('nearest-neighbor recall is invalid')
    return {
        'id': workload_id,
        'search_latency_ms': statistics.median(row['candidate_ms'] for row in samples),
        'naive_latency_ms': statistics.median(row['naive_ms'] for row in samples),
        'iso_recall': min(row['recall'] for row in samples),
        'nn_recall': min(row['nn_recall'] for row in samples),
        'speedup': statistics.median(row['naive_ms'] / row['candidate_ms'] for row in samples),
        'timed_samples': samples,
        'valid': True,
    }


def measure(source_root: Path, oracle_root: Path, workload: dict, config: dict) -> dict:
    import h5py
    import numpy as np
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable')
    synchronize, perf_counter = torch.cuda.synchronize, time.perf_counter
    sys.path.insert(0, str(oracle_root))
    naive = importlib.import_module('refivfpq')
    frozen_build, frozen_search = naive.ivf_pq_build, naive.ivf_pq_search
    validate_distance_config(config['distance_check'], allow_calibration=True)
    data_path = verified_data_path(config['data'])
    with h5py.File(data_path, 'r') as handle:
        base = torch.from_numpy(np.asarray(handle['train'], dtype=np.float32)).cuda().contiguous()
        queries = torch.from_numpy(np.asarray(handle['test'], dtype=np.float32)).cuda().contiguous()
    if base.ndim != 2 or queries.ndim != 2 or base.shape[1] != queries.shape[1] or int(workload['Q']) > queries.shape[0]:
        raise RuntimeError('SIFT train/test arrays do not satisfy the original workload')
    seed = int(workload['base_seed']) + 1000 * (int(workload['ordinal']) + 1)
    nlist = int(workload['nlist'])
    index = frozen_build(base, nlist, m=int(workload['m']), nprobe=int(workload['nprobe']), seed=seed,
                         train_size=min(base.shape[0], max(200000, nlist * 40)),
                         pq_train_size=min(base.shape[0], 100000))
    sys.path.insert(0, str(source_root))
    candidate = importlib.import_module('ivfpqlib')

    def data_for(iteration_seed):
        generator = torch.Generator(device='cuda').manual_seed(iteration_seed)
        permutation = torch.randperm(queries.shape[0], generator=generator, device='cuda')[:int(workload['Q'])]
        q = queries.index_select(0, permutation)
        return fresh_index(index), q.to(torch.bfloat16).to(q.dtype).contiguous()

    def call(function, data):
        return function(data[0], data[1], int(workload['k']), nprobe=int(workload['nprobe']))

    def timed(function, data):
        synchronize()
        start = perf_counter()
        output = call(function, data)
        synchronize()
        return (perf_counter() - start) * 1000.0, output

    for iteration in range(int(config['warmup_iterations'])):
        data = data_for(seed + 100 + iteration)
        call(frozen_search, data)
        call(candidate.ivf_pq_search, data)
        synchronize()
        del data
    torch.cuda.empty_cache()
    samples = []
    for iteration in range(int(config['timed_iterations'])):
        data = data_for(seed + 10000 + iteration)
        naive_ms, naive_out = timed(frozen_search, data)
        verification_queries = data[1].clone()
        try:
            candidate_ms, candidate_out = timed(candidate.ivf_pq_search, data)
            validate_output(workload, candidate_out, int(base.shape[0]))
            if candidate_out[1].device != data[1].device:
                raise ValueError('search outputs must remain on the input GPU')
            recall = iso_recall(candidate_out[1], naive_out[1])
            if recall < float(config['recall_threshold']):
                return {'id': workload['id'], 'valid': False, 'reason': 'recall_below_0.99'}
        except (ValueError, TypeError, IndexError) as error:
            return {'id': workload['id'], 'valid': False, 'reason': type(error).__name__ + ': ' + str(error)[:160]}
        truth = adc_distances(fresh_index(index), verification_queries, candidate_out[1])
        errors = distance_errors(candidate_out[0], candidate_out[1], truth, config['distance_check'])
        if errors['distance_gate_passed'] is False:
            return {'id': workload['id'], 'valid': False, 'reason': 'distance_inconsistency', 'distance_errors': errors}
        # Beat spec v3: recall@k against the exact nearest neighbor of the
        # uncompressed database (outside timing, on the actual queries).
        nearest = exact_nearest_distances(base, verification_queries)
        found = nn_recall(base, verification_queries, candidate_out[1], nearest)
        sample = {'naive_ms': naive_ms, 'candidate_ms': candidate_ms, 'recall': recall, 'nn_recall': found,
                  'distance_errors': errors}
        if config['distance_check']['mode'] == 'calibration':
            naive_truth = adc_distances(fresh_index(index), verification_queries, naive_out[1])
            sample['naive_distance_errors'] = distance_errors(naive_out[0], naive_out[1], naive_truth, config['distance_check'])
            del naive_truth
        samples.append(sample)
        del data, naive_out, candidate_out, verification_queries, truth, nearest
        torch.cuda.empty_cache()
    return summarize_samples(workload['id'], samples)


def main(argv: list[str]) -> int:
    if len(argv) != 5:
        raise ValueError('worker requires source, oracle, workload JSON, config JSON and result path')
    result = measure(Path(argv[0]), Path(argv[1]), json.loads(argv[2]), json.loads(argv[3]))
    Path(argv[4]).write_text(json.dumps(result, allow_nan=False, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
