"""KNN-specific GPU sampling, paired timing and ball-recall verification."""
from __future__ import annotations
import importlib
import importlib.util
import sys
import time
from pathlib import Path


def validate_neighbor_rows(distances, indices, database_rows):
    """Check distinct real neighbors; padding is only legal after the full base."""
    import math
    for values, ids in zip(distances, indices):
        real = min(len(ids), database_rows)
        if len(values) != len(ids) or len(set(ids[:real])) != real:
            raise ValueError('duplicate valid neighbors or mismatched output rows')
        if any(type(index) is not int or not 0 <= index < database_rows for index in ids[:real]):
            raise ValueError('neighbor index is outside the database')
        if any(not math.isfinite(value) or value < 0 for value in values[:real]):
            raise ValueError('valid neighbor distances must be finite and nonnegative')
        if any(left > right for left, right in zip(values[:real], values[1:real])):
            raise ValueError('neighbor distances must be in ascending order')
        if any(index != -1 or value != math.inf for index, value in zip(ids[real:], values[real:])):
            raise ValueError('padding must be trailing index -1 with distance +inf')


def validate_output(output, queries, database, k, *, baseline=False):
    import torch
    if not isinstance(output, (tuple, list)) or len(output) != 2:
        raise ValueError('KNN must return distances and indices')
    distances, indices = output
    if not isinstance(distances, torch.Tensor) or not isinstance(indices, torch.Tensor):
        raise ValueError('KNN outputs must be tensors')
    expected = (queries.shape[0], k)
    if tuple(distances.shape) != expected or tuple(indices.shape) != expected:
        raise ValueError('KNN output shape is invalid')
    if distances.device != queries.device or indices.device != queries.device:
        raise ValueError('KNN outputs must remain on the input device')
    # Preserve the original naive cdist result dtype; submissions return fp32.
    allowed_distances = (torch.float32, torch.bfloat16) if baseline else (torch.float32,)
    if distances.dtype not in allowed_distances or indices.dtype != torch.int64:
        raise ValueError('KNN output dtype is invalid')
    validate_neighbor_rows(distances.tolist(), indices.tolist(), database.shape[0])
    return distances, indices


def ball_recall(queries, database, returned_indices, naive_indices, tolerance=0.0):
    """Use the legacy naive kth neighbor's recomputed fp32 radius plus 1e-6."""
    import torch
    qf, dbf = queries.to(torch.float32), database.to(torch.float32)
    count = min(returned_indices.shape[1], database.shape[0])
    if count == 0 or queries.shape[0] == 0:
        raise ValueError('nonempty queries and database are required')
    kth = dbf.index_select(0, naive_indices[:, count - 1].long())
    radius = ((qf - kth) ** 2).sum(1) * (1 + tolerance) + 1e-6
    hits = 0
    for start in range(0, queries.shape[0], 256):
        ids = returned_indices[start:start + 256, :count]
        neighbors = dbf.index_select(0, ids.reshape(-1)).reshape(ids.shape[0], count, -1)
        distances = ((qf[start:start + 256, None, :] - neighbors) ** 2).sum(2)
        hits += int((distances <= radius[start:start + 256, None]).sum().item())
    return hits / (queries.shape[0] * count)


def exact_kth_distances(queries, database, k, block=128):
    """FP64 squared distance of each query's exact k-th nearest database row.

    Candidates are ranked by an FP32 brute force over the actual bf16 inputs and
    the k selected rows are recomputed with direct FP64 differences, so the
    radius is never below the true k-th distance.
    """
    import torch
    count = min(k, database.shape[0])
    dbf = database.to(torch.float32)
    radius = torch.empty(queries.shape[0], dtype=torch.float64, device=queries.device)
    for start in range(0, queries.shape[0], block):
        q = queries[start:start + block].to(torch.float32)
        top = torch.cdist(q, dbf).topk(count, dim=1, largest=False).indices
        selected = database.index_select(0, top.reshape(-1)).reshape(q.shape[0], count, -1).to(torch.float64)
        exact = ((q.to(torch.float64)[:, None, :] - selected) ** 2).sum(2)
        radius[start:start + block] = exact.max(1).values
    return radius


def exact_recall(queries, database, returned_indices, radius, block=64):
    """Share of returned neighbors within the exact k-th nearest distance (FP64)."""
    import torch
    count = min(returned_indices.shape[1], database.shape[0])
    if count == 0 or queries.shape[0] == 0:
        raise ValueError('nonempty queries and database are required')
    hits = 0
    for start in range(0, queries.shape[0], block):
        ids = returned_indices[start:start + block, :count]
        selected = database.index_select(0, ids.reshape(-1)).reshape(ids.shape[0], count, -1).to(torch.float64)
        q = queries[start:start + block].to(torch.float64)
        exact = ((q[:, None, :] - selected) ** 2).sum(2)
        hits += int((exact <= radius[start:start + block, None]).sum().item())
    return hits / (queries.shape[0] * count)


def load_ann(data_root, name, device):
    import h5py
    import numpy as np
    import torch
    expected = {'sift': ((1000000, 128), (10000, 128)),
                'gist': ((1000000, 960), (1000, 960))}
    with h5py.File(Path(data_root) / f'{name}.hdf5', 'r') as source:
        if (tuple(source['train'].shape), tuple(source['test'].shape)) != expected[name]:
            raise ValueError('real ANN dataset shape differs from its original corpus')
        base = torch.from_numpy(np.asarray(source['train'], dtype=np.float32)).to(device)
        queries = torch.from_numpy(np.asarray(source['test'], dtype=np.float32)).to(device)
    return base.contiguous(), queries.contiguous()


def sample_inputs(workload, seed, base, queries_all):
    import torch
    generator = torch.Generator(device=base.device).manual_seed(seed)
    qperm = torch.randperm(queries_all.shape[0], generator=generator, device=base.device)[:workload['Q']]
    dperm = torch.randperm(base.shape[0], generator=generator, device=base.device)[:workload['M']]
    return (queries_all.index_select(0, qperm).to(torch.bfloat16).contiguous(),
            base.index_select(0, dperm).to(torch.bfloat16).contiguous())


def iteration_seeds(workload, warmups, iterations):
    seed = workload['base_seed'] + 1000 * (workload['seed_index'] + 1)
    return ([seed + 100 + index for index in range(warmups)],
            [seed + 10000 + index for index in range(iterations)])


def measure_workload(source_root, workload, warmups, iterations, naive_path, data_root, *, distance_policy=None):
    import torch
    perf, synchronize = time.perf_counter, torch.cuda.synchronize
    if not torch.cuda.is_available() or 'H100' not in torch.cuda.get_device_name():
        raise RuntimeError('KNN evaluation requires the declared H100')
    spec = importlib.util.spec_from_file_location('frontier_naive_knn', naive_path)
    naive = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(naive)
    baseline_call = naive.knn
    base, queries_all = load_ann(data_root, workload['dataset'], 'cuda')
    generate, validate, quality, distance_check = sample_inputs, validate_output, ball_recall, distance_error_metrics
    sys.path.insert(0, str(source_root))
    implementation = importlib.import_module('knnlib').knn
    warmup_seeds, timed_seeds = iteration_seeds(workload, warmups, iterations)
    k = workload['k']

    def timed(call, queries, database):
        synchronize()
        started = perf()
        output = call(queries, database, k)
        synchronize()
        return (perf() - started) * 1000, output

    with torch.inference_mode():
        for seed in warmup_seeds:
            queries, database = generate(workload, seed, base, queries_all)
            baseline_call(queries, database, k)
            implementation(queries.clone(), database.clone(), k)
            synchronize()
            del queries, database
        torch.cuda.empty_cache()
        row = {'id': workload['id'], 'hardware': {'gpu': torch.cuda.get_device_name(), 'torch_version': torch.__version__, 'input_dtype': 'bfloat16'},
               'naive_latency_ms': [], 'latency_ms': [], 'ball_recall': [], 'exact_recall': [], 'distance_errors': []}
        for seed in timed_seeds:
            queries, database = generate(workload, seed, base, queries_all)
            # Keep pristine input tensors for the oracle outside the timed call.
            submitted_queries, submitted_database = queries.clone(), database.clone()
            naive_ms, naive_output = timed(baseline_call, queries, database)
            candidate_ms, output = timed(implementation, submitted_queries, submitted_database)
            validate(naive_output, queries, database, k, baseline=True)
            validate(output, queries, database, k)
            recall = quality(queries, database, output[1], naive_output[1])
            row['naive_latency_ms'].append(naive_ms)
            row['latency_ms'].append(candidate_ms)
            row['ball_recall'].append(recall)
            # Beat spec v3: recall against the exact k nearest rows, outside timing.
            row['exact_recall'].append(exact_recall(queries, database, output[1],
                                                    exact_kth_distances(queries, database, k)))
            atol, rtol = distance_policy if distance_policy is not None else (None, None)
            row['distance_errors'].append(distance_check(queries, database, output, absolute_tolerance=atol, relative_tolerance=rtol))
            del queries, database, submitted_queries, submitted_database, naive_output, output
            torch.cuda.empty_cache()
    return row


def distance_error_metrics(queries, database, output, *, absolute_tolerance=None, relative_tolerance=None):
    """Recompute selected bf16 squared-L2 distances using direct FP64 differences."""
    import torch
    distances, indices = output
    count = min(indices.shape[1], database.shape[0])
    max_abs = max_relative = zero_max_abs = 0.0
    zero_count = zero_nonzero_count = compared = violations = 0
    calibrated = absolute_tolerance is not None and relative_tolerance is not None
    for start in range(0, queries.shape[0], 64):
        ids = indices[start:start + 64, :count]
        # Only selected rows are widened; no full Q-by-M distance matrix is built.
        selected = database.index_select(0, ids.reshape(-1)).reshape(ids.shape[0], count, -1).to(torch.float64)
        q = queries[start:start + 64].to(torch.float64)
        exact = ((q[:, None, :] - selected) ** 2).sum(2)
        reported = distances[start:start + 64, :count].to(torch.float64)
        error = (reported - exact).abs()
        nonzero = exact != 0
        zeros = ~nonzero
        compared += exact.numel()
        max_abs = max(max_abs, float(error.max().item()))
        if nonzero.any():
            max_relative = max(max_relative, float((error[nonzero] / exact[nonzero].abs()).max().item()))
        if zeros.any():
            zero_count += int(zeros.sum().item())
            zero_nonzero_count += int((error[zeros] != 0).sum().item())
            zero_max_abs = max(zero_max_abs, float(error[zeros].max().item()))
        if calibrated:
            violations += int((error > absolute_tolerance + relative_tolerance * exact.abs()).sum().item())
    return {'oracle': 'direct_squared_l2_fp64_from_actual_bf16_inputs',
            'compared_distances': compared, 'max_absolute_error': max_abs,
            'max_relative_error_nonzero': max_relative, 'zero_distance_count': zero_count,
            'zero_distance_nonzero_error_count': zero_nonzero_count, 'zero_distance_max_absolute_error': zero_max_abs,
            'tolerance_configured': calibrated, 'violations': violations if calibrated else None}
