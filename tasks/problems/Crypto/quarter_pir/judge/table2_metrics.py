"""Gate decisions for one paired QuarterPIR experiment.

Every number here comes from the trusted conductor's result record or the
trusted server-time replay; nothing is parsed from submission-writable output.
"""
import math

# One workload per experiment; every experiment is judged on every metric.
GATED_METRICS = (
    'online_communication_bytes',
    'server_compute_time_ns',
    'preprocessing_client_time_minutes',
    'end_to_end_online_time_ms',
    'client_storage_bytes',
)
# Measurement jitter of the live-heap storage measure, not an allowance for
# real storage. Repeated reference runs spread over 19,344 bytes at 65,536
# entries (14 runs) and 10,232 bytes at 67,108,864 entries (6 runs); this is the
# next power of two above the larger spread (see
# docs/tasks/quarter_pir/full-compute-scope.md).
STORAGE_JITTER_BYTES = 32768
# Growth of the reference client's live heap over the volume phase, from its
# last reading before that phase to the reading after the second-to-last
# volume query: the generator's query history and the client's per-query
# cache. The paired reference run skips the volume phase (its run time), so
# its storage there is its own reading before the phase plus this growth,
# measured on full reference runs (see docs/tasks/quarter_pir/full-compute-scope.md).
REFERENCE_VOLUME_GROWTH_BYTES = {65536: 83328, 67108864: 9461920}
# Incidental safety limit on the client process, far above any honest design.
PEAK_RSS_LIMIT_BYTES = 4 * 1024 * 1024 * 1024
SCORE_WEIGHTS = {'communication': 1 / 2, 'server': 1 / 8, 'preprocessing': 1 / 8, 'online': 1 / 8,
                 'storage': 1 / 8}
INT_FIELDS = ('db_size', 'actual_queries', 'planned_queries', 'threads',
              'offline_elapsed_ns', 'online_elapsed_ns', 'client_storage_bytes', 'client_storage_volume_bytes',
              'client_peak_rss_bytes', 'hint_export_bytes', 'offline_communication_bytes',
              'probe_queries', 'probe_answered', 'probe_volume_queries', 'audit_hints',
              'probe_online_ns', 'probe_volume_online_ns')
# Checks a coverage-probe query can fail, as the trusted conductor names them.
PROBE_FAILURES = frozenset((
    'client_crashed_or_protocol', 'request_not_reduction', 'reduction_nondeterministic', 'reducer_crashed',
    'server_rpc_failed', 'wrong_value', 'message_longer_than_measured'))


def parameters(size, threads=4):
    """Public parameters of one database size, as util.GenParams and refwire derive them."""
    chunk = 1
    while chunk < int(2 * math.sqrt(size)):
        chunk *= 2
    sets = ((math.ceil(size / chunk) + 3) // 4) * 4
    super_size = 1
    while super_size < int(2 * math.sqrt(sets)):
        super_size *= 2
    super_num = ((math.ceil(sets / super_size) + 3) // 4) * 4
    sets = super_size * super_num
    planned = int(math.sqrt(size) * math.log(size))
    primary = int(math.ceil(math.log(2) * 41 + math.log(planned))) * chunk
    primary = (primary + threads - 1) // threads * threads
    backup = 3 * int(planned / sets)
    backup = (backup + threads - 1) // threads * threads
    return {'chunk_size': chunk, 'set_size': sets, 'superblock_size': super_size,
            'superblocks': super_num, 'planned_queries': planned,
            'primary_hints': primary, 'backup_hints': backup * sets,
            # refwire.Params.HintExportBytes: tag and parity per primary hint; tag, three
            # entries per backup hint; index and entry per replacement (one per backup hint).
            'hint_export_bytes': 20 * primary + 76 * backup * sets}


def reference_message_bytes(size):
    """(upload, download) of the reference at its information-tight encoding."""
    p = parameters(size)
    offset_bits = (p['chunk_size'] - 1).bit_length()
    return (16 * p['superblocks'] + (p['superblock_size'] * offset_bits + 7) // 8,
            2 * p['superblocks'] * 16)


def score_scales(size):
    """Fixed scales of the continuous score: public constants of the size, no measurement."""
    p = parameters(size)
    return {
        'communication': float(sum(reference_message_bytes(size))),                      # bytes per query
        'server': 50.0 * p['set_size'],                                                  # ns per query
        'preprocessing': 5.0 * p['set_size'] * (p['primary_hints'] + p['backup_hints']),  # ns per experiment
        'online': 50.0 * (p['chunk_size'] + p['set_size']),                              # ns per query
        'storage': float(p['hint_export_bytes']),                                        # bytes per experiment
    }


def continuous_score(rows):
    """Weighted geometric mean of non-saturating terms scale/(scale+value).

    rows: one dict per experiment with db_size, bytes (per-query list),
    server_ns (per-query list), offline_ns, online_ns (total over the
    uniformly random queries the run answered: the measured ones and the
    volume phase) and queries (their number), and storage (client storage
    over the online phase). Only the run's own measurements enter.
    """
    logs = {name: [] for name in SCORE_WEIGHTS}
    for row in rows:
        scale = score_scales(row['db_size'])
        values = {
            'communication': row['bytes'], 'server': row['server_ns'],
            'preprocessing': [row['offline_ns']],
            'online': [row['online_ns'] / row['queries']],
            'storage': [row['storage']],
        }
        for name, items in values.items():
            for value in items:
                if not math.isfinite(value) or value <= 0:
                    raise ValueError('invalid score measurement')
                logs[name].append(math.log(scale[name] / (scale[name] + value)))
    if any(not items for items in logs.values()):
        raise ValueError('incomplete score measurements')
    return 100.0 * math.exp(sum(SCORE_WEIGHTS[name] * sum(items) / len(items)
                                for name, items in logs.items()))


def check_result(result, *, size, queries=1000, threads=4):
    """Structural checks of one trusted conductor record; returns it."""
    if not isinstance(result, dict) or result.get('valid') is not True or result.get('failure') != '':
        raise ValueError('experiment record is not a valid completed run')
    if any(type(result.get(key)) is not int or not 0 <= result[key] < 2**63 for key in INT_FIELDS):
        raise ValueError('experiment fields must be bounded exact integers')
    expected = {'db_size': size, 'actual_queries': queries, 'threads': threads,
                'planned_queries': int(math.sqrt(size) * math.log(size)),
                'offline_communication_bytes': size * 16}
    if any(result[key] != value for key, value in expected.items()):
        raise ValueError('experiment identity or denominator differs')
    if result['offline_elapsed_ns'] <= 0 or result['online_elapsed_ns'] <= 0 or result['planned_queries'] <= 0:
        raise ValueError('nonpositive duration or denominator')
    for key in ('upload_bytes', 'download_bytes'):
        rows = result.get(key)
        if (not isinstance(rows, list) or len(rows) != queries
                or any(type(value) is not int or value <= 0 for value in rows)):
            raise ValueError('per-request transport byte counts are incomplete')
    for key in ('hint_export_sha256', 'transcript_sha256', 'probe_sha256'):
        value = result.get(key)
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError('missing digest')
    # Coverage probe: a depth part, the primary-hint audit, then a volume part
    # up to the planned query budget, which the paired reference run skips.
    # Either every query asked was answered or the record names the check that
    # the first unanswered one failed.
    failure, skipped = result.get('probe_failure'), result.get('probe_volume_skipped')
    depth = result['probe_queries'] - result['probe_volume_queries']
    if (depth <= 0 or result['actual_queries'] + result['probe_queries'] > result['planned_queries']
            or result['probe_volume_queries'] != max(0, result['planned_queries'] - result['actual_queries'] - depth)
            or type(skipped) is not bool or (skipped and (failure != '' or result.get('audit_failure') != ''))):
        raise ValueError('coverage probe is empty, exceeds or misses the planned budget, or skipped wrongly')
    if failure == '':
        asked = depth if skipped or result.get('audit_failure') != '' else result['probe_queries']
        if result['probe_answered'] != asked:
            raise ValueError('coverage probe is incomplete without a failure')
    elif failure not in PROBE_FAILURES or result['probe_answered'] >= result['probe_queries']:
        raise ValueError('coverage probe failure is malformed')
    # Probe clocks: positive exactly for the parts that answered queries.
    volume_answered = max(0, result['probe_answered'] - depth)
    if ((result['probe_online_ns'] > 0) != (min(result['probe_answered'], depth) > 0)
            or (result['probe_volume_online_ns'] > 0) != (volume_answered > 0)):
        raise ValueError('probe clocks do not match the answered probe queries')
    # Volume-phase storage readings: none when the phase was skipped, and
    # present when it completed.
    completed = failure == '' and not skipped and result.get('audit_failure') == ''
    if ((skipped and result['client_storage_volume_bytes'] != 0)
            or (completed and result['probe_volume_queries'] >= 2 and result['client_storage_volume_bytes'] == 0)):
        raise ValueError('volume-phase storage reading missing or unexpected')
    # Primary-hint audit: run exactly when the depth part completed.
    mismatched, digest = result.get('audit_mismatched'), result.get('audit_sha256')
    if result['probe_answered'] < depth:
        if result['audit_hints'] != 0 or mismatched != [] or result.get('audit_failure') != '':
            raise ValueError('primary-hint audit ran after a failed probe')
    elif (result['audit_hints'] <= 0 or not isinstance(digest, str) or len(digest) != 64
          or result.get('audit_failure') not in ('', 'client_crashed_or_protocol')
          or not isinstance(mismatched, list)
          or any(type(i) is not int or not 0 <= i < result['audit_hints'] for i in mismatched)
          or mismatched != sorted(set(mismatched))
          or (result['audit_failure'] != '' and mismatched)):
        raise ValueError('primary-hint audit record is malformed')
    return result


def audit_failure(candidate, reference):
    """Paired primary-hint audit verdict; None when the candidate passes.

    After the depth part of the probe both runs are asked for the canonical
    record (tag, set parity) of the same sample of primary hints that no query
    has consumed.
    A record counts against the candidate only when the reference client
    produced the canonical one for that hint.
    """
    if candidate['audit_hints'] == 0 or reference['audit_hints'] == 0:
        return None  # a probe stopped early; probe_failure decides
    if (candidate['audit_hints'] != reference['audit_hints']
            or candidate['audit_sha256'] != reference['audit_sha256']):
        raise ValueError('candidate/reference audit samples differ')
    if reference['audit_failure'] != '':
        return None
    if candidate['audit_failure'] != '':
        return candidate['audit_failure']
    if set(candidate['audit_mismatched']) - set(reference['audit_mismatched']):
        return 'record_not_canonical'
    return None


def probe_failure(candidate, reference):
    """Paired coverage-probe verdict; None when the candidate passes.

    Both runs are planned the identical additional query sequence after the
    measured queries. The candidate fails when it fails a query the reference
    answered. A query the reference itself failed, and everything after it,
    does not count against the candidate. The reference run skips the volume
    part, which it serves: no volume query lies in an exhausted chunk.
    """
    skips = (reference['probe_failure'] == '' and reference['audit_failure'] == ''
             and reference['probe_volume_queries'] > 0)
    if (candidate['probe_queries'] != reference['probe_queries']
            or candidate['probe_sha256'] != reference['probe_sha256']
            or candidate['probe_volume_skipped'] or reference['probe_volume_skipped'] != skips):
        raise ValueError('candidate/reference probe sequences differ')
    answered = reference['probe_queries'] if reference['probe_volume_skipped'] else reference['probe_answered']
    if candidate['probe_failure'] != '' and candidate['probe_answered'] < answered:
        return candidate['probe_failure']
    return None


def timing_values(result):
    """Reporting units only; decisions use the exact integer nanoseconds."""
    return {
        'preprocessing_client_time_minutes': result['offline_elapsed_ns'] / 60_000_000_000,
        'end_to_end_online_time_ms': result['online_elapsed_ns'] / (result['actual_queries'] * 1_000_000),
        'amortized_offline_time_ms': result['offline_elapsed_ns'] / (result['planned_queries'] * 1_000_000),
    }


def validity_failure(candidate, reference):
    """Paired validity conditions; None when the candidate run is valid."""
    if (candidate['hint_export_bytes'] != reference['hint_export_bytes']
            or candidate['hint_export_sha256'] != reference['hint_export_sha256']):
        return 'hint_export_mismatch'
    if candidate['client_peak_rss_bytes'] > PEAK_RSS_LIMIT_BYTES:
        return 'process_memory_limit'
    return None


def storage_peak(result):
    """Client storage of one run: the largest reading over the online phase."""
    return max(result['client_storage_bytes'], result['client_storage_volume_bytes'])


def reference_volume_storage(reference):
    """The reference's volume-phase reading. When its run took none (the paired
    run skips that phase), it is the reference's last reading before the phase
    plus the reference client's measured growth over the phase; the
    reference's storage only grows, so that last reading is its largest one."""
    volume = reference['client_storage_volume_bytes']
    if volume == 0:
        volume = reference['client_storage_bytes'] + REFERENCE_VOLUME_GROWTH_BYTES[reference['db_size']]
    return volume


def paired_online_ns(candidate, reference):
    """(candidate, reference) online time over the queries both runs answered
    identically: the measured queries plus the depth part of the probe when
    both runs completed it (the reference skips the volume phase)."""
    depth = candidate['probe_queries'] - candidate['probe_volume_queries']
    both = candidate['probe_answered'] >= depth and reference['probe_answered'] >= depth
    return tuple(run['online_elapsed_ns'] + (run['probe_online_ns'] if both else 0)
                 for run in (candidate, reference))


def score_online(result):
    """(online ns, queries) over the uniformly random queries a run answered:
    the measured ones and the volume phase of the probe (skipped by the paired
    reference run). The depth part is chosen adversarially and is compared
    paired instead."""
    depth = result['probe_queries'] - result['probe_volume_queries']
    volume = max(0, result['probe_answered'] - depth) if not result['probe_volume_skipped'] else 0
    return result['online_elapsed_ns'] + result['probe_volume_online_ns'], result['actual_queries'] + volume


def compare_experiment(candidate, reference, candidate_server_ns, reference_server_ns):
    """The five gate decisions of one experiment.

    Strictly fewer bytes and strictly lower replayed server time on every
    measured query; strict improvement on both experiment clocks, online time
    taken over the queries both runs answered (equality fails); client storage
    not above the reference's, up to the measurement jitter, both before and in
    the volume phase."""
    for key in ('db_size', 'threads', 'planned_queries', 'actual_queries'):
        if candidate[key] != reference[key]:
            raise ValueError('candidate/reference experiments differ')
    queries = candidate['actual_queries']
    if any(len(rows) != queries for rows in (candidate['upload_bytes'], candidate['download_bytes'],
                                             reference['upload_bytes'], reference['download_bytes'],
                                             candidate_server_ns, reference_server_ns)):
        raise ValueError('per-query measurements are incomplete')
    candidate_online, reference_online = paired_online_ns(candidate, reference)
    # Paired runs share positive denominators, so comparing the integer
    # nanosecond numerators is the exact normalized comparison.
    return {
        'online_communication_bytes': all(
            0 < up + down < reference_up + reference_down
            for up, down, reference_up, reference_down in zip(
                candidate['upload_bytes'], candidate['download_bytes'],
                reference['upload_bytes'], reference['download_bytes'])),
        'server_compute_time_ns': all(0 < mine < theirs
                                      for mine, theirs in zip(candidate_server_ns, reference_server_ns)),
        'preprocessing_client_time_minutes': candidate['offline_elapsed_ns'] < reference['offline_elapsed_ns'],
        'end_to_end_online_time_ms': candidate_online < reference_online,
        'client_storage_bytes': (candidate['client_storage_bytes']
                                 <= reference['client_storage_bytes'] + STORAGE_JITTER_BYTES
                                 and candidate['client_storage_volume_bytes']
                                 <= reference_volume_storage(reference) + STORAGE_JITTER_BYTES),
    }
