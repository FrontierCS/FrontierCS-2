import argparse
import logging
import multiprocessing
import os
import traceback
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import ujson
import pandas as pd
from collections import defaultdict
from datetime import datetime

from prettytable import PrettyTable
from tqdm import tqdm

from sql_testing.constraint_parser import ConstraintParser
from sql_testing.environment import Environment
from sql_testing.logger import logger
from evaluation.calcite.tester import MySQLTester
from sql_testing.utils import chunkify

logger.setLevel(logging.ERROR)

BENCHMARK_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'benchmarks')
DB_CONFIG = {'host': 'localhost', 'user': 'root', 'password': 'pinhan'}

verieql_out = 'benchmarks/leetcode/leetcode_RQ1_2023_03_27_23_07_03.out'
verieql_rejected = defaultdict(list)  # maps a problem id to all user-submitted queries rejected
verieql_time = {}

with open(f"{os.path.join(os.path.dirname(__file__), '..', '..')}/{verieql_out}") as f:
    for line in f:
        try:
            line = ujson.loads(line)
            problem_id = line['file'][line['file'].rfind('/') + 1:line['file'].rfind('.')]
            if line['states'] is not None and 'NEQ' in line['states'] and line['counterexample'] is not None:
                verieql_rejected[problem_id].append(line['pair'][1].upper().replace("'", ''))
            if line['states'] is not None and ('NEQ' in line['states'] or 'EQU' in line['states']):
                verieql_time[(problem_id, line['pair'][1].upper().replace("'", ''))] = line['times']
        except:
            pass


def leetcode_worker(worker_id, problem_id, groundtruth, user_queries, bound):
    refuted = 0
    err = 0
    diff_verieql = 0

    benchmark_dir = os.path.join(BENCHMARK_DIR, 'leetcode')

    schema = ujson.load(open(f'{benchmark_dir}/schemas/{problem_id}.json'))
    groundtruths = pd.read_csv(f'{benchmark_dir}/groundtruths/{problem_id}.csv')

    constraint_parser = ConstraintParser()
    constraints = constraint_parser.parse_from_file(f"{benchmark_dir}/constraints/{problem_id}.yml")

    table = PrettyTable([
        'ProblemID', 'QueryID', '?Refuted', 'Counterexample', 'Time', '?VeriEQLRefuted', 'VeriEQL time',
        '#Iters', '#Backtracking', '%Backtracking Type 2', 'Unsat core sizes', 'M sizes', 'solving times',
        'AST size', 'nodes changed'
    ])

    for qid, query in tqdm(user_queries, desc=f'{problem_id} worker {worker_id}'):
        default_k = {
            'filter': 4,
            'project': 4,
            'union': 4,
            'inner join': 4,
            'left join': 4,
            'right join': 4,
            'full join': 4,
            'product': 4,
            'order by': 2
        }
        env = Environment(schema['Tables'], constraints, bound=bound, default_k=default_k, time_budget=60)

        if query.upper().replace("'", '') in verieql_rejected[str(problem_id)]:
            verieql_refuted = True
        else:
            verieql_refuted = False

        if (str(problem_id), query.upper().replace("'", '')) in verieql_time:
            idx = min(bound - 1, len(verieql_time[(str(problem_id), query.upper().replace("'", ''))]) - 1)
            if verieql_time[(str(problem_id), query.upper().replace("'", ''))][idx] is None or \
                    any(x is None for x in
                        verieql_time[(str(problem_id), query.upper().replace("'", ''))][idx]):
                veq_time = None
            else:
                veq_time = sum(verieql_time[(str(problem_id), query.upper().replace("'", ''))][idx])
        else:
            # print((str(problem_id), query.upper().replace("'", '')))
            veq_time = None

        try:
            eq, cex, checking_time, total_time, ret = env.check(groundtruth, query)
        except Exception as e:
            logger.error(''.join(traceback.format_tb(e.__traceback__)) + str(e))
            eq = None
        rejected = False
        if eq is not None:
            if not eq:
                with MySQLTester(DB_CONFIG, schema['Tables'], db_predix=f'leetcode_{worker_id}') as tester:
                    tester.create_all_databases([cex], constraints)
                    rejected = tester.test_pair(groundtruth, query)
                if rejected:
                    refuted += 1
            table.add_row((problem_id, qid, rejected, cex, total_time, verieql_refuted, veq_time,
                           ret['iters'], ret['backtracks'],
                           ret['type_2_backtracks'] / ret['backtracks'] if ret['backtracks'] > 0 else None,
                           ret['unsat_core_sizes'],
                           ret['M_sizes'],
                           str(ret['solving_time_per_iter']),
                           ret['ast_size'],
                           ret['num_nodes_changed']
                           ))
        else:
            table.add_row((problem_id, qid, None, None, None, verieql_refuted, veq_time,
                           None, None, None,
                           None,
                           None, None,
                           None, None
                           ))
            err += 1
        if verieql_refuted and not rejected:
            diff_verieql += 1
            logger.warning('failed to reject ' + query)

    with open(f'experiment_result_lc_{problem_id}_{worker_id}.csv', 'w', newline='') as f:
        f.write(table.get_csv_string())

    return os.getpid(), refuted, err, diff_verieql


def run_leetcode_problem(problem_id, bound):
    benchmark_dir = os.path.join(BENCHMARK_DIR, 'leetcode')

    schema = ujson.load(open(f'{benchmark_dir}/schemas/{problem_id}.json'))
    groundtruths = pd.read_csv(f'{benchmark_dir}/groundtruths/{problem_id}.csv')

    constraint_parser = ConstraintParser()
    constraints = constraint_parser.parse_from_file(f"{benchmark_dir}/constraints/{problem_id}.yml")

    table = PrettyTable([
        'ProblemID', 'QueryID', '?Refuted', 'Time', '?VeriEQLRefuted', 'VeriEQL time',
        '#Backtracking', '%Backtracking Type 2', 'Unsat core size', '%unsat core', 'solving times'
    ])

    queries = pd.read_csv(f'{benchmark_dir}/verieql_queries/{problem_id}.csv')

    problem_benchmarks = []
    for _, query in queries.iterrows():
        problem_benchmarks.append((query['qid'], query['mysql_query']))

    NUM_PROCESSES = 10
    pool = multiprocessing.Pool(processes=NUM_PROCESSES)

    results = []
    for idx, chunk in enumerate(chunkify(problem_benchmarks, NUM_PROCESSES)):
        results.append(pool.apply_async(leetcode_worker, args=(idx, problem_id, groundtruths['mysql_groundtruth'][0], chunk, bound)))

    pool.close()
    pool.join()

    refuted = 0
    err = 0
    diff_verieql = 0

    pids = []
    for result in results:
        pid, p_refuted, p_err, p_diff_verieql = result.get()
        pids.append(pid)
        refuted += p_refuted
        err += p_err
        diff_verieql += p_diff_verieql

    # combine into one csv
    csv_files = [f'experiment_result_lc_{problem_id}_{worker_id}.csv' for worker_id in range(NUM_PROCESSES)]
    first_csv = pd.read_csv(csv_files[0])
    header = list(first_csv.columns)

    combined_data = pd.DataFrame(columns=header)

    for csv_file in csv_files:
        data = pd.read_csv(csv_file)
        combined_data = pd.concat([combined_data, data], ignore_index=True)

    combined_data.to_csv(f'experiment_result_lc_{problem_id}.csv', index=False)

    # clear
    for csv_file in csv_files:
        os.remove(csv_file)

    for pid in pids:
        os.system(f'pkill -9 -P {pid}')
        os.system(f'pkill -9 -g {pid}')
        os.system(f'pkill -9 -G {pid}')

    return problem_id, queries.shape[0], refuted, err, len(verieql_rejected[str(problem_id)]), diff_verieql


def leetcode(subset):
    output_file_name = 'experiment_result_lc.csv'
    if subset:
        problems = [1581]
    else:
        problems = [619, 595, 175, 182, 511, 577, 584, 596, 1050, 1378, 1587,
                    512, 183, 613, 1045, 1264, 1350, 1421, 1581, 1693, 1715, 1783,
                    607, 610, 1083, 1084, 1113, 1141, 1440, 1777, 1747, 1809,
                    550, 585, 1075, 1097, 1173, 1211, 1251, 1468, 1661,
                    1789, 1795, 1127, 1435,  # union
                    603, 614, 1148, 1149, 1364, 1398, 1677, 1731,  # order by
                    ]

    table = PrettyTable([
        'ProblemID', '#Total', '#Refuted', '#Err', '#VeriEQLRejected', '#DiffVeriEQL'
    ])

    logger.info('experiment started')

    for problem_id in problems:
        bound = 2
        if problem_id in [619, 1050, 1398]:
            bound = 3
        if problem_id in [1661]:
            bound = 4
        if problem_id in [596, 1149]:
            bound = 5
        line = run_leetcode_problem(problem_id, bound=bound)
        table.add_row(line)

    print(table)

    problem_csv_files = [
        f'experiment_result_lc_{problem_id}.csv' for problem_id in problems
    ]
    combined_df = pd.concat([pd.read_csv(file) for file in problem_csv_files])
    combined_df.to_csv(f"{output_file_name}_detailed.csv", index=False)
    for csv_file in problem_csv_files:
        if os.path.exists(csv_file):
            os.remove(csv_file)

    with open(output_file_name, 'w', newline='') as f:
        f.write(table.get_csv_string())

    logger.info('experiment ends')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--subset', action='store_true')

    args = parser.parse_args()
    print(args)

    leetcode(args.subset)
