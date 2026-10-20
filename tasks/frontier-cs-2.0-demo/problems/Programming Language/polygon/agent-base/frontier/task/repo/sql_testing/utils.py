import errno
import signal
import functools
from z3 import *

from sql_testing.schemas import TableSchema, ColumnSchema


def all_smt(s, initial_terms):
    def block_term(s, m, t):
        s.add(t != m.eval(t, model_completion=True))

    def fix_term(s, m, t):
        s.add(t == m.eval(t, model_completion=True))

    def all_smt_rec(terms):
        if sat == s.check():
            m = s.model()
            yield m
            for i in range(len(terms)):
                s.push()
                block_term(s, m, terms[i])
                for j in range(i):
                    fix_term(s, m, terms[j])
                yield from all_smt_rec(terms[i:])
                s.pop()

    yield from all_smt_rec(list(initial_terms))


def copy_group_by_mapping(copy_from: TableSchema, copy_to: TableSchema, env):
    for group_id in range(max(copy_from.ctx['groups_considered'])):
        for tuple_id in range(copy_from.ancestors[0].bound):
            env.formulas.append(
                env.grouping(copy_to.table_id, tuple_id, group_id) == env.grouping(copy_from.table_id, tuple_id, group_id),
                label='misc'
            )

    env.formulas.append(env.size(copy_to.table_id) == env.size(copy_from.table_id), label='misc')
    copy_to.ctx['groups_considered'] = copy_from.ctx['groups_considered']
    copy_to.lineage = copy_from.lineage


def create_empty_table(row: int, col: int, env):
    table_id = env.next_table_id()
    table_name = f'empty_table_{row}x{col}'
    table_schema = TableSchema(table_id, table_name, row)

    for column_id in range(col):
        column_schema = ColumnSchema(column_id, str(column_id), 'int', table_name=table_name)
        table_schema.append(column_schema)

    return table_schema


def timeout(seconds=10, error_message=os.strerror(errno.ETIME)):
    def decorator(func):
        def _handle_timeout(signum, frame):
            raise TimeoutError(error_message)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
            return result

        return wrapper

    return decorator


def chunkify(lst, n):
    """Partition a list into n approximately equal-sized chunks."""
    size = len(lst) // n
    remainder = len(lst) % n
    chunks = [lst[i * size + min(i, remainder):(i + 1) * size + min(i + 1, remainder)] for i in range(n)]
    return chunks
