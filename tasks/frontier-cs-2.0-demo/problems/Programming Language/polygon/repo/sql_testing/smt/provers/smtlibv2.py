import datetime
import fractions
import subprocess
import traceback

from subprocess import Popen, PIPE

from sql_testing.errors import SMTSolverError
from sql_testing.logger import logger
from sql_testing.smt.ast import *


class SMTLIBv2Visitor:
    def __init__(self):
        pass

    def visit_SMTCell(self, node):
        return f'(cell {node.table_id} {node.row_id} {node.column_id})'

    def visit_SMTNull(self, node):
        return f'(null {node.table_id} {node.row_id} {node.column_id})'

    def visit_SMTGrouping(self, node):
        return f'(grouping {node.table_id} {node.tuple_id} {node.group_id})'

    def visit_Deleted(self, node):
        return f'(deleted {node.table_id} {node.tuple_id})'

    def visit_SMTBelongsToGroup(self, node):
        return f'(belongstogroup {node.qid} {node.gid})'

    def visit_SMTSize(self, node):
        return f'(size {node.table_id})'

    def visit_Choice(self, node):
        return f'(choice {node.table_id} {node.bit_id})'

    def visit_And(self, node):
        if len(node.conjunct) == 0:
            return 'true'
        # return f'(and {" ".join([x.accept(self) for x in node.conjunct])})'

        try:
            result = []
            for x in node.conjunct:
                value = x.accept(self)
                if value == 0:
                    result.append('false')
                elif isinstance(value, int):
                    result.append('true')
                else:
                    result.append(value)
            return f'(and {" ".join(result)})'
        except:
            raise Exception([str(x.accept(self)) for x in node.conjunct])

    def visit_Or(self, node):
        if len(node.disjunct) == 0:
            return 'true'
        return f'(or {" ".join([x.accept(self) for x in node.disjunct])})'

    def visit_Xor(self, node):
        return f'(xor {node.a.accept(self)} {node.b.accept(self)})'

    def visit_Not(self, node):
        return f'(not {node.node.accept(self)})'

    def visit_Implies(self, node):
        return f'(=> {node.premise.accept(self)} {node.conclusion.accept(self)})'

    def visit_If(self, node):
        return f'(ite {node.a.accept(self)} {node.b.accept(self)} {node.c.accept(self)})'

    def visit_Gt(self, node):
        return f'(> {node.a.accept(self)} {node.b.accept(self)})'

    def visit_Gte(self, node):
        return f'(>= {node.a.accept(self)} {node.b.accept(self)})'

    def visit_Lt(self, node):
        return f'(< {node.a.accept(self)} {node.b.accept(self)})'

    def visit_Lte(self, node):
        return f'(<= {node.a.accept(self)} {node.b.accept(self)})'

    def visit_Eq(self, node):
        return f'(= {node.a.accept(self)} {node.b.accept(self)})'

    def visit_Neq(self, node):
        return f'(not (= {node.a.accept(self)} {node.b.accept(self)}))'

    def visit_Plus(self, node):
        return f'(+ {node.a.accept(self)} {node.b.accept(self)})'

    def visit_Minus(self, node):
        return f'(- {node.a.accept(self)} {node.b.accept(self)})'

    def visit_Mul(self, node):
        return f'(* {node.a.accept(self)} {node.b.accept(self)})'

    def visit_Div(self, node):
        return f'(div {node.a.accept(self)} {node.b.accept(self)})'

    def visit_Neg(self, node):
        return f'(- {node.x.accept(self)})'

    def visit_Int(self, node):
        return node.x

    def visit_Bool(self, node):
        if node.x:
            return 'true'
        return 'false'

    def visit_PbEq(self, node):
        coefficients = ' '.join(map(str, node.coefficients))
        args = ' '.join([arg.accept(self) for arg in node.args])
        return f'((_ pbeq {node.k} {coefficients}) {args})'

    def visit(self, node):
        print(type(node))
        raise NotImplementedError


class SMTLIBv2:
    def __init__(self, executable_path, executable_options=None, theory='QF_UFNIA'):  # QF_UFNIRA
        self.executable_path = executable_path
        if executable_options is None:
            executable_options = []
        self.executable_options = executable_options
        self.theory = theory

        self.visitor = SMTLIBv2Visitor()
        self.smt_process = None
        self.parser_env = None
        self.model = None
        self.unsat_core = None
        self.cell = None
        self.null = None
        self.grouping = None
        self.size = None

        self.checking_time = 0
        self.unsat_core_time = 0

    def check(self, formula: str):
        # (set-option :smt.core.minimize true)
        smt2 = f'''
(set-logic {self.theory})
(set-option :produce-models true)
(set-option :produce-unsat-cores true)
(set-option :smt.arith.solver 2)
(set-option :smt.arith.random_initial_value true)
(set-option :smt.phase_selection 2)

(declare-fun cell (Int Int Int) Int)
(declare-fun null (Int Int Int) Bool)
(declare-fun grouping (Int Int Int) Bool)
(declare-fun deleted (Int Int) Bool)
(declare-fun choice (Int Int) Int)
(declare-fun size (Int) Int)

(declare-fun belongstogroup (Int Int) Bool)

{formula}

(check-sat)
        '''

        try:
            self.smt_process = Popen(
                [self.executable_path, *self.executable_options],
                stdin=PIPE,
                stdout=PIPE,
                universal_newlines=True
            )
            self.smt_process.stdin.write(smt2)
            self.smt_process.stdin.flush()

            start = datetime.datetime.now()
            state = self.smt_process.stdout.readline().strip()
            self.checking_time = (datetime.datetime.now() - start).total_seconds()

            while state not in ['sat', 'unsat']:
                if 'error' in state.lower() or 'unsupported' in state.lower():
                    raise SMTSolverError(state)
                elif 'warning' in state.lower():
                    logger.warning(f'Solver msg: {state}')
                state = self.smt_process.stdout.readline().strip()

            if state == 'sat':
                return True
            elif state == 'unsat':
                start = datetime.datetime.now()
                self.smt_process.stdin.write('(get-unsat-core)\n')
                self.smt_process.stdin.flush()
                self.unsat_core = self.smt_process.stdout.readline().strip()[1:-1].split()

                self.unsat_core_time = (datetime.datetime.now() - start).total_seconds()

                if len(self.unsat_core) == 0:
                    self.unsat_core = None

                return False
            else:
                logger.error(f'Solver msg: {state}')
                return False
        except Exception as e:
            logger.error(''.join(traceback.format_tb(e.__traceback__)) + str(e))
            raise SMTSolverError

    def evaluate(self, term: str, args: list = None):
        command = f'(eval ({term}'
        if args is not None:
            command += ' ' + ' '.join([str(v) for v in args])
        command += '))'

        self.smt_process.stdin.write(f'{command}\n')
        self.smt_process.stdin.flush()
        out = self.smt_process.stdout.readline().rstrip()
        if out.startswith('(-'):
            out = '-' + out[3:-1]
        return out

    def evaluate_choice_vector(self, table):
        table_id = table.table_id
        vec = []
        if table.lineage is not None and 'Grouped' in table.lineage:
            vec_size = table.bound * 2
        else:
            vec_size = table.bound
        for bit_id in range(vec_size):
            bit = self.evaluate('choice', [table_id, bit_id])
            try:
                vec.append(int(bit))
            except ValueError:
                vec.append('T')
        return vec

    def evaluate_table(self, table, db, env):
        cex = []

        # table header
        table_id = table.table_id
        cex.append([column.column_name for column in db.schemas[table_id]])

        for tuple_id in range(table.bound):
            # print(f'choice({table_id}, {tuple_id})', self.evaluate('choice', [table_id, tuple_id]))

            # if table_id in [2, 3]:
            #     print(f'grouping({table_id}, {tuple_id}, 2)', self.evaluate('grouping', [table_id, tuple_id, 2]))
            deleted = self.evaluate('deleted', [table_id, tuple_id])
            if deleted == 'true':
                continue

            row = []
            for column in db.schemas[table_id]:
                if self.evaluate('null', [table_id, tuple_id, column.column_id]) == 'true':
                    row.append(None)
                else:
                    value = int(self.evaluate('cell', [table_id, tuple_id, column.column_id]))
                    if column.column_type is None:
                        column_type = 'int'
                    else:
                        column_type = column.column_type.lower()
                    if 'char' in column_type:
                        column_type = 'varchar'
                    match column_type:
                        case 'varchar' | 'text':
                            row.append(env.lookup_string(value))
                        case 'date':
                            date = datetime.date(1000, 1, 1)
                            try:
                                date = date + datetime.timedelta(days=value)
                            except OverflowError:
                                if value > 0:
                                    date = datetime.date(9999, 12, 31)
                                else:
                                    date = datetime.date(1, 1, 1)
                            row.append(str(date))
                        case 'time':
                            date = datetime.datetime(1900, 1, 1, 0, 0, 0)
                            try:
                                date = date + datetime.timedelta(seconds=value)
                            except OverflowError:
                                if value > 0:
                                    date = datetime.datetime(1900, 1, 1, 23, 59, 59)
                                else:
                                    date = datetime.datetime(1900, 1, 1, 0, 0, 0)
                            date = date.time()
                            row.append(str(date))
                        case 'bool':
                            if value == 0:
                                row.append(False)
                            else:
                                row.append(True)
                        case _:
                            row.append(value)
            cex.append(row)
        return cex


if __name__ == '__main__':
    prover = SMTLIBv2(
        executable_path='z3',
        executable_options=['--in', f'-T:5'],
    )
    visitor = SMTLIBv2Visitor()

    formula = And([
        SMTNull(2, 1, 1) == Bool(True),
        SMTCell(2, 1, 1) == Int(1234)
    ])

    if prover.check(f'(assert {formula.accept(visitor)})'):
        print(prover.evaluate('null', [2, 1, 1]))
        print(prover.evaluate('cell', [2, 1, 1]))
    else:
        print('unsat')
