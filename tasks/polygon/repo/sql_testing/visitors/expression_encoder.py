import datetime
import functools
import time

from sql_testing.ast.expressions.attribute import Attribute
from sql_testing.ast.expressions.expression import Expression
from sql_testing.ast.expressions.literal import Literal
from sql_testing.ast.query import Query
from sql_testing.smt.ast import *

from sql_testing.ast.expressions.operator import Operator
from sql_testing.variables import Variable


def Max(args):
    def Max2(x, y):
        x_val, x_null, x_deleted = x
        y_val, y_null, y_deleted = y

        # val = If(
        #     Or([
        #         And([x_val <= y_val, Not(Or([x_null, y_null]))]),
        #         And([x_null, Not(y_null)])
        #     ]),
        #     y_val,
        #     x_val
        # )
        # null = And([x_null, y_null])
        # return val, null

        val = If(
            And([Not(x_deleted), y_deleted]),
            x_val,
            If(
                And([x_deleted, Not(y_deleted)]),
                y_val,
                If(
                    And([Not(x_deleted), Not(y_deleted)]),
                    If(
                        Or([
                            And([x_val <= y_val, Not(Or([x_null, y_null]))]),
                            And([x_null, Not(y_null)]),
                        ]),
                        y_val,
                        x_val
                    ),
                    Int(0)
                )
            )
        )
        null = And([Implies(Not(x_deleted), x_null), Implies(Not(y_deleted), y_null)])
        deleted = And([x_deleted, y_deleted])

        return val, null, deleted

    val, null, deleted = functools.reduce(Max2, args)
    return val, Or([null, deleted])


def Min(args):
    def Min2(x, y):
        x_val, x_null, x_deleted = x
        y_val, y_null, y_deleted = y

        val = If(
            And([Not(x_deleted), y_deleted]),
            x_val,
            If(
                And([x_deleted, Not(y_deleted)]),
                y_val,
                If(
                    And([Not(x_deleted), Not(y_deleted)]),
                    If(
                        Or([
                            And([x_val >= y_val, Not(Or([x_null, y_null]))]),
                            And([x_null, Not(y_null)]),
                        ]),
                        y_val,
                        x_val
                    ),
                    Int(0)
                )
            )
        )
        null = And([Implies(Not(x_deleted), x_null), Implies(Not(y_deleted), y_null)])
        deleted = And([x_deleted, y_deleted])

        return val, null, deleted

    val, null, deleted = functools.reduce(Min2, args)
    return val, Or([null, deleted])


def Count(args):
    val = Sum([If(And([Not(deleted), Not(null)]), Int(1), Int(0)) for _, null, deleted in args])
    return val, Bool(False)


def Count_Distinct(args):
    val = Sum([
        If(
            And([
                Not(deleted),
                Not(null),
                And([
                    Implies(
                        And([Not(prev_deleted), Not(prev_null)]),
                        val != prev_val
                    )
                    for prev_val, prev_null, prev_deleted in args[:idx]
                ])
            ]),
            Int(1),
            Int(0))
        for idx, (val, null, deleted) in enumerate(args)
    ])
    return val, Bool(False)


def AggSum(args):
    val = Sum([If(And([Not(deleted), Not(null)]), val, Int(0)) for val, null, deleted in args])
    return val, And([Or([null, deleted]) for _, null, deleted in args])


def Sum_Distinct(args):
    val = Sum([
        If(
            And([
                Not(deleted),
                Not(null),
                And([
                    Implies(
                        And([Not(prev_deleted), Not(prev_null)]),
                        val != prev_val
                    )
                    for prev_val, prev_null, prev_deleted in args[:idx]
                ])
            ]),
            val,
            Int(0))
        for idx, (val, null, deleted) in enumerate(args)
    ])
    return val, And([Or([null, deleted]) for _, null, deleted in args])


def Avg(args):
    sum_val, _ = AggSum(args)
    count_val, _ = Count(args)
    return sum_val / count_val, And([Or([null, deleted]) for _, null, deleted in args])


def Avg_Distinct(args):
    sum_distinct_val, _ = Sum_Distinct(args)
    count_distinct_val, _ = Count_Distinct(args)
    return sum_distinct_val / count_distinct_val, And([Or([null, deleted]) for _, null, deleted in args])


def Abs(x):
    return If(x >= Int(0), x, -x)


class ExpressionEncoder:
    def __init__(self, schema, env, outer_tuple_id=None, projected_list=None):
        self.table = schema
        self.tuple_idx = None
        self.env = env
        self.outer_tuple_id = outer_tuple_id

        if projected_list is not None:
            self.projected_list = projected_list
        self.subquery_table_map = {}

    def expression_for_tuple(self, exp, idx: int):
        self.tuple_idx = idx

        val, null = exp.accept(self)
        val = EnsureInt(val)
        return val, null

    def visit_Variable(self, var):
        return var.VAL, var.NULL

    def visit_CaseWhen(self, node):
        def CaseWhen(case_idx):
            if case_idx >= len(node.cases):
                if node.default is not None:
                    default_val, default_null = node.default.accept(self)
                    default_val = EnsureInt(default_val)
                    return default_val, default_null
                else:
                    return Int(0), Bool(True)
            else:
                cond_val, cond_null = node.cases[case_idx][0].accept(self)
                cond_val = EnsureBool(cond_val)
                result_val, result_null = node.cases[case_idx][1].accept(self)
                result_val = EnsureInt(result_val)

                next_val, next_null = CaseWhen(case_idx + 1)
                val = If(
                    And([Not(cond_null), cond_val]),
                    result_val,
                    next_val
                )
                null = If(
                    And([Not(cond_null), cond_val]),
                    result_null,
                    next_null
                )
                return val, null

        return CaseWhen(0)

    def visit_Expression(self, node):
        if isinstance(node.operator_callable, Operator):
            args = [arg.accept(self) for arg in node.args]
            vals = [val for val, _ in args]
            null = Or([null for _, null in args])
            if node.operator in ['gt', 'gte', 'lt', 'lte', 'eq', 'neq']:
                for i in range(len(vals)):
                    vals[i] = EnsureInt(vals[i])
                val = node.operator_callable()(*vals)
                return val, null
            elif node.operator in ['add', 'sub', 'mul', 'div', 'neg', 'not']:
                val = node.operator_callable()(*vals)
                # division by zero = NULL
                if node.operator == 'div':
                    null = Or([null, vals[-1] == Int(0)])
                return val, null
            elif node.operator in ['and', 'or']:
                for i in range(len(vals)):
                    vals[i] = EnsureBool(vals[i])
                val = node.operator_callable()([*vals])
                if node.operator == 'or':
                    null = And([null, And([Implies(Not(null), Not(val)) for val, (_, null) in zip(vals, args)])])
                if node.operator == 'and':
                    null = And([null, And([Implies(Not(null), val) for val, (_, null) in zip(vals, args)])])
                return val, null
            else:
                raise NotImplementedError
        elif node.operator == 'is_null' or node.operator == 'isnull':
            if isinstance(node.args[0], Query):
                from sql_testing.visitors.query_encoder import QueryEncoder

                if str(node.args[0]) in self.subquery_table_map:
                    in_table = self.subquery_table_map[str(node.args[0])]
                else:
                    visitor = QueryEncoder(self.env)
                    in_table = node.args[0].accept(visitor)
                    self.subquery_table_map[str(node.args[0])] = in_table
                return And([Deleted(in_table.table_id, tuple_id) for tuple_id in range(in_table.bound)]), Bool(False)
            else:
                _, null = node.args[0].accept(self)
                return null, Bool(False)
        elif node.operator == 'is_not_null':
            if isinstance(node.args[0], Query):
                from sql_testing.visitors.query_encoder import QueryEncoder

                if str(node.args[0]) in self.subquery_table_map:
                    in_table = self.subquery_table_map[str(node.args[0])]
                else:
                    visitor = QueryEncoder(self.env)
                    in_table = node.args[0].accept(visitor)
                    self.subquery_table_map[str(node.args[0])] = in_table
                return Or([Not(Deleted(in_table.table_id, tuple_id)) for tuple_id in range(in_table.bound)]), Bool(False)
            else:
                _, null = node.args[0].accept(self)
                return Not(null), Bool(False)
        elif node.operator == 'in':
            # rhs is a value set
            if isinstance(node.args[1], list):
                lhs_val, lhs_null = node.args[0].accept(self)
                exp_pairs = []
                for arg in node.args[1]:
                    exp_pairs.append(arg.accept(self))

                lhs_in_rhs = []
                for exp in exp_pairs:
                    lhs_in_rhs.append(And([
                        Not(lhs_null), Not(exp[1]),
                        lhs_val == exp[0]
                    ]))
                val = Or(lhs_in_rhs)

                null = Or([
                    lhs_null,
                    And([
                        Not(lhs_null),
                        Not(val),
                        Or([rhs_null for _, rhs_null in exp_pairs])
                    ])
                ])
                return val, null

            # rhs is a subquery
            from sql_testing.visitors.query_encoder import QueryEncoder

            if str(node.args[1]) in self.subquery_table_map:
                in_table = self.subquery_table_map[str(node.args[1])]
            else:
                visitor = QueryEncoder(self.env)
                in_table = node.args[1].accept(visitor)
                self.subquery_table_map[str(node.args[1])] = in_table

            # lhs is a list of [(exp1_val, exp1_null), (exp2_val, exp2_null), ...]
            lhs = []
            if isinstance(node.args[0], Attribute | Expression):
                assert len(in_table.columns) == 1
                lhs.append(node.args[0].accept(self))
            else:
                assert len(in_table.columns) == len(node.args[0])
                for arg in node.args[0]:
                    lhs.append(arg.accept(self))

            rhs = []
            for tuple_id in range(in_table.bound):
                rhs.append([
                    (self.env.cell(in_table.table_id, tuple_id, column_idx),
                     self.env.null(in_table.table_id, tuple_id, column_idx))
                    for column_idx in range(len(lhs))
                ])

            tuple_in_rhs = []
            for tuple_id, rhs_row in enumerate(rhs):
                row_in = [Not(Deleted(in_table.table_id, tuple_id))]
                for (lhs_val, lhs_null), (rhs_val, rhs_null) in zip(lhs, rhs_row):
                    row_in.append(And([
                        And([Not(lhs_null), Not(rhs_null)]),
                        lhs_val == rhs_val
                    ]))
                tuple_in_rhs.append(And(row_in))

            val = Or(tuple_in_rhs)

            return val, Bool(False)
        elif node.operator == 'nin':
            # rhs is a value set
            if isinstance(node.args[1], list):
                lhs_val, lhs_null = node.args[0].accept(self)
                exp_pairs = []
                for arg in node.args[1]:
                    exp_pairs.append(arg.accept(self))

                lhs_in_rhs = []
                for exp in exp_pairs:
                    lhs_in_rhs.append(And([
                        Not(lhs_null), Not(exp[1]),
                        lhs_val == exp[0]
                    ]))
                val = Not(Or(lhs_in_rhs))

                null = Or([
                    lhs_null,
                    And([
                        Not(lhs_null),
                        Not(val),
                        Or([rhs_null for _, rhs_null in exp_pairs])
                    ])
                ])
                return val, null

            # rhs is a subquery
            from sql_testing.visitors.query_encoder import QueryEncoder

            if str(node.args[1]) in self.subquery_table_map:
                in_table = self.subquery_table_map[str(node.args[1])]
            else:
                visitor = QueryEncoder(self.env)
                in_table = node.args[1].accept(visitor)
                self.subquery_table_map[str(node.args[1])] = in_table

            lhs = []
            if isinstance(node.args[0], Attribute | Expression):
                assert len(in_table.columns) == 1
                lhs.append(node.args[0].accept(self))
            else:
                assert len(in_table.columns) == len(node.args[0])
                for arg in node.args[0]:
                    lhs.append(arg.accept(self))

            rhs = []
            for tuple_id in range(in_table.bound):
                rhs.append([
                    (self.env.cell(in_table.table_id, tuple_id, column_idx),
                     self.env.null(in_table.table_id, tuple_id, column_idx))
                    for column_idx in range(len(lhs))
                ])

            tuple_not_in_rhs = []
            for tuple_idx, rhs_row in enumerate(rhs):
                row_not_in = []
                for (lhs_val, lhs_null), (rhs_val, rhs_null) in zip(lhs, rhs_row):
                    row_not_in.append(
                        And([
                            Not(Or([lhs_null, rhs_null])),
                            lhs_val != rhs_val
                        ])
                    )
                tuple_not_in_rhs.append(
                    Implies(
                        Not(Deleted(in_table.table_id, tuple_idx)),
                        Or(row_not_in)
                    )
                )

            val = Or([
                And([Deleted(in_table.table_id, tuple_idx) for tuple_idx in range(in_table.bound)]),
                And(tuple_not_in_rhs)
            ])

            return val, Bool(False)
        elif node.operator == 'between':
            value_val, value_null = node.args[0].accept(self)
            lower_bound_val, lower_bound_null = node.args[1].accept(self)
            upper_bound_val, upper_bound_null = node.args[2].accept(self)

            val = And([value_val >= lower_bound_val, value_val <= upper_bound_val])
            null = Or([
                value_null,
                And([
                    Not(value_null), lower_bound_null, Not(upper_bound_null),
                    Not(value_val > upper_bound_val)
                ]),
                And([
                    Not(value_null), Not(lower_bound_null), upper_bound_null,
                    Not(value_val < lower_bound_val)
                ]),
                And([lower_bound_null, upper_bound_null])
            ])
            return val, null
        elif node.operator == 'not_between':
            value_val, value_null = node.args[0].accept(self)
            lower_bound_val, lower_bound_null = node.args[1].accept(self)
            upper_bound_val, upper_bound_null = node.args[2].accept(self)

            val = Or([value_val < lower_bound_val, value_val > upper_bound_val])
            null = Or([
                value_null,
                And([
                    Not(value_null), lower_bound_null, Not(upper_bound_null),
                    Not(value_val > upper_bound_val)
                ]),
                And([
                    Not(value_null), Not(lower_bound_null), upper_bound_null,
                    Not(value_val < lower_bound_val)
                ]),
                And([lower_bound_null, upper_bound_null])
            ])
            return val, null
        elif node.operator in ['like', 'not_like']:
            cell = node.args[0].accept(self)
            return self.env.underapproximator.encode(node.operator, cell, node.args[1].value)
        else:
            # function

            if node.operator == 'abs':
                val, null = node.args[0].accept(self)
                return Abs(val), null

            if node.operator == 'if':
                condition_val, condition_null = node.args[0].accept(self)
                condition_val = EnsureBool(condition_val)
                then_val, then_null = node.args[1].accept(self)
                else_val, else_null = node.args[2].accept(self)
                val = If(
                    And([Not(condition_null), condition_val]),
                    then_val,
                    else_val
                )
                null = If(
                    And([Not(condition_null), condition_val]),
                    then_null,
                    else_null
                )
                return val, null

            if node.operator == 'ifnull':
                if_val, if_null = node.args[0].accept(self)
                default_val, default_null = node.args[1].accept(self)

                return If(Not(if_null), if_val, default_val), And([if_null, default_null])

            if node.operator == 'round':
                return node.args[0].accept(self)

            # aggregate functions
            if node.operator in ['min', 'max', 'count', 'sum', 'avg']:
                if node.operator == 'count' and not node.args[0] and isinstance(node.args[1], Attribute) and node.args[1].name == '*':
                    return Sum([Not(Deleted(self.table.table_id, tuple_id)) for tuple_id in range(self.table.bound)]), Bool(False)

                agg_exp_encoder = ExpressionEncoder(self.table, self.env)
                to_be_aggregated = []
                for tuple_id in range(self.table.bound):
                    # ret is a (val, null, deleted) pair
                    ret = (
                        *agg_exp_encoder.expression_for_tuple(node.args[1], tuple_id),
                        Deleted(self.table.table_id, tuple_id)
                    )
                    to_be_aggregated.append(ret)

                match node.operator:
                    case 'max':
                        return Max(to_be_aggregated)
                    case 'min':
                        return Min(to_be_aggregated)
                    case 'count':
                        if node.args[0]:
                            return Count_Distinct(to_be_aggregated)
                        return Count(to_be_aggregated)
                    case 'sum':
                        if node.args[0]:
                            return Sum_Distinct(to_be_aggregated)
                        return AggSum(to_be_aggregated)
                    case 'avg':
                        if node.args[0]:
                            return Avg_Distinct(to_be_aggregated)
                        return Avg(to_be_aggregated)
                    case _:
                        raise NotImplementedError

            if node.operator == 'coalesce':
                def Next(idx):
                    val, null = node.args[idx].accept(self)
                    if idx >= len(node.args) - 1:
                        return val, null
                    next_val, next_null = Next(idx + 1)
                    return If(Not(null), val, next_val), And([null, next_null])

                return Next(0)

            if node.operator == 'timestamp':
                assert len(node.args) == 2
                date_val, date_null = node.args[0].accept(self)
                time_val, time_null = node.args[1].accept(self)
                return time_val * Int(100000) + date_val, Or([date_null, time_null])

            if node.operator == 'date_add':
                date_val, date_null = node.args[0].accept(self)
                days_to_add, _ = node.args[1].accept(self)
                return date_val + days_to_add, date_null

            if node.operator == 'adddate':
                if isinstance(node.args[1], Literal):
                    date_val, date_null = node.args[0].accept(self)
                    return date_val + Int(node.args[1].value), date_null
                else:
                    date_val, date_null = node.args[0].accept(self)
                    days_to_add, _ = node.args[1].accept(self)
                    return date_val + days_to_add, date_null

            if node.operator == 'subdate':
                if isinstance(node.args[1], Literal):
                    date_val, date_null = node.args[0].accept(self)
                    return date_val - Int(node.args[1].value), date_null
                else:
                    date_val, date_null = node.args[0].accept(self)
                    days_to_sub, _ = node.args[1].accept(self)
                    return date_val - days_to_sub, date_null

            if node.operator == 'date_sub':
                date_val, date_null = node.args[0].accept(self)
                days_to_sub, _ = node.args[1].accept(self)
                return date_val - days_to_sub, date_null

            if node.operator == 'datediff':
                date1_val, date1_null = node.args[0].accept(self)
                date2_val, date2_null = node.args[1].accept(self)
                return date1_val - date2_val, Or([date1_null, date2_null])

            if node.operator == 'timestampdiff':
                if node.args[0].name.upper() == 'DAY':
                    date1_val, date1_null = node.args[1].accept(self)
                    date2_val, date2_null = node.args[2].accept(self)
                    return date2_val - date1_val, Or([date1_null, date2_null])
                else:
                    raise NotImplementedError

            if node.operator == 'str_to_date':
                return node.args[0].accept(self)

            if node.operator == 'interval':
                if node.args[1].name == 'day':
                    return Int(node.args[0].value), Bool(False)
                else:
                    raise NotImplementedError

            if node.operator == 'power':
                if not isinstance(node.args[0], Literal) or not isinstance(node.args[1], Literal):
                    raise NotImplementedError
                return node.args[0].value ** node.args[1].value, Bool(False)

            if node.operator == 'cast':
                return node.args[0].accept(self)
                # if isinstance(node.args[0].value, datetime.date) and node.args[1].value == 'date':
                #     return node.args[0].accept(self)
                # else:
                #     raise NotImplementedError(repr(node))

            if node.operator == 'any_value':
                return node.args[0].accept(self)

            # skipping operations in the outermost projection
            # for TPC benchmarks
            if node.operator == 'extract':
                return node.args[1].accept(self)
            if node.operator in ['concat', 'trim', 'ltrim', 'rtrim']:
                return node.args[0].accept(self)
            # print(self.table)
            raise NotImplementedError(repr(node))

    def visit_Attribute(self, node):
        try:
            return self.env.db[self.table.table_id, self.tuple_idx, self.table[node.name].column_id].accept(self)
        except SyntaxError:
            # expression used an alias in the select clause
            if self.projected_list is not None:
                for target in self.projected_list:
                    if target.alias == node.name:
                        return target.accept(self)

            # correlated subquery - try to locate the attribute from the main query table
            table = self.env.db.find_table_by_name(node.name.split('.')[0], self.env)
            if self.outer_tuple_id is None:
                raise AssertionError('outer tuple id is none')
            return self.env.db[table.table_id, self.outer_tuple_id, table[node.name].column_id].accept(self)

    def visit_Query(self, node):
        from sql_testing.visitors.query_encoder import QueryEncoder

        visitor = QueryEncoder(self.env, outer_tuple_id=self.tuple_idx)
        sub_table = node.accept(visitor)
        return self.env.cell(sub_table.table_id, 0, 0), self.env.null(sub_table.table_id, 0, 0)

    def visit_Literal(self, node):
        if node.value is None:
            return Int(0), Bool(True)
        elif isinstance(node.value, bool):
            return Bool(node.value), Bool(False)
        elif isinstance(node.value, int | float):
            return Int(node.value), Bool(False)
        elif isinstance(node.value, str):
            return Int(self.env.string_hash(node.value)), Bool(False)
        elif isinstance(node.value, datetime.date):
            first = datetime.date(1000, 1, 1)
            diff = node.value - first
            return Int(diff.days), Bool(False)
        elif isinstance(node.value, time.struct_time):
            zero = time.strptime('0:0:0', '%H:%M:%S')
            diff = int(time.mktime(node.value) - time.mktime(zero))
            return Int(diff), Bool(False)
        else:
            raise NotImplementedError
