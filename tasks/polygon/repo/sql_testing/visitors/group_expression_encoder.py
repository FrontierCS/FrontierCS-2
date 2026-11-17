import datetime
import functools

from sql_testing.ast.expressions.attribute import Attribute
from sql_testing.ast.expressions.literal import Literal
from sql_testing.smt.ast import *

from sql_testing.ast.expressions.operator import Operator
from sql_testing.variables import Variable
from sql_testing.visitors.expression_encoder import ExpressionEncoder


def GroupMax(args):
    def Max2(x, y):
        x_val, x_null, x_in_group = x
        y_val, y_null, y_in_group = y

        # val = If(
        #     Or([
        #         And([x_val <= y_val, Not(Or([x_null, y_null]))]),
        #         And([x_null, Not(y_null)]),
        #         And([Not(x_in_group), y_in_group]),
        #     ]),
        #     y_val,
        #     x_val
        # )
        val = If(
            And([x_in_group, Not(y_in_group)]),
            x_val,
            If(
                And([y_in_group, Not(x_in_group)]),
                y_val,
                If(
                    And([x_in_group, y_in_group]),
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
        null = And([Implies(x_in_group, x_null), Implies(y_in_group, y_null)])
        # null = And([x_null, y_null])
        in_group = Or([x_in_group, y_in_group])

        return val, null, in_group

    val, null, in_group = functools.reduce(Max2, args)
    return val, null


def GroupMin(args):
    def Min2(x, y):
        x_val, x_null, x_in_group = x
        y_val, y_null, y_in_group = y

        val = If(
            And([x_in_group, Not(y_in_group)]),
            x_val,
            If(
                And([y_in_group, Not(x_in_group)]),
                y_val,
                If(
                    And([x_in_group, y_in_group]),
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
        null = And([Implies(x_in_group, x_null), Implies(y_in_group, y_null)])
        # null = And([x_null, y_null])
        in_group = Or([x_in_group, y_in_group])

        return val, null, in_group

    val, null, _ = functools.reduce(Min2, args)
    return val, null


def GroupCount(args):
    val = Sum([If(And([Not(null), in_group]), Int(1), Int(0)) for _, null, in_group in args])
    return val, Bool(False)


def GroupCount_Distinct(args):
    val = Sum([
        If(
            And([
                Not(null),
                in_group,
                And([
                    Implies(
                        And([prev_in_group, Not(prev_null)]),
                        val != prev_val
                    )
                    for prev_val, prev_null, prev_in_group in args[:idx]
                ])
            ]),
            Int(1),
            Int(0)
        )
        for idx, (val, null, in_group) in enumerate(args)
    ])
    return val, Bool(False)


def GroupSum(args):
    val = Sum([If(And([Not(null), in_group]), val, Int(0)) for val, null, in_group in args])
    return val, And([Implies(in_group, null) for _, null, in_group in args])


def GroupSum_Distinct(args):
    val = Sum([
        If(
            And([
                Not(null),
                in_group,
                And([
                    Implies(
                        And([prev_in_group, Not(prev_null)]),
                        val != prev_val
                    )
                    for prev_val, prev_null, prev_in_group in args[:idx]
                ])
            ]),
            val,
            Int(0)
        )
        for idx, (val, null, in_group) in enumerate(args)
    ])
    return val, And([Implies(in_group, null) for _, null, in_group in args])


def GroupAvg(args):
    sum_val, _ = GroupSum(args)
    count_val, _ = GroupCount(args)
    val = sum_val / count_val
    return val, And([Implies(in_group, null) for _, null, in_group in args])


def GroupAvg_Distinct(args):
    sum_distinct_val, _ = GroupSum_Distinct(args)
    count_distinct_val, _ = GroupCount_Distinct(args)
    val = sum_distinct_val / count_distinct_val
    return val, And([Implies(in_group, null) for _, null, in_group in args])


class GroupExpressionEncoder:
    def __init__(self, group_by_table, input_table, env, projected_list=None):
        self.groupby_table = group_by_table
        self.input_table = input_table
        self.group_id = None
        self.output_cell_null = False
        self.env = env
        if projected_list is not None:
            self.projected_list = projected_list
        self.subquery_table_map = {}

    def expression_for_group(self, exp, group_id: int, convert_to_bool=False):
        self.group_id = group_id

        val, null = exp.accept(self)
        # the expression evaluates to a bool
        if val.return_type() == 'Bool' and not convert_to_bool:
            val = If(val, Int(1), Int(0))
        if val.return_type() == 'Int' and convert_to_bool:
            val = (val != Int(0))
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
                return NotImplementedError
        else:
            # function

            if node.operator == 'round':
                return node.args[0].accept(self)

            if node.operator in ['min', 'max', 'count', 'sum', 'avg']:
                if node.operator == 'count' and not node.args[0] and isinstance(node.args[1], Attribute) and node.args[1].name == '*':
                    in_groups = [SMTGrouping(self.groupby_table.table_id, tuple_id, self.group_id)
                                 for tuple_id in range(self.input_table.bound)]
                    return Sum([If(in_group, Int(1), Int(0)) for in_group in in_groups]), Bool(False)

                agg_exp_encoder = ExpressionEncoder(self.input_table, self.env)

                # find
                to_be_aggregated = []
                for tuple_id in range(self.input_table.bound):
                    val, null = agg_exp_encoder.expression_for_tuple(node.args[1], tuple_id)
                    if val.return_type() == 'Bool':
                        val = If(val, Int(1), Int(0))
                    # ret is a (val, null, in_group) pair
                    ret = (
                        val,
                        null,
                        SMTGrouping(self.groupby_table.table_id, tuple_id, self.group_id)
                    )
                    to_be_aggregated.append(ret)

                match node.operator:
                    case 'max':
                        return GroupMax(to_be_aggregated)
                    case 'min':
                        return GroupMin(to_be_aggregated)
                    case 'count':
                        if node.args[0]:
                            return GroupCount_Distinct(to_be_aggregated)
                        return GroupCount(to_be_aggregated)
                    case 'sum':
                        if node.args[0]:
                            return GroupSum_Distinct(to_be_aggregated)
                        return GroupSum(to_be_aggregated)
                    case 'avg':
                        if node.args[0]:
                            return GroupAvg_Distinct(to_be_aggregated)
                        return GroupAvg(to_be_aggregated)
                    case _:
                        raise NotImplementedError
            elif node.operator == 'if':
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
            elif node.operator == 'ifnull':
                if_val, if_null = node.args[0].accept(self)
                default_val, default_null = node.args[1].accept(self)

                return If(Not(if_null), if_val, default_val), default_null
            elif node.operator == 'is_null':
                _, null = node.args[0].accept(self)

                return null, Bool(False)
            elif node.operator == 'is_not_null':
                _, null = node.args[0].accept(self)
                return Not(null), Bool(False)
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
            elif node.operator == 'coalesce':
                def Next(idx):
                    val, null = node.args[idx].accept(self)
                    if idx >= len(node.args) - 1:
                        return val, null
                    next_val, next_null = Next(idx + 1)
                    return If(Not(null), val, next_val), And([null, next_null])
                return Next(0)
            elif node.operator == 'timestamp':
                assert len(node.args) == 2
                date_val, date_null = node.args[0].accept(self)
                time_val, time_null = node.args[1].accept(self)
                return time_val * Int(100000) + date_val, Or([date_null, time_null])
            elif node.operator == 'date_add':
                date_val, date_null = node.args[0].accept(self)
                days_to_add, _ = node.args[1].accept(self)
                return date_val + days_to_add, date_null
            elif node.operator == 'adddate':
                if isinstance(node.args[1], Literal):
                    date_val, date_null = node.args[0].accept(self)
                    return date_val + Int(node.args[1].value), date_null
                else:
                    date_val, date_null = node.args[0].accept(self)
                    days_to_add, _ = node.args[1].accept(self)
                    return date_val + days_to_add, date_null
            elif node.operator == 'subdate':
                if isinstance(node.args[1], Literal):
                    date_val, date_null = node.args[0].accept(self)
                    return date_val - Int(node.args[1].value), date_null
                else:
                    date_val, date_null = node.args[0].accept(self)
                    days_to_sub, _ = node.args[1].accept(self)
                    return date_val - days_to_sub, date_null
            elif node.operator == 'date_sub':
                date_val, date_null = node.args[0].accept(self)
                days_to_sub, _ = node.args[1].accept(self)
                return date_val - days_to_sub, date_null
            elif node.operator == 'datediff':
                date1_val, date1_null = node.args[0].accept(self)
                date2_val, date2_null = node.args[1].accept(self)
                return date1_val - date2_val, Or([date1_null, date2_null])
            elif node.operator == 'timestampdiff':
                if node.args[0].name.upper() == 'DAY':
                    date1_val, date1_null = node.args[1].accept(self)
                    date2_val, date2_null = node.args[2].accept(self)
                    return date2_val - date1_val, Or([date1_null, date2_null])
                else:
                    raise NotImplementedError
            elif node.operator == 'str_to_date':
                return node.args[0].accept(self)
            elif node.operator == 'interval':
                if node.args[1].name == 'day':
                    return Int(node.args[0].value), Bool(False)
                else:
                    raise NotImplementedError
            elif node.operator == 'cast':
                return node.args[0].accept(self)
                # if node.args[1].value == 'integer':
                #     return node.args[0].accept(self)
                # elif isinstance(node.args[0].value, datetime.date) and node.args[1].value == 'date':
                #     return node.args[0].accept(self)
                # else:
                #     raise NotImplementedError
            else:
                raise NotImplementedError(repr(node))

    def visit_Attribute(self, node):
        to_be_folded = []
        for input_tuple_id in range(self.input_table.bound):
            try:
                self.input_table[node.name]
            except SyntaxError:
                # expression in having clause used an alias in the select clause
                if self.projected_list is not None:
                    for target in self.projected_list:
                        if target.alias == node.name:
                            return target.accept(self)

            to_be_folded.append((
                SMTGrouping(self.groupby_table.table_id, input_tuple_id, self.group_id),
                *self.env.db[self.input_table.table_id, input_tuple_id, self.input_table[node.name].column_id].accept(self)
            ))

        def _foldl(x, y):
            x_grouping, x_val, x_null = x
            y_grouping, y_val, y_null = y
            val = If(
                And([
                    y_grouping,
                    Not(x_grouping)
                ]),
                y_val,
                x_val
            )
            null = If(
                And([
                    y_grouping,
                    Not(x_grouping)
                ]),
                y_null,
                x_null
            )
            grouping = Or([x_grouping, y_grouping])
            return grouping, val, null

        _, val, null = functools.reduce(_foldl, to_be_folded)
        return val, null

    def visit_Query(self, node):
        from sql_testing.visitors.query_encoder import QueryEncoder

        if str(node) in self.subquery_table_map:
            sub_table = self.subquery_table_map[str(node)]
        else:
            visitor = QueryEncoder(self.env)
            sub_table = node.accept(visitor)
            self.subquery_table_map[str(node)] = sub_table

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
        else:
            raise NotImplementedError
