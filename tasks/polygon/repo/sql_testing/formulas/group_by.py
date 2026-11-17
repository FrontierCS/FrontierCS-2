from copy import deepcopy
from itertools import combinations, product

from sql_testing.ast.expressions.attribute import Attribute
from sql_testing.ast.expressions.expression import Expression
from sql_testing.ast.expressions.literal import Literal
from sql_testing.ast.scan import Scan
from sql_testing.schemas import TableSchema, ColumnSchema
from sql_testing.smt.ast import *
from sql_testing.visitors.expression_encoder import ExpressionEncoder
from sql_testing.visitors.group_expression_encoder import GroupExpressionEncoder


def group_partition(tuple_indices: [int], groups_num: int):
    def group_numbers(numbers, n):
        if n == 1:
            yield [numbers]
        elif n == len(numbers):
            yield [[num] for num in numbers]
        elif n > len(numbers):
            return
        else:
            for i in range(1, len(numbers)):
                for comb in combinations(numbers, i):
                    first_group = [list(comb)]
                    remaining_groups = group_numbers([x for x in numbers if x not in comb], n - 1)
                    for group in remaining_groups:
                        yield first_group + group

    def deduplicate_groups(list_of_list_of_lists):
        unique_set = set()

        deduplicated_list = []
        for inner_list in list_of_list_of_lists:
            inner_list_set = frozenset(tuple(sub_list) for sub_list in inner_list)
            if inner_list_set not in unique_set:
                deduplicated_list.append([list(sub_list) for sub_list in inner_list_set])
                unique_set.add(inner_list_set)

        return deduplicated_list

    for groups in deduplicate_groups(group_numbers(tuple_indices, groups_num)):
        not_grouped = []

        # get tuple pairs that shouldn't be grouped (i.e., GroupBy expressions shouldn't be equivalent)
        for group in groups:
            for group_member in group:
                for idx in tuple_indices:
                    if idx != group_member and idx not in group:
                        not_grouped.append([group_member, idx])

        yield groups, not_grouped


class FGroupBy:
    def __init__(self,
                 input_table: TableSchema,
                 node,
                 env,
                 k,
                 ):
        self.input = input_table
        self.node = node
        self.env = env
        self.mapping = {}  # maps new column id to the column id in the input

        self.encoder = ExpressionEncoder(self.input, env)
        self.output = self.create_output_table(self.input)
        self.output.node = self.node
        self.semantics(self.input, self.output)

    def create_output_table(self, input_table: TableSchema) -> TableSchema:
        input_table_id, input_table_name = input_table.get_info()
        output_table_id = self.env.next_table_id()
        new_table = TableSchema(output_table_id, input_table_name, input_table.bound)

        new_table.ancestors.append(input_table)
        new_table.lineage = f"Grouped from T{input_table_id}"
        self.env.db.add_table(new_table)
        return new_table

    def semantics(self, input_table: TableSchema, output_table: TableSchema):
        input_table_id = input_table.table_id
        output_table_id = output_table.table_id

        # input_size_variable = self.env.size(input_table_id)
        # output_size_variable = self.env.size(output_table_id)

        encoder = ExpressionEncoder(input_table, self.env, projected_list=self.node.ctx['select_list'])
        group_encoder = GroupExpressionEncoder(output_table, input_table, self.env, projected_list=self.node.ctx['select_list'])

        group_by_exp = {}
        for tuple_idx in range(input_table.bound):
            # encode each expression in group by
            expressions = []
            for exp in self.node.expressions:
                # group by clause syntactic sugar
                if isinstance(exp, Literal) and not isinstance(exp.value, bool):
                    exp = self.node.ctx['select_list'][exp.value - 1]

                expressions.append(encoder.expression_for_tuple(exp, tuple_idx))
            group_by_exp[tuple_idx] = expressions

        def group_by_exp_equal(t1, t2):
            f_equ = []
            VAL, NULL = 0, 1
            for t1_group_exp, t2_group_exp in zip(group_by_exp[t1], group_by_exp[t2]):
                f_equ.append(
                    Or([
                        And([
                            t1_group_exp[NULL],
                            t2_group_exp[NULL]
                        ]),
                        And([
                            Not(Or([t1_group_exp[NULL], t2_group_exp[NULL]])),
                            t1_group_exp[VAL] == t2_group_exp[VAL]
                        ])
                    ])
                )
            return And(f_equ)

        cases = []
        choice_constraints = []

        # temp
        # cases.append(Choice(13, 0) == Int(0))
        # cases.append(Choice(13, 1) == Int(0))
        # cases.append(Choice(13, 2) == Int(0))
        # cases.append(Choice(13, 3) == Int(0))
        # cases.append(Choice(13, 4) == Int(0))
        # cases.append(Choice(13, 5) == Int(0))
        # cases.append(Choice(-13, 0) == Int(0))
        # cases.append(Choice(-13, 1) == Int(0))
        # cases.append(Choice(-13, 2) == Int(0))

        # constrain grouping predicates
        for tuple_idx in range(input_table.bound):
            cases.append(
                Implies(
                    Deleted(input_table_id, tuple_idx),
                    And([
                        Not(self.env.grouping(output_table_id, tuple_idx, group_idx))
                        for group_idx in range(output_table.bound)
                    ])
                )
            )
            cases.append(
                Implies(
                    Not(Deleted(input_table_id, tuple_idx)),
                    Sum([
                        If(self.env.grouping(output_table_id, tuple_idx, group_idx), Int(1), Int(0))
                        for group_idx in range(output_table.bound)
                    ]) == Int(1)
                )
            )

        for group_idx in range(output_table.bound):
            choice_constraints.append(Or([
                Choice(output_table_id, group_idx) == Int(1),
                Choice(output_table_id, group_idx) == Int(0)
            ]))

            # choice 1 - the tuple has a unique unseen group expression so far, forming a new group to exist
            cases.append(
                Implies(
                    Choice(output_table_id, group_idx) == Int(1),
                    And([
                        # input tuple is not deleted
                        Not(Deleted(input_table_id, group_idx)),
                        # input tuple's group by expression is unique
                        Not(Or([
                            And([
                                Not(Deleted(-output_table_id, prev_group_idx)),
                                group_by_exp_equal(group_idx, prev_group_idx)
                            ])
                            for prev_group_idx in range(0, group_idx)
                        ])) if group_idx > 0 else Bool(True),

                        self.env.grouping(output_table_id, group_idx, group_idx),
                        Not(Deleted(-output_table_id, group_idx)),
                    ])
                )
            )

            # choice 0 - the tuple is deleted or has a duplicate group by expression
            cases.append(
                Implies(
                    Choice(output_table_id, group_idx) == Int(0),
                    And([
                        Or([
                            # choice 0 case 1: input tuple deleted
                            Deleted(input_table_id, group_idx),
                            # choice 0 case 2: input tuple's group by expression duplicate
                            And([
                                Not(Deleted(input_table_id, group_idx)),
                                Or([
                                    And([
                                        Not(Deleted(-output_table_id, prev_group_idx)),  # table id?
                                        group_by_exp_equal(group_idx, prev_group_idx),
                                        self.env.grouping(output_table_id, group_idx, prev_group_idx)
                                    ])
                                    for prev_group_idx in range(0, group_idx)
                                ]) if group_idx > 0 else Bool(False)
                            ]),
                        ]),

                        Deleted(-output_table_id, group_idx)
                    ])
                )
            )

        # then encode having predicate
        having_vector_start_index = output_table.bound
        for group_idx in range(output_table.bound):
            choice_constraints.append(Or([
                Choice(output_table_id, having_vector_start_index + group_idx) == Int(1),
                Choice(output_table_id, having_vector_start_index + group_idx) == Int(0)
            ]))

            if self.node.having is not None:
                having_val, having_null = group_encoder.expression_for_group(self.node.having, group_idx, convert_to_bool=True)
            else:
                having_val, having_null = Bool(True), Bool(False)

            cases.append(
                Implies(
                    Choice(output_table_id, having_vector_start_index + group_idx) == Int(1),
                    And([
                        Not(Deleted(-output_table_id, group_idx)),
                        And([Not(having_null), having_val]),

                        Not(Deleted(output_table_id, group_idx))
                    ])
                )
            )
            cases.append(
                Implies(
                    Choice(output_table_id, having_vector_start_index + group_idx) == Int(0),
                    And([
                        Or([
                            Deleted(-output_table_id, group_idx),
                            And([
                                Not(Deleted(-output_table_id, group_idx)),
                                Not(And([Not(having_null), having_val]))
                            ]),
                        ]),

                        Deleted(output_table_id, group_idx)
                    ])
                )
            )

        self.env.formulas.append(And([*cases, *choice_constraints]), label=self.node.label)
