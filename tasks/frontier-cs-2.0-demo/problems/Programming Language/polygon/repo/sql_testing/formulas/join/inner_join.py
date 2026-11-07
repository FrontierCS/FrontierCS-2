from itertools import product

from sql_testing.ast.expressions.literal import Literal
from sql_testing.formulas.join.base_join import FJoin
from sql_testing.schemas import TableSchema
from sql_testing.smt.ast import *
from sql_testing.visitors.predicate_encoder import JoinPredicateEncoder


class FInnerJoin(FJoin):
    def semantics(self, left_table: TableSchema, right_table: TableSchema, output_table: TableSchema):
        left_table_id = left_table.table_id
        right_table_id = right_table.table_id
        output_table_id = output_table.table_id

        encoder = JoinPredicateEncoder(left_table, right_table, self.condition, self.env)
        cases = []
        choice_constraints = []

        join_bijectives = list(product(*[
            [left_idx for left_idx in range(left_table.bound)],
            [right_idx for right_idx in range(right_table.bound)]
        ]))

        # semantics encoding
        for bit_index in range(len(join_bijectives)):
            val, null = encoder.predicate_for_tuple_pair(*join_bijectives[bit_index])
            if self.condition is not None:
                choice_constraints.append(Or([
                    Choice(output_table_id, bit_index) == Int(1),
                    Choice(output_table_id, bit_index) == Int(0)
                ]))

                choice_1_tuples_mapping = []
                for column in output_table:
                    column_id = column.column_id
                    input_table_id, input_column_id = self.mapping[column_id]
                    if input_table_id == left_table.table_id:
                        input_tuple_idx = join_bijectives[bit_index][0]
                    else:
                        input_tuple_idx = join_bijectives[bit_index][1]
                    input_cell = self.env.db[input_table_id, input_tuple_idx, input_column_id]

                    output_cell = self.env.db[output_table.table_id, bit_index, column_id]
                    choice_1_tuples_mapping.append(self.env.copy_cell(input_cell, output_cell))

                cases.append(
                    Implies(
                        Choice(output_table_id, bit_index) == Int(1),
                        And([
                            Not(Deleted(left_table_id, join_bijectives[bit_index][0])),
                            Not(Deleted(right_table_id, join_bijectives[bit_index][1])),
                            And([Not(null), val]),

                            And(choice_1_tuples_mapping),
                            Not(Deleted(output_table_id, bit_index)),
                        ])
                    )
                )
                cases.append(
                    Implies(
                        Choice(output_table_id, bit_index) == Int(0),
                        And([
                            Or([
                                Deleted(left_table_id, join_bijectives[bit_index][0]),
                                Deleted(right_table_id, join_bijectives[bit_index][1]),
                                And([
                                    Not(Or([
                                        Deleted(left_table_id, join_bijectives[bit_index][0]),
                                        Deleted(right_table_id, join_bijectives[bit_index][1])
                                    ])),
                                    Or([null, And([Not(null), Not(val)])])
                                ])
                            ]),

                            Deleted(output_table_id, bit_index)
                        ])
                    )
                )

        if self.approximated_output is None:
            f = And([*cases, *choice_constraints])

            self.env.formulas.append(f, label=self.node.label)
            return

        # mapping real output vector to under-approximated vector
        output_table_size = Sum([
            Choice(output_table_id, bit_id)
            for bit_id in range(len(join_bijectives))
        ])

        self.env.formulas.append(output_table_size <= Int(self.approximated_output.bound), label=f'size_{output_table_id}')

        for mapped_to_tuple_id in range(self.approximated_output.bound):
            mapping = []
            for output_tuple_id in range(output_table.bound):
                is_nth_non_deleted_tuple = [
                    Choice(output_table_id, output_tuple_id) == Int(1),
                    Sum([
                        Choice(output_table_id, prev_output_tuple_id)
                        for prev_output_tuple_id in range(output_tuple_id)
                    ]) == Int(mapped_to_tuple_id)
                ]
                mapping.append(
                    Implies(
                        # this output_tuple_id is the nth non deleted tuple id
                        And(is_nth_non_deleted_tuple),
                        And([
                            self.env.copy_cell(
                                self.env.db[output_table_id, output_tuple_id, column_id],
                                self.env.db[self.approximated_output.table_id, mapped_to_tuple_id, column_id]
                            )
                            for column_id in range(len(output_table.columns))
                        ])
                    )
                )

            cases.append(
                Implies(
                    output_table_size >= Int(mapped_to_tuple_id + 1),
                    And([
                        Not(Deleted(self.approximated_output.table_id, mapped_to_tuple_id)),
                        And(mapping)
                    ])
                )
            )
            cases.append(
                Implies(
                    Not(output_table_size >= Int(mapped_to_tuple_id + 1)),
                    Deleted(self.approximated_output.table_id, mapped_to_tuple_id)
                )
            )

        f = And([*cases, *choice_constraints])

        self.env.formulas.append(f, label=self.node.label)
