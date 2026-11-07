from copy import deepcopy
from itertools import product

from sql_testing.ast.expressions.expression import Expression
from sql_testing.schemas import TableSchema
from sql_testing.smt.ast import *
from sql_testing.visitors.predicate_encoder import PredicateEncoder


class FFilter:
    def __init__(self,
                 input_table: TableSchema,
                 node,
                 env,
                 k=2,
                 outer_tuple_id=None
                 ):
        self.input = input_table
        self.node = node
        self.predicate = node.predicate
        self.env = env
        self.under_vector_size = k
        self.outer_tuple_id = outer_tuple_id

        # precise output
        output = self.create_output_table(self.input)
        output.node = self.node
        self.output = output

        if self.under_vector_size < output.bound:
            # under output
            self.approximated_output = deepcopy(output)
            self.approximated_output.table_id = self.env.next_table_id()
            self.approximated_output.bound = self.under_vector_size
            self.env.db.add_table(self.approximated_output)
            self.env.formulas.under_table_id_to_original[self.approximated_output.table_id] = output.table_id
            self.output = self.approximated_output
        else:
            self.approximated_output = None

        self.semantics(self.input, output)

    def create_output_table(self, input_table: TableSchema) -> TableSchema:
        input_table_id, input_table_name = input_table.get_info()
        output_table_id = self.env.next_table_id()
        new_table = TableSchema(output_table_id, input_table_name, input_table.bound)
        for column in input_table:
            output_column = deepcopy(column)
            new_table.append(output_column)
        new_table.lineage = f"Filtered from T{input_table_id} ({self.predicate})"
        new_table.ancestors.append(input_table)
        self.env.db.add_table(new_table)
        return new_table

    def semantics(self, input_table: TableSchema, output_table: TableSchema):
        input_table_id = input_table.table_id
        output_table_id = output_table.table_id

        encoder = PredicateEncoder(input_table, self.predicate, self.env, self.outer_tuple_id)
        cases = []
        choice_constraints = []
        for tuple_idx in range(output_table.bound):
            choice_constraints.append(Or([
                Choice(output_table_id, tuple_idx) == Int(1),
                Choice(output_table_id, tuple_idx) == Int(0)
            ]))

            val, null = encoder.predicate_for_tuple(tuple_idx)

            choice_1_tuples_mapping = []
            for column in input_table:
                column_id = column.column_id
                input_cell = self.env.db[input_table_id, tuple_idx, column_id]
                output_cell = self.env.db[output_table_id, tuple_idx, column_id]
                choice_1_tuples_mapping.append(self.env.copy_cell(input_cell, output_cell))

            cases.append(
                Implies(
                    Choice(output_table_id, tuple_idx) == Int(1),
                    And([
                        Not(Deleted(input_table_id, tuple_idx)),
                        And([Not(null), val]),

                        And(choice_1_tuples_mapping),
                        Not(Deleted(output_table_id, tuple_idx)),
                    ])
                )
            )
            cases.append(
                Implies(
                    Choice(output_table_id, tuple_idx) == Int(0),
                    And([
                        Or([
                            Deleted(input_table_id, tuple_idx),
                            And([Not(Deleted(input_table_id, tuple_idx)), Or([null, And([Not(null), Not(val)])])])
                        ]),

                        Deleted(output_table_id, tuple_idx)
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
            for bit_id in range(output_table.bound)
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
