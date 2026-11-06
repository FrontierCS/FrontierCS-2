import operator
from copy import deepcopy
from itertools import product

from sql_testing.schemas import TableSchema
from sql_testing.smt.ast import *


class FDistinct:
    def __init__(self,
                 input_table: TableSchema,
                 node,
                 env):
        self.input = input_table
        self.node = node
        self.env = env

        self.output = self.create_output_table(self.input)
        self.output.node = self.node
        self.semantics(self.input, self.output)

    def create_output_table(self, input_table: TableSchema) -> TableSchema:
        input_table_id, input_table_name = input_table.get_info()
        output_table_id = self.env.next_table_id()
        new_table = TableSchema(output_table_id, input_table_name, input_table.bound)
        for column in input_table:
            output_column = deepcopy(column)
            new_table.append(output_column)
        new_table.lineage = f"Duplicate eliminated from T{input_table_id}"
        new_table.ancestors.append(input_table)
        self.env.db.add_table(new_table)
        return new_table

    def semantics(self, input_table: TableSchema, output_table: TableSchema):
        input_table_id = input_table.table_id
        output_table_id = output_table.table_id

        # input_size_variable = self.env.size(input_table_id)
        # output_size_variable = self.env.size(output_table_id)

        def tuple_equal(t1, t2):
            f_equ = []
            for col_idx in range(len(input_table.columns)):
                f_equ.append(
                    Or([
                        And([
                            self.env.null(input_table_id, t1, col_idx),
                            self.env.null(input_table_id, t2, col_idx)
                        ]),
                        And([
                            Not(Or([
                                self.env.null(input_table_id, t1, col_idx),
                                self.env.null(input_table_id, t2, col_idx)
                            ])),
                            self.env.cell(input_table_id, t1, col_idx) == self.env.cell(input_table_id, t2, col_idx)
                        ])
                    ])
                )
            return And(f_equ)

        cases = []
        choice_constraints = []
        for tuple_idx in range(output_table.bound):
            choice_constraints.append(Or([
                Choice(output_table_id, tuple_idx) == Int(1),
                Choice(output_table_id, tuple_idx) == Int(0)
            ]))

            choice_1_tuples_mapping = []
            for column in input_table:
                column_id = column.column_id
                input_cell = self.env.db[input_table_id, tuple_idx, column_id]
                output_cell = self.env.db[output_table_id, tuple_idx, column_id]
                choice_1_tuples_mapping.append(self.env.copy_cell(input_cell, output_cell))

            # choice 1 - the tuple is a unique tuple so far
            cases.append(
                Implies(
                    Choice(output_table_id, tuple_idx) == Int(1),
                    And([
                        # input tuple is not deleted
                        Not(Deleted(input_table_id, tuple_idx)),
                        # input tuple's group by expression is unique
                        Not(Or([
                            And([
                                Not(Deleted(output_table_id, prev_tuple_idx)),
                                tuple_equal(tuple_idx, prev_tuple_idx)
                            ])
                            for prev_tuple_idx in range(0, tuple_idx)
                        ])) if tuple_idx > 0 else Bool(True),

                        And(choice_1_tuples_mapping),
                        Not(Deleted(output_table_id, tuple_idx))
                    ])
                )
            )

            # choice 0 - the tuple is deleted or duplicate
            cases.append(
                Implies(
                    Choice(output_table_id, tuple_idx) == Int(0),
                    And([
                        Or([
                            # choice 0 case 1: input tuple deleted
                            Deleted(input_table_id, tuple_idx),
                            # choice 0 case 2: input tuple's group by expression duplicate
                            And([
                                Not(Deleted(input_table_id, tuple_idx)),
                                Or([
                                    And([
                                        Not(Deleted(output_table_id, prev_tuple_idx)),
                                        tuple_equal(tuple_idx, prev_tuple_idx),
                                    ])
                                    for prev_tuple_idx in range(0, tuple_idx)
                                ]) if tuple_idx > 0 else Bool(False)
                            ]),
                        ]),

                        Deleted(output_table_id, tuple_idx)
                    ])
                )
            )

        self.env.formulas.append(
            And([*cases, *choice_constraints]),
            label=self.node.distinct_label
        )
