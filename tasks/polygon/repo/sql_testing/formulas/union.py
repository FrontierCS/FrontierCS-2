from copy import deepcopy
from itertools import product
from typing import List

from sql_testing.formulas.distinct import FDistinct
from sql_testing.schemas import TableSchema
from sql_testing.smt.ast import *


class FUnion:
    def __init__(self,
                 input_tables: List[TableSchema],
                 node,
                 env,
                 k=4
                 ):
        self.inputs = input_tables
        self.node = node
        self.allow_duplicates = node.allow_duplicates
        self.env = env
        self.k = k

        self.output = self.create_output_table(self.inputs)
        self.output.node = self.node
        self.semantics(self.inputs, self.output)

        if not self.allow_duplicates:
            f = FDistinct(self.output, self.node, self.env)
            self.output = f.output

    def create_output_table(self, input_tables: [TableSchema]) -> TableSchema:
        output_table_id = self.env.next_table_id()
        new_table = TableSchema(
            output_table_id,
            f"!{'_UNION_'.join([table.table_name for table in input_tables])}!",
            sum([table.bound for table in input_tables])
        )
        for column in input_tables[0]:
            output_column = deepcopy(column)
            new_table.append(output_column)
        new_table.lineage = f"Union of T{', '.join([str(table.table_id) for table in input_tables])}"
        new_table.ancestors.append(input_tables)
        self.env.db.add_table(new_table)
        return new_table

    def semantics(self, input_tables: [TableSchema], output_table: TableSchema):
        output_table_id = output_table.table_id

        cases = []
        choice_constraints = []
        output_row_id = 0
        for table in input_tables:
            for tuple_idx in range(table.bound):
                choice_constraints.append(Or([
                    Choice(output_table_id, output_row_id) == Int(1),
                    Choice(output_table_id, output_row_id) == Int(0)
                ]))

                choice_1_tuples_mapping = []
                for column in table:
                    input_cell = self.env.db[table.table_id, tuple_idx, column.column_id]
                    output_cell = self.env.db[output_table.table_id, output_row_id, column.column_id]
                    choice_1_tuples_mapping.append(self.env.copy_cell(input_cell, output_cell))

                cases.append(
                    Implies(
                        Choice(output_table_id, output_row_id) == Int(1),
                        And([
                            Not(Deleted(table.table_id, tuple_idx)),

                            And(choice_1_tuples_mapping),
                            Not(Deleted(output_table_id, output_row_id)),
                        ])
                    )
                )
                cases.append(
                    Implies(
                        Choice(output_table_id, output_row_id) == Int(0),
                        And([
                            Deleted(table.table_id, tuple_idx),

                            Deleted(output_table_id, output_row_id)
                        ])
                    )
                )

                output_row_id += 1

        f = And([*cases, *choice_constraints])

        self.env.formulas.append(f, label=self.node.label)
