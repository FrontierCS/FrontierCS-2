from copy import deepcopy

from sql_testing.ast.expressions.attribute import Attribute
from sql_testing.ast.expressions.expression import Expression
from sql_testing.ast.expressions.literal import Literal
from sql_testing.schemas import TableSchema
from sql_testing.smt.ast import *
from sql_testing.visitors.expression_encoder import ExpressionEncoder


class FOrderBy:
    def __init__(self,
                 input_table: TableSchema,
                 node,
                 env,
                 k=2):
        self.input = input_table
        self.node = node
        self.sorting_expressions = node.expressions
        self.sort_orders = node.sort_orders
        self.limit = node.limit
        self.env = env
        self.k = k

        self.output = self.create_output_table(self.input)
        self.output.node = self.node
        self.encoder = ExpressionEncoder(self.input, env)
        self.semantics(self.input, self.output)

    def create_output_table(self, input_table: TableSchema) -> TableSchema:
        input_table_id, input_table_name = input_table.get_info()
        output_table_id = self.env.next_table_id()
        # new_table = TableSchema(output_table_id, input_table_name, input_table.bound)
        new_table = TableSchema(output_table_id, input_table_name, self.k)
        for column in input_table:
            output_column = deepcopy(column)
            new_table.append(output_column)
        new_table.lineage = f"Sorted from T{input_table_id} ({self.sorting_expressions})"
        new_table.ancestors.append(input_table)
        self.env.db.add_table(new_table)
        return new_table

    def semantics(self, input_table: TableSchema, output_table: TableSchema):
        input_table_id = input_table.table_id
        output_table_id = output_table.table_id

        cases = []
        choice_constraints = []

        input_table_size = Sum([
            If(Deleted(input_table_id, tuple_id), Int(0), Int(1))
            for tuple_id in range(input_table.bound)
        ])

        self.env.formulas.append(input_table_size <= Int(self.k), label=f'size_{input_table_id}')

        for tuple_idx in range(input_table.bound):
            choice_constraints.append(And([
                Choice(output_table_id, tuple_idx) >= Int(0),
                Choice(output_table_id, tuple_idx) <= Int(self.k),  # <= input_table_size?
            ]))

            for ordering in range(self.k + 1):
                if ordering == 0:
                    cases.append(
                        Implies(
                            Choice(output_table_id, tuple_idx) == Int(ordering),
                            Deleted(input_table_id, tuple_idx)
                        )
                    )
                else:
                    num_others_before_this_tuple = []

                    for other_tuple_idx in range(input_table.bound):
                        if other_tuple_idx == tuple_idx:
                            continue

                        is_before = []
                        all_prev_expression_eq = []
                        for expression_idx, expression in enumerate(self.sorting_expressions):
                            if isinstance(expression, Literal):
                                # syntactic sugar
                                if isinstance(expression.value, bool):
                                    continue
                                this_tuple_cell = self.env.db[input_table.table_id, tuple_idx, expression.value - 1]
                                this_tuple_cell = this_tuple_cell.VAL, this_tuple_cell.NULL
                                other_tuple_cell = self.env.db[input_table.table_id, other_tuple_idx, expression.value - 1]
                                other_tuple_cell = other_tuple_cell.VAL, other_tuple_cell.NULL
                            else:
                                # attribute or expression
                                this_tuple_cell = self.encoder.expression_for_tuple(expression, tuple_idx)
                                other_tuple_cell = self.encoder.expression_for_tuple(expression, other_tuple_idx)
                            VAL, NULL = 0, 1
                            is_before.append(
                                And([
                                    And(deepcopy(all_prev_expression_eq)),
                                    And([
                                        Not(Deleted(input_table_id, other_tuple_idx)),
                                        Or([
                                            And([other_tuple_cell[NULL], Not(this_tuple_cell[NULL])]),
                                            And([
                                                And([Not(other_tuple_cell[NULL]), Not(this_tuple_cell[NULL])]),
                                                other_tuple_cell[VAL] <= this_tuple_cell[VAL]
                                                if self.sort_orders[expression_idx] == 'asc'
                                                else other_tuple_cell[VAL] >= this_tuple_cell[VAL]
                                            ])
                                        ])
                                    ])
                                ])
                            )
                            all_prev_expression_eq.append(
                                And([
                                    Not(Deleted(input_table_id, other_tuple_idx)),
                                    Or([
                                        And([other_tuple_cell[NULL], this_tuple_cell[NULL]]),
                                        And([
                                            And([Not(other_tuple_cell[NULL]), Not(this_tuple_cell[NULL])]),
                                            other_tuple_cell[VAL] == this_tuple_cell[VAL]
                                        ])
                                    ])
                                ])
                            )
                        num_others_before_this_tuple.append(Or(is_before))

                    mapping = []
                    for column in input_table:
                        input_cell = self.env.db[input_table.table_id, tuple_idx, column.column_id]
                        output_cell = self.env.db[output_table.table_id, ordering - 1, column.column_id]
                        mapping.append(self.env.copy_cell(input_cell, output_cell))

                    cases.append(
                        Implies(
                            Choice(output_table_id, tuple_idx) == Int(ordering),
                            And([
                                Not(Deleted(input_table_id, tuple_idx)),

                                Not(Deleted(output_table_id, ordering - 1)),
                                Sum([
                                    If(indicator, Int(1), Int(0))
                                    for indicator in num_others_before_this_tuple
                                ]) == Int(ordering - 1),
                                And(mapping)
                            ])
                        )
                    )

        for tuple_idx in range(output_table.bound):
            # choice_constraints.append(
            #     Sum([
            #         If(Choice(output_table_id, bit) == Int(tuple_idx + 1), Int(1), Int(0))
            #         for bit in range(input_table.bound)
            #     ]) <= Int(1)
            # )

            cases.append(
                Implies(
                    Sum([
                        If(Choice(output_table_id, bit) != Int(0), Int(1), Int(0))
                        for bit in range(input_table.bound)
                    ]) <= Int(tuple_idx),
                    Deleted(output_table_id, tuple_idx)
                )
            )

        if self.limit is not None:
            output_table.bound = self.limit

        f = And([*cases, *choice_constraints])

        self.env.formulas.append(f, label=self.node.label)
