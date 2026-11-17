from copy import deepcopy

from sql_testing.ast.expressions.attribute import Attribute
from sql_testing.ast.expressions.expression import Expression
from sql_testing.ast.expressions.literal import Literal
from sql_testing.schemas import TableSchema, ColumnSchema
from sql_testing.smt.ast import *
from sql_testing.utils import copy_group_by_mapping
from sql_testing.visitors.expression_encoder import ExpressionEncoder
from sql_testing.visitors.group_expression_encoder import GroupExpressionEncoder


class FProject:
    def __init__(self,
                 input_table: TableSchema,
                 node,
                 env,
                 k=2):
        self.input = deepcopy(input_table)
        self.node = node
        self.target_list = node.target_list
        self.env = env
        self.under_vector_size = k

        self.mapping = {}  # maps new column id to the column id in the input

        self.has_aggregate = False

        if 'Grouped from' in self.input.lineage:
            self.from_group_by = True
        else:
            self.from_group_by = False

        output = self.create_output_table(self.input)
        if self.from_group_by:
            group_by_label = output.node.label
            self.node.group_by_label = group_by_label
        output.node = self.node
        self.output = output

        if not self.has_aggregate and self.under_vector_size < output.bound:
            # under output
            self.approximated_output = deepcopy(output)
            self.approximated_output.table_id = self.env.next_table_id()
            self.approximated_output.bound = self.under_vector_size
            self.env.db.add_table(self.approximated_output)
            self.env.formulas.under_table_id_to_original[self.approximated_output.table_id] = output.table_id
            self.output = self.approximated_output
        else:
            self.approximated_output = None

        # for mutations happen in the projection but the upstream operator is a group by table,
        # copy the group by mapping
        # if 'Grouped' in self.gt_input.lineage and self.original_output.table_id != self.gt_input.table_id:
        #     copy_group_by_mapping(self.gt_input, self.original_output, self.env)

        if self.from_group_by:
            self.encoder = GroupExpressionEncoder(self.input, self.input.ancestors[0], env)

            self.semantics_group_by(self.input.ancestors[0], output)
        else:
            self.encoder = ExpressionEncoder(self.input, env)

            self.semantics(self.input, output)

    def create_output_table(self, input_table: TableSchema) -> TableSchema:
        # GROUP BY table
        if self.from_group_by:
            new_table = input_table
            # input_table = input_table.ancestors[0]
            # input_table_id, input_table_name = input_table.get_info()

            # output_table_id = self.env.next_table_id()
            # new_table = TableSchema(output_table_id, input_table.table_name, input_table.bound)
            # new_table.lineage = input_table.lineage
            # new_table.ancestors = input_table.ancestors
            # copy_group_by_mapping(input_table, new_table, self.env)

            input_table = input_table.ancestors[0]
            input_table_id, input_table_name = input_table.get_info()
        else:
            input_table_id, input_table_name = input_table.get_info()
            output_table_id = self.env.next_table_id()
            new_table = TableSchema(output_table_id, input_table_name, input_table.bound)

        prev_target_list = deepcopy(self.target_list)

        # remove * from the target list
        new_target_list = []
        for idx, target in enumerate(deepcopy(self.target_list)):
            if isinstance(target, Attribute):
                if target.name == '*':
                    for column in input_table:
                        new_target_list.append(Attribute(name=f"{column.table_name}.{column.column_name}"))
                elif '*' in target.name:
                    table_name = target.name.split('.')[0].lower()

                    for column in input_table:
                        # expand * in the target list as per the order in schema
                        if column.table_name.lower() == table_name.lower():
                            new_target_list.append(Attribute(name=f"{column.table_name}.{column.column_name}"))
                else:
                    new_target_list.append(target)
            else:
                new_target_list.append(target)

        self.target_list = new_target_list

        column_id = 0
        for idx, target in enumerate(self.target_list):
            if isinstance(target, Attribute):
                if '.' in target.name:
                    name = target.name.split('.')
                    table = name[0]
                    attr = name[1]
                else:
                    table = None
                    attr = target.name

                found = False
                for column in input_table:
                    if (((table is None or column.table_name is None) or (column.table_name.lower() == table.lower()))
                            and column.column_name.lower() == attr.lower()):
                        output_column = deepcopy(column)
                        output_column.column_id = column_id
                        if target.alias is not None:
                            output_column.column_name = target.alias
                            output_column.name_before_project = target.name
                        self.mapping[column_id] = column.column_id
                        column_id += 1
                        new_table.append(output_column)
                        found = True
                        break
                if not found:
                    raise SyntaxError(f"Attribute '{target.name}' does not exist in T{input_table_id} ({input_table_name})\n{input_table}")
            elif isinstance(target, Expression):
                column_name = target.alias if target.alias is not None else str(target)
                output_column = ColumnSchema(column_id=column_id, column_name=column_name, column_type='int')
                if target.alias is not None:
                    output_column.name_before_project = str(Expression(operator=target.operator, args=target.args))
                column_id += 1
                new_table.append(output_column)

                if any(f'{agg_op}(' in str(target).lower() for agg_op in ['min', 'max', 'count', 'sum', 'avg']):
                    self.has_aggregate = True
                # if target.operator in ['min', 'max', 'count', 'sum', 'avg']:
                #     self.has_aggregate = True
                # elif target.operator == 'ifnull' and target.args[0].operator in ['min', 'max', 'count', 'sum', 'avg']:
                #     self.has_aggregate = True
            elif isinstance(target, Literal):
                column_name = target.alias if target.alias is not None else str(target)
                output_column = ColumnSchema(column_id=column_id, column_name=column_name, column_type='int')
                column_id += 1
                new_table.append(output_column)
            else:
                raise TypeError(f"Unexpected target type: {type(target)}, target: {target}")

        new_table.ancestors.append(input_table)
        if new_table.lineage is not None and 'Grouped from' in new_table.lineage:
            new_table.lineage = f"Grouped and projected from T{input_table_id}"
        else:
            new_table.lineage = f"Projected from T{input_table_id}"
        self.env.db.add_table(new_table)

        # self.target_list = prev_target_list

        return new_table

    def semantics(self, input_table: TableSchema, output_table: TableSchema):
        input_table_id = input_table.table_id
        output_table_id = output_table.table_id

        # input_size_variable = self.env.size(input_table_id)
        # output_size_variable = self.env.size(output_table_id)

        cases = []
        choice_constraints = []

        # an aggregate function in projection -> output table has one row only
        if self.has_aggregate:
            output_table.bound = 1
            choice_constraints.append(Choice(output_table_id, 0) == Int(1))
            cases.append(Not(Deleted(output_table_id, 0)))

            mapping = []
            for column_id, target in enumerate(self.target_list):
                output_cell = self.env.db[output_table_id, 0, column_id]
                if isinstance(target, Attribute):
                    for input_tuple_id in range(input_table.bound):
                        mapping.append(
                            Implies(
                                And([
                                    Not(Deleted(input_table_id, input_tuple_id)),
                                    And([
                                        Deleted(input_table_id, prev_input_tuple_id)
                                        for prev_input_tuple_id in range(input_tuple_id)
                                    ])
                                ]),
                                self.env.copy_cell(
                                    self.env.db[input_table_id, input_tuple_id, self.mapping[column_id]],
                                    output_cell
                                )
                            )
                        )
                elif isinstance(target, Expression | Literal):
                    val, null = self.encoder.expression_for_tuple(target, 0)

                    mapping.append(output_cell.VAL == val)
                    mapping.append(output_cell.NULL == null)
                else:
                    raise TypeError(repr(target))
            cases.append(
                And(mapping)
            )
            f = And([*cases, *choice_constraints])
            self.env.formulas.append(f, label=self.node.label)
            return

        for tuple_id in range(output_table.bound):
            choice_constraints.append(Or([
                Choice(output_table_id, tuple_id) == Int(1),
                Choice(output_table_id, tuple_id) == Int(0)
            ]))

            choice_1_tuples_mapping = []
            for column_id, target in enumerate(self.target_list):
                output_cell = self.env.db[output_table_id, tuple_id, column_id]
                if isinstance(target, Attribute):
                    input_cell = self.env.db[input_table_id, tuple_id, self.mapping[column_id]]

                    choice_1_tuples_mapping.append(output_cell.VAL == input_cell.VAL)
                    choice_1_tuples_mapping.append(output_cell.NULL == input_cell.NULL)
                elif isinstance(target, Expression | Literal):
                    val, null = self.encoder.expression_for_tuple(target, tuple_id)

                    choice_1_tuples_mapping.append(output_cell.VAL == val)
                    choice_1_tuples_mapping.append(output_cell.NULL == null)
                else:
                    raise TypeError(repr(target))

            cases.append(
                Implies(
                    Choice(output_table_id, tuple_id) == Int(1),
                    And([
                        Not(Deleted(input_table_id, tuple_id)),

                        And(choice_1_tuples_mapping),
                        Not(Deleted(output_table_id, tuple_id)),
                    ])
                )
            )
            cases.append(
                Implies(
                    Choice(output_table_id, tuple_id) == Int(0),
                    And([
                        Deleted(input_table_id, tuple_id),

                        Deleted(output_table_id, tuple_id)
                    ])
                )
            )

        # print('='*50)
        # print(cases[2])
        # print('=' * 50)

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

    def semantics_group_by(self, input_table: TableSchema, groupby_table: TableSchema):
        input_table_id = input_table.table_id

        groupby_table_id = groupby_table.table_id
        # groupby_table_size_variable = self.env.size(groupby_table_id)

        cases = []
        choice_constraints = []

        vector_start_index = groupby_table.bound
        for bit_id in range(groupby_table.bound * 2):
            choice_constraints.append(Or([
                Choice(groupby_table_id, bit_id) == Int(1),
                Choice(groupby_table_id, bit_id) == Int(0)
            ]))

        for group_id in range(groupby_table.bound):
            choice_1_tuples_mapping = []
            for column_id, target in enumerate(self.target_list):
                output_cell = self.env.db[groupby_table_id, group_id, column_id]
                if isinstance(target, Attribute):
                    for input_tuple_id in range(input_table.bound):
                        choice_1_tuples_mapping.append(
                            Implies(
                                And([
                                    self.env.grouping(groupby_table_id, input_tuple_id, group_id),
                                    And([
                                        Not(self.env.grouping(groupby_table_id, prev_input_tuple_id, group_id))
                                        for prev_input_tuple_id in range(input_tuple_id)
                                    ])
                                ]),
                                self.env.copy_cell(
                                    self.env.db[input_table_id, input_tuple_id, self.mapping[column_id]],
                                    output_cell
                                )
                            )
                        )
                elif isinstance(target, Expression | Literal):
                    input_cell = self.encoder.expression_for_group(target, group_id)
                    choice_1_tuples_mapping.append(output_cell.VAL == input_cell[0])
                    choice_1_tuples_mapping.append(output_cell.NULL == input_cell[1])
                else:
                    raise TypeError(repr(target))

            cases.append(
                Implies(
                    Choice(groupby_table_id, vector_start_index + group_id) == Int(1),
                    And([
                        # Not(Deleted(input_table_id, group_id)),

                        And(choice_1_tuples_mapping),
                        # Not(Deleted(groupby_table_id, group_id)),
                    ])
                )
            )

        if self.approximated_output is None:
            self.env.formulas.append(And([*cases, *choice_constraints]), label=self.node.label)
            return

        # mapping real output vector to under-approximated vector
        output_table_size = Sum([
            Choice(groupby_table_id, vector_start_index + group_id)
            for group_id in range(groupby_table.bound)
        ])

        self.env.formulas.append(output_table_size <= Int(self.approximated_output.bound), label=f'size_{groupby_table_id}')

        for mapped_to_tuple_id in range(self.approximated_output.bound):
            mapping = []
            for output_tuple_id in range(groupby_table.bound):
                is_nth_non_deleted_tuple = [
                    Choice(groupby_table_id, vector_start_index + output_tuple_id) == Int(1),
                    Sum([
                        Choice(groupby_table_id, vector_start_index + prev_output_tuple_id)
                        for prev_output_tuple_id in range(output_tuple_id)
                    ]) == Int(mapped_to_tuple_id)
                ]
                mapping.append(
                    Implies(
                        # this output_tuple_id is the nth non deleted tuple id
                        And(is_nth_non_deleted_tuple),
                        And([
                            self.env.copy_cell(
                                self.env.db[groupby_table_id, output_tuple_id, column_id],
                                self.env.db[self.approximated_output.table_id, mapped_to_tuple_id, column_id]
                            )
                            for column_id in range(len(groupby_table.columns))
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

        self.env.formulas.append(And([*cases, *choice_constraints]), label=self.node.label)
