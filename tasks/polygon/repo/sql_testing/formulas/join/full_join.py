from collections import defaultdict
from itertools import combinations, product

from sql_testing.formulas.join.base_join import FJoin
from sql_testing.schemas import TableSchema
from sql_testing.smt.ast import *
from sql_testing.visitors.predicate_encoder import JoinPredicateEncoder


class FFullJoin(FJoin):
    def semantics(self, left_table: TableSchema, right_table: TableSchema, output_table: TableSchema):
        left_size_variable = self.env.size(left_table.table_id)
        right_size_variable = self.env.size(right_table.table_id)

        output_size_variable = self.env.size(output_table.table_id)

        encoder = JoinPredicateEncoder(left_table, right_table, self.condition, self.env)
        cases = []
        approximation_constraints = []

        branches = 0
        for input_table_sizes in self._gen_size([left_table, right_table], output_table):
            left_size_constraint = (left_size_variable == Int(input_table_sizes[0]))
            right_size_constraint = (right_size_variable == Int(input_table_sizes[1]))

            join_bijectives = list(product(*[
                [left_idx for left_idx in range(input_table_sizes[0])],
                [right_idx for right_idx in range(input_table_sizes[1])]
            ]))

            for assignment in product([0, 1], repeat=len(join_bijectives)):
                branches += 1
                if branches > self.branches_considered:
                    break

                assignment = list(assignment)
                left_table_tuple_joins = defaultdict(int)
                right_table_tuple_joins = defaultdict(int)

                case = [left_size_constraint, right_size_constraint]
                then = []

                output_tuple_idx = 0

                should_continue = False
                for bijective_index, bijective_assignment in enumerate(assignment):
                    val, null = encoder.predicate_for_tuple_pair(*join_bijectives[bijective_index])
                    if self.condition is not None:
                        if bijective_assignment == 1:
                            left_table_tuple_joins[join_bijectives[bijective_index][0]] += 1
                            right_table_tuple_joins[join_bijectives[bijective_index][1]] += 1

                            # check if within approximate scope
                            joins_occurred = input_table_sizes[0] + input_table_sizes[1]
                            for joins in left_table_tuple_joins.values():
                                if joins > 1:
                                    joins_occurred += joins - 1
                                if joins_occurred > self.k:
                                    should_continue = True
                                    break
                            if should_continue:
                                break

                            joins_occurred = input_table_sizes[0] + input_table_sizes[1]
                            for joins in right_table_tuple_joins.values():
                                if joins > 1:
                                    joins_occurred += joins - 1
                                if joins_occurred > self.k:
                                    should_continue = True
                                    break
                            if should_continue:
                                break

                            case.append(And([val, Not(null)]))

                            for column in output_table:
                                column_id = column.column_id
                                input_table_id, input_column_id = self.mapping[column_id]
                                if input_table_id == left_table.table_id:
                                    input_tuple_idx = join_bijectives[bijective_index][0]
                                else:
                                    input_tuple_idx = join_bijectives[bijective_index][1]
                                input_cell = self.env.db[input_table_id, input_tuple_idx, input_column_id]

                                output_cell = self.env.db[output_table.table_id, output_tuple_idx, column_id]
                                then.append(self.env.copy_cell(input_cell, output_cell))
                            output_tuple_idx += 1
                        else:
                            case.append(Or([null, And([Not(null), Not(val)])]))

                if should_continue:
                    continue

                # all rows from left table
                for tuple_idx in range(input_table_sizes[0]):
                    if left_table_tuple_joins[tuple_idx] == 0:
                        for column in output_table:
                            column_id = column.column_id
                            input_table_id, input_column_id = self.mapping[column_id]
                            output_cell = self.env.db[output_table.table_id, output_tuple_idx, column_id]
                            if input_table_id == left_table.table_id:
                                input_cell = self.env.db[input_table_id, tuple_idx, input_column_id]

                                then.append(self.env.copy_cell(input_cell, output_cell))
                            else:
                                then.append(output_cell.NULL)

                        output_tuple_idx += 1

                # all rows from right table
                for tuple_idx in range(input_table_sizes[1]):
                    if left_table_tuple_joins[tuple_idx] == 0:
                        for column in output_table:
                            column_id = column.column_id
                            input_table_id, input_column_id = self.mapping[column_id]
                            output_cell = self.env.db[output_table.table_id, output_tuple_idx, column_id]
                            if input_table_id == right_table.table_id:
                                input_cell = self.env.db[input_table_id, tuple_idx, input_column_id]

                                then.append(self.env.copy_cell(input_cell, output_cell))
                            else:
                                then.append(output_cell.NULL)

                        output_tuple_idx += 1

                then.append(output_size_variable == Int(output_tuple_idx))
                approximation_constraints.append(And(case))

                case_then = Implies(And(case), And(then))
                cases.append(case_then)

        lower_bound = (output_size_variable >= Int(0))
        upper_bound = (output_size_variable <= Int(output_table.bound))

        f = And([lower_bound, upper_bound, *cases])

        self.env.formulas.append(And([f, Or(approximation_constraints)]), label=self.node.label)
