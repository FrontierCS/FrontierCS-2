import operator
from copy import deepcopy

from sql_testing.ast.expressions.case_when import CaseWhen
from sql_testing.ast.expressions.expression import Expression
from sql_testing.ast.filter import Filter
from sql_testing.ast.group_by import GroupBy
from sql_testing.ast.join import Join
from sql_testing.ast.node import MutatedNode
from sql_testing.ast.project import Project
from sql_testing.logger import logger
from sql_testing.schemas import TableSchema
from sql_testing.smt.ast import *
from sql_testing.visitors.expression_encoder import ExpressionEncoder
from sql_testing.visitors.predicate_encoder import PredicateEncoder
from sql_testing.visitors.predicate_encoder import JoinPredicateEncoder


class FMutation:
    def __init__(self,
                 original_output_table: TableSchema,
                 mutant_output_table: TableSchema,
                 node: MutatedNode,
                 env):
        # if 'Duplicate eliminated' in original_output_table.lineage:
        #     original_output_table = original_output_table.ancestors[0]
        # if 'Duplicate eliminated' in mutant_output_table.lineage:
        #     mutant_output_table = mutant_output_table.ancestors[0]
        #
        # assert original_output_table.ancestors == mutant_output_table.ancestors

        self.original_output_table = original_output_table
        self.mutant_output_table = mutant_output_table
        self.node = node
        self.env = env

        if hasattr(self, f"rule_{node.rule}"):
            return
            f = getattr(self, f"rule_{node.rule}")()
            self.env.formulas.append(And(f), label='GMC')
        else:
            logger.warning(f"{node.rule} doesn't have a corresponding GMC rule")

    def rule_comparison(self):
        f = []

        if isinstance(self.node.original, Filter):
            encoder_original = PredicateEncoder(self.original_output_table.ancestors[0], self.node.original.predicate, self.env)
            encoder_mutant = PredicateEncoder(self.mutant_output_table.ancestors[0], self.node.mutant.predicate, self.env)

            input_table_size = self.env.size(self.original_output_table.ancestors[0].table_id)
            f.append(Not(input_table_size == Int(0)))

            for table_size in range(1, self.original_output_table.ancestors[0].bound + 1):
                val_lst = []
                for tuple_idx in range(table_size):
                    val_original, null_original = encoder_original.predicate_for_tuple(tuple_idx)
                    # print(f"debug:null_original:{null_original}")
                    # print(f"debug:val_original:{val_original}")
                    val_mutant, null_mutant = encoder_mutant.predicate_for_tuple(tuple_idx)
                    # print(f"debug:null_mutant:{null_mutant}")
                    val_lst.append(Or([
                        Xor(null_original, null_mutant),
                        And([
                            Not(null_original),
                            Not(null_mutant),
                            Xor(val_original, val_mutant)
                        ])
                    ]))
                f.append(Implies(
                            input_table_size == Int(table_size),
                            Or(val_lst)
                        ))
        elif isinstance(self.node.original, Join):
            match self.node.original.join_type:
                case 'inner join' | 'join':
                    f.append(self.env.size(self.original_output_table.table_id) !=
                             self.env.size(self.mutant_output_table.table_id))
                case 'left join' | 'left outer join':
                    ...
        elif isinstance(self.node.original, GroupBy):
            f.append(
                self.env.size(self.original_output_table.table_id) !=
                self.env.size(self.mutant_output_table.table_id)
            )
        elif isinstance(self.node.original, Project):
            input_table_size = self.env.size(self.original_output_table.ancestors[0].table_id)
            f.append(input_table_size >= Int(1))

            for idx, (original_target, mutant_target) in enumerate(zip(self.node.original.target_list, self.node.mutant.target_list)):
                if original_target == self.node.expression_pair[0] and mutant_target == self.node.expression_pair[1]:
                    if isinstance(self.node.expression_pair[0], CaseWhen):
                        original_cases = original_target.cases
                        mutant_cases = mutant_target.cases

                        for original_case, mutant_case in zip(original_cases, mutant_cases):
                            if isinstance(original_case[0], Expression) and isinstance(mutant_case[0], Expression) \
                                    and original_case[0].operator != mutant_case[0].operator:
                                encoder_original = PredicateEncoder(
                                    self.original_output_table.ancestors[0],
                                    original_case[0],
                                    self.env
                                )
                                encoder_mutant = PredicateEncoder(
                                    self.mutant_output_table.ancestors[0],
                                    mutant_case[0],
                                    self.env
                                )

                                for table_size in range(1, self.original_output_table.ancestors[0].bound + 1):
                                    val_lst = []
                                    for tuple_idx in range(table_size):
                                        val_original, null_original = encoder_original.predicate_for_tuple(tuple_idx)
                                        val_mutant, null_mutant = encoder_mutant.predicate_for_tuple(tuple_idx)
                                        val_lst.append(Or([
                                            Xor(null_original, null_mutant),
                                            And([
                                                Not(null_original),
                                                Not(null_mutant),
                                                Xor(val_original, val_mutant)
                                            ])
                                        ]))
                                    f.append(Implies(
                                        input_table_size == Int(table_size),
                                        Or(val_lst)
                                    ))
                    else:
                        f.append(Or([
                            Xor(
                                self.env.null(self.original_output_table.table_id, 0, idx),
                                self.env.null(self.mutant_output_table.table_id, 0, idx),
                            ),
                            And([
                                Not(Or([
                                    self.env.null(self.original_output_table.table_id, 0, idx),
                                    self.env.null(self.mutant_output_table.table_id, 0, idx),
                                ])),
                                self.env.cell(self.original_output_table.table_id, 0, idx) != self.env.cell(
                                    self.mutant_output_table.table_id, 0, idx)
                            ])
                        ]))
        else:
            logger.warning(f"Warning: comparison for {self.node} not implemented")
        return f

    def rule_drop_where(self):
        return [self.env.size(self.original_output_table.table_id) != self.env.size(self.mutant_output_table.table_id)]

    def rule_join_type(self):
        if isinstance(self.node.original, Filter):
            return [
                self.env.size(self.original_output_table.table_id) != self.env.size(self.mutant_output_table.table_id)
            ]

        types = [self.node.original.join_type.replace('outer join', 'join'),
                 self.node.mutant.join_type.replace('outer join', 'join')]
        f = []
        if ('inner join' in types and any([join_type in types for join_type in ['left join', 'right join', 'full join']])) or ('full join' in types and any([join_type in types for join_type in ['left join', 'right join']])):
            # print(f"debug: {types} satisfied in case 1")
            f.append(self.env.size(self.original_output_table.table_id) !=
                     self.env.size(self.mutant_output_table.table_id))
        elif 'right join' in types and 'left join' in types:
            # print(f"debug: {types} satisfied in case 2")
            left_table_size = self.env.size(self.original_output_table.ancestors[0].table_id)
            right_table_size = self.env.size(self.original_output_table.ancestors[1].table_id)
            encoder = JoinPredicateEncoder(self.original_output_table.ancestors[0], self.original_output_table.ancestors[1], self.node.original.condition, self.env)
            f.append(Not(Or([left_table_size == Int(0),right_table_size == Int(0)])))
            for _left_table_size in range(1, self.original_output_table.ancestors[0].bound + 1):
                for _right_table_size in range(1, self.original_output_table.ancestors[1].bound + 1):
                    option_a_or_option_b = []
                    for left_table_tuple_idx in range(_left_table_size):
                        val_lst = []
                        for right_table_tuple_idx in range(_right_table_size):
                            val, null = encoder.predicate_for_tuple_pair(left_table_tuple_idx, right_table_tuple_idx)
                            val_lst.append(Or([null, And([Not(null), Not(val)])]))
                        option_a_or_option_b.append(And(val_lst))
                    for right_table_tuple_index in range(_right_table_size):
                        val_lst = []
                        for left_table_tuple_index in range(_left_table_size):
                            val, null = encoder.predicate_for_tuple_pair(left_table_tuple_index, right_table_tuple_index)
                            val_lst.append(Or([null, And([Not(null), Not(val)])]))
                        option_a_or_option_b.append(And(val_lst))
                    f.append(Implies(
                                And([
                                    left_table_size == Int(_left_table_size),
                                    right_table_size == Int(_right_table_size)
                                ]), 
                                Or(option_a_or_option_b)
                            ))
        return f
        # types = [self.node.original.join_type.replace('inner join', 'join').replace('outer join', 'join'),
        #          self.node.mutant.join_type.replace('inner join', 'join').replace('outer join', 'join')]
        # f = []
        # if 'join' in types and any([join_type in types for join_type in ['left join', 'right join', 'full join']]):
        #     print(f"debug: {types} handled")
        #     f.append(self.env.size(self.original_output_table.table_id) !=
        #              self.env.size(self.mutant_output_table.table_id))
        # else:
        #     ...
        # return f

    def rule_agg_fun(self):
        f = []
        if isinstance(self.node.original, GroupBy | Filter):
            # mutation of agg fun happens in having clause
            f.append(self.env.size(self.original_output_table.table_id) != self.env.size(self.mutant_output_table.table_id))
        elif isinstance(self.node.original, Project):
            for idx, (original_target, mutant_target) in enumerate(zip(self.node.original.target_list, self.node.mutant.target_list)):
                if (isinstance(original_target, Expression) and isinstance(mutant_target, Expression)
                        and original_target.operator != mutant_target.operator):
                    f.append(self.env.size(self.original_output_table.ancestors[0].table_id) > Int(1))

                    encoder = ExpressionEncoder(self.original_output_table.ancestors[0], self.env)

                    for size in range(1, self.original_output_table.ancestors[0].bound + 1):
                        original_target_val, original_target_null = encoder.expression_for_tuple(original_target, 0, size)
                        mutant_target_val, mutant_target_null = encoder.expression_for_tuple(mutant_target, 0, size)

                        f.append(
                            Implies(
                                self.env.size(self.original_output_table.ancestors[0].table_id) == Int(size),
                                Or([
                                    Xor(original_target_null, mutant_target_null),
                                    And([
                                        Not(Or([original_target_null, mutant_target_null])),
                                        original_target_val != mutant_target_val
                                    ])
                                ])
                            )
                        )

                    # f.append(Or([
                    #     Xor(
                    #         self.env.null(self.original_output_table.table_id, 0, idx),
                    #         self.env.null(self.mutant_output_table.table_id, 0, idx),
                    #     ),
                    #     And([
                    #         Not(Or([
                    #             self.env.null(self.original_output_table.table_id, 0, idx),
                    #             self.env.null(self.mutant_output_table.table_id, 0, idx),
                    #         ])),
                    #         self.env.cell(self.original_output_table.table_id, 0, idx) != self.env.cell(
                    #             self.mutant_output_table.table_id, 0, idx)
                    #     ])
                    # ]))
            # print(repr(self.node.mutant), self.original_output_table, self.mutant_output_table)
        else:
            raise NotImplementedError
        return f

    def rule_agg_fun_distinct(self):
        f = []
        if isinstance(self.node.original, GroupBy | Filter):
            f.append(
                self.env.size(self.original_output_table.table_id) != self.env.size(self.mutant_output_table.table_id))
        elif isinstance(self.node.original, Project):
            for idx, (original_target, mutant_target) in enumerate(zip(self.node.original.target_list, self.node.mutant.target_list)):
                if (isinstance(original_target, Expression) and isinstance(mutant_target, Expression)
                        and original_target.args[0] != mutant_target.args[0]):

                    f.append(self.env.size(self.original_output_table.ancestors[0].table_id) > Int(1))

                    encoder = ExpressionEncoder(self.original_output_table.ancestors[0], self.env)

                    for size in range(1, self.original_output_table.ancestors[0].bound + 1):
                        original_target_val, original_target_null = encoder.expression_for_tuple(original_target, 0, size)
                        mutant_target_val, mutant_target_null = encoder.expression_for_tuple(mutant_target, 0, size)

                        f.append(
                            Implies(
                                self.env.size(self.original_output_table.ancestors[0].table_id) == Int(size),
                                Or([
                                    Xor(original_target_null, mutant_target_null),
                                    And([
                                        Not(Or([original_target_null, mutant_target_null])),
                                        original_target_val != mutant_target_val
                                    ])
                                ])
                            )
                        )
        else:
            raise NotImplementedError
        return f

    def rule_drop_having(self):
        f = [self.env.size(self.original_output_table.table_id) != self.env.size(self.mutant_output_table.table_id)]
        return f

    def rule_select_distinct(self):
        f = []
        # input_table = self.mutant_output_table.ancestors[0]
        # input_size_variable = self.env.size(input_table.table_id)
        # f.append(input_size_variable >= 2)

        # print(input_table)
        # print(self.env.db)
        #
        # def _f_duplicate_of(table, tuple_idx, others):
        #     f_duplicate = []
        #     for idx in others:
        #         tuple_distinct = []
        #         for column in table:
        #             column_id = column.column_id
        #             this_tuple_cell = self.env.db[table.table_id, tuple_idx, column_id]
        #             other_tuple_cell = self.env.db[table.table_id, idx, column_id]
        #             tuple_distinct.append(self.env.compare_cell(operator.eq, this_tuple_cell, other_tuple_cell))
        #         f_duplicate.append(And(tuple_distinct))
        #     return Or(*f_duplicate)
        #
        # f = []
        # for table_size in range(input_table.bound + 1):
        #     case = (input_size_variable == table_size)
        #
        #     then = Or([_f_duplicate_of(input_table, this_idx, [i for i in range(table_size) if i != this_idx]) for this_idx in range(table_size)])
        #
        #     f.append(Implies(case, then))
        # print(f)
        # return And(*f)

        # print(self.env.db)
        # print(self.original_output_table.table_id)
        # print(self.mutant_output_table.table_id)
        # print(self.mutant_output_table.ancestors[0].table_id)

        # f.append(And([ self.env.size(self.mutant_output_table.ancestors[0].table_id) > Int(1)]))

        # output sizes different
        f.append(
            self.env.size(self.mutant_output_table.table_id) !=
            self.env.size(self.mutant_output_table.ancestors[0].table_id)
        )

        return f

    def filter_encoding_helper(self):
        f = []
        encoder_original = PredicateEncoder(self.original_output_table.ancestors[0], self.node.original.predicate, self.env)
        encoder_mutant = PredicateEncoder(self.mutant_output_table.ancestors[0], self.node.mutant.predicate, self.env)
        input_table_size = self.env.size(self.original_output_table.ancestors[0].table_id)
        f.append(Not(input_table_size == Int(0)))
        for table_size in range(1, self.original_output_table.ancestors[0].bound + 1):
            val_lst = []
            for tuple_idx in range(table_size):
                val_original, null_original = encoder_original.predicate_for_tuple(tuple_idx)
                val_mutant, null_mutant = encoder_mutant.predicate_for_tuple(tuple_idx)
                val_lst.append(Or([
                    Xor(null_original, null_mutant),
                    And([
                        Not(null_original),
                        Not(null_mutant),
                        Xor(val_original, val_mutant)
                    ])
                ]))
            f.append(Implies(
                        input_table_size == Int(table_size),
                        Or(val_lst)
                    ))
        return f

    def rule_conjunction_drop_left(self):
        f = []
        if isinstance(self.node.original, Filter):
            f.extend(self.filter_encoding_helper())
        elif isinstance(self.node.original, GroupBy):
            f.append(self.env.size(self.original_output_table.table_id) != self.env.size(self.mutant_output_table.table_id))
        return f
    
    def rule_conjunction_drop_right(self):
        f = []
        if isinstance(self.node.original, Filter):
            f.extend(self.filter_encoding_helper())
        elif isinstance(self.node.original, GroupBy):
            f.append(self.env.size(self.original_output_table.table_id) != self.env.size(self.mutant_output_table.table_id))
        return f
    
    def rule_disjunction_drop_left(self):
        f = []
        if isinstance(self.node.original, Filter):
            f.extend(self.filter_encoding_helper())
        elif isinstance(self.node.original, GroupBy):
            f.append(self.env.size(self.original_output_table.table_id) != self.env.size(self.mutant_output_table.table_id))
        return f
    
    def rule_disjunction_drop_right(self):
        f = []
        if isinstance(self.node.original, Filter):
            f.extend(self.filter_encoding_helper())
        elif isinstance(self.node.original, GroupBy):
            f.append(self.env.size(self.original_output_table.table_id) != self.env.size(self.mutant_output_table.table_id))
        return f

    def rule_bool_operator(self):
        return [self.env.size(self.original_output_table.table_id) != self.env.size(self.mutant_output_table.table_id)]

    def rule_input_table(self):
        return []
        # return [self.env.size(self.mutant_output_table.table_id) != self.env.size(self.original_output_table.table_id)]

    def rule_union_distinct(self):
        return [self.env.size(self.mutant_output_table.table_id) != self.env.size(self.original_output_table.table_id)]

    def rule_drop_orderby(self):
        f = []
        input_table_size = self.env.size(self.original_output_table.ancestors[0].table_id)
        f.append(input_table_size > Int(0))
        for size in range(1, self.mutant_output_table.bound + 1):
            case = []
            for idx in range(size):
                for column in self.original_output_table:
                    case.append(Or([
                        self.env.null(self.original_output_table.table_id, idx, column.column_id) !=
                        self.env.null(self.mutant_output_table.table_id, idx, column.column_id),
                        self.env.cell(self.original_output_table.table_id, idx, column.column_id) !=
                        self.env.cell(self.mutant_output_table.table_id, idx, column.column_id)
                    ]))
            case = Implies(
                self.env.size(input_table_size.table_id) == Int(size),
                Or(case)
            )
            f.append(case)

        return f

    def rule_agg_fun_count(self):
        f = []
        if isinstance(self.node.original, GroupBy | Filter):
            f.append(self.env.size(self.original_output_table.table_id) != self.env.size(self.mutant_output_table.table_id))
        elif isinstance(self.node.original, Project):
            for idx, (original_target, mutant_target) in enumerate(zip(self.node.original.target_list, self.node.mutant.target_list)):
                if (isinstance(original_target, Expression) and isinstance(mutant_target, Expression)
                        and len(original_target.args) >= 2 and len(mutant_target.args) >= 2
                        and original_target.args[1] != mutant_target.args[1]):

                    f.append(self.env.size(self.original_output_table.ancestors[0].table_id) > Int(1))

                    encoder = ExpressionEncoder(self.original_output_table.ancestors[0], self.env)

                    for size in range(1, self.original_output_table.ancestors[0].bound + 1):
                        or_null = []
                        for tuple_idx in range(size):
                            _, original_target_null = encoder.expression_for_tuple(original_target.args[1], tuple_idx)
                            or_null.append(original_target_null)

                        f.append(Implies(
                            self.env.size(self.original_output_table.ancestors[0].table_id) == Int(size),
                            Or(or_null)
                        ))
        else:
            raise NotImplementedError
        return f

    def rule_comparison_to_is_null(self):
        f = []

        if isinstance(self.node.original, Filter):
            encoder_mutant = PredicateEncoder(self.mutant_output_table.ancestors[0], self.node.mutant.predicate, self.env)

            input_table_size = self.env.size(self.original_output_table.ancestors[0].table_id)
            f.append(input_table_size > Int(0))

            for table_size in range(1, self.original_output_table.ancestors[0].bound + 1):
                val_lst = []
                for tuple_idx in range(table_size):
                    val_mutant, null_mutant = encoder_mutant.predicate_for_tuple(tuple_idx)
                    val_lst.append(And([Not(null_mutant), val_mutant]))
                f.append(Implies(
                            input_table_size == Int(table_size),
                            Or(val_lst)
                        ))
        elif isinstance(self.node.original, Join):
            match self.node.original.join_type:
                case 'inner join' | 'join':
                    f.append(self.env.size(self.original_output_table.table_id) !=
                             self.env.size(self.mutant_output_table.table_id))
                case 'left join' | 'left outer join':
                    ...
        elif isinstance(self.node.original, GroupBy):
            f.append(
                self.env.size(self.original_output_table.table_id) !=
                self.env.size(self.mutant_output_table.table_id)
            )
        else:
            # logger.warning(f"Warning: comparison for {self.node} not implemented")
            pass
        return f
