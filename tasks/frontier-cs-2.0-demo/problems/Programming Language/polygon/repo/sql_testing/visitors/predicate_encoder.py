from sql_testing.smt.ast import *
from sql_testing.visitors.expression_encoder import ExpressionEncoder


class PredicateEncoder(ExpressionEncoder):
    def __init__(self, schema, predicate, env, outer_tuple_id=None):
        super().__init__(schema, env, outer_tuple_id)
        self.predicate = predicate
        self.tuple_idx = None

    def predicate_for_tuple(self, idx: int):
        self.tuple_idx = idx

        val, null = self.predicate.accept(self)
        val = EnsureBool(val)
        return val, null


class JoinPredicateEncoder(PredicateEncoder):
    def __init__(self, left_table, right_table, predicate, env):
        self.left_table = left_table
        self.right_table = right_table
        self.predicate = predicate
        self.env = env
        self.tuple_idx_pair = None
        self.subquery_table_map = {}

    def predicate_for_tuple_pair(self, left_idx: int, right_idx: int):
        self.tuple_idx_pair = (left_idx, right_idx)

        val, null = self.predicate.accept(self)
        val = EnsureBool(val)
        return val, null

    def visit_Attribute(self, node):
        try:
            column = self.left_table[node.name]
            return self.env.db[self.left_table.table_id, self.tuple_idx_pair[0], column.column_id].accept(self)
        except SyntaxError:
            column = self.right_table[node.name]
            return self.env.db[self.right_table.table_id, self.tuple_idx_pair[1], column.column_id].accept(self)
