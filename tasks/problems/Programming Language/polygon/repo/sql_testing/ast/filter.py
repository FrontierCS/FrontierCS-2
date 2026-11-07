from sql_testing.ast.expressions.expression import Expression
from sql_testing.ast.expressions.literal import Literal
from sql_testing.ast.node import Node


class Filter(Node):
    def __init__(self, predicate: Expression | Literal):
        super().__init__()
        self.predicate = predicate

    def __str__(self):
        return str(self.predicate)

    def __repr__(self):
        return f"Filter(predicate={repr(self.predicate)})"
