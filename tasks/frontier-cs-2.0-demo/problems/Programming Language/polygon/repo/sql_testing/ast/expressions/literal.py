from typing import Any

from sql_testing.ast.node import Node


class Literal(Node):
    def __init__(self, value: Any, alias: str = None):
        self.value = value
        self.alias = alias

    def __str__(self):
        if isinstance(self.value, str):
            value_str = f"'{self.value}'"
        else:
            value_str = str(self.value)
        if self.alias is None:
            return value_str
        else:
            return f"{value_str} AS {self.alias}"

    def __repr__(self):
        if self.alias is None:
            return f"Literal(value={repr(self.value)})"
        return f"Literal(value={repr(self.value)}, alias={repr(self.alias)})"
