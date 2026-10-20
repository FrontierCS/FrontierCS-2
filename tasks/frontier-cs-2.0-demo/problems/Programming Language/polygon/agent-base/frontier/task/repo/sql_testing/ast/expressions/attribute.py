from typing import Optional

from sql_testing.ast.node import Node


class Attribute(Node):
    def __init__(self, name: str, alias: Optional[str] = None):
        self.name = name
        self.alias = alias

    def __str__(self):
        if self.alias is None:
            return self.name
        else:
            return f"{self.name} AS {self.alias}"

    def __repr__(self):
        if self.alias is None:
            return f"Attribute(name={repr(self.name)})"
        else:
            return f"Attribute(name={repr(self.name)}, alias={repr(self.alias)})"
