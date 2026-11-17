from sql_testing.ast.node import Node


class FunctionCall(Node):
    def __init__(self, name: str):
        self.name = name

    def __str__(self):
        return self.name

    def __repr__(self):
        return f"FunctionCall(name={repr(self.name)})"
