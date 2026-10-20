from sql_testing.ast.node import Node


class Scan(Node):
    def __init__(self,
                 table: str,
                 alias: str = None):
        super().__init__()
        self.table = table
        self.alias = alias

    def __str__(self):
        if self.alias is None:
            return str(self.table)
        else:
            return f"{self.table} AS {self.alias}"

    def __repr__(self):
        if self.alias is None:
            return f"Scan(table={repr(self.table)})"
        else:
            return f"Scan(table={repr(self.table)}, alias={repr(self.alias)})"
