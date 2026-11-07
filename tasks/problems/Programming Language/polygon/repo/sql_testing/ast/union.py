from typing import List, Optional

from sql_testing.ast.node import Node
from sql_testing.ast.query import Query


class Union(Node):
    def __init__(self,
                 queries: List[Query],
                 allow_duplicates: Optional[bool] = False,
                 alias: str = None):
        super().__init__()
        self.queries = queries
        self.allow_duplicates = allow_duplicates
        self.alias = alias

        self.initialized = False

    def __str__(self):
        if self.alias is None:
            if self.allow_duplicates:
                return " UNION ALL ".join([str(query) for query in self.queries])
            return " UNION ".join([str(query) for query in self.queries])
        else:
            if self.allow_duplicates:
                return f'({" UNION ALL ".join([str(query) for query in self.queries])}) AS {self.alias}'
            return f'({" UNION ".join([str(query) for query in self.queries])}) AS {self.alias}'

    def __repr__(self):
        if self.alias is None:
            return f"Union(queries={repr(self.queries)}, allow_duplicates={repr(self.allow_duplicates)})"
        else:
            return f"Union(queries={repr(self.queries)}, allow_duplicates={repr(self.allow_duplicates)}, alias={self.alias})"
