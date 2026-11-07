from typing import Optional, Mapping
from typing_extensions import Self

from sql_testing.ast.filter import Filter
from sql_testing.ast.group_by import GroupBy
from sql_testing.ast.join import Join
from sql_testing.ast.node import Node, MutatedNode
from sql_testing.ast.order_by import OrderBy
from sql_testing.ast.project import Project
from sql_testing.ast.scan import Scan


class Query(Node):
    def __init__(self,
                 select_clause: Project | MutatedNode,
                 from_clause: Scan | Self | Join | MutatedNode,
                 where_clause: Optional[Filter | MutatedNode] = None,
                 group_by_clause: Optional[GroupBy] = None,
                 order_by_clause: Optional[OrderBy] = None,
                 alias: Optional[str] = None,
                 cte: Mapping[str, Self] = None,
                 ):

        super().__init__()
        self.cte = cte
        self.select_clause = select_clause
        self.from_clause = from_clause
        self.where_clause = where_clause
        self.group_by_clause = group_by_clause
        self.order_by_clause = order_by_clause
        self.alias = alias

        self.initialized = False

    def __str__(self):
        from_clause_str = str(self.from_clause)

        # from clause is a subquery
        if isinstance(self.from_clause, Query):
            from_clause_str = f"({from_clause_str})"
            if self.from_clause.alias is not None:
                from_clause_str = f"{from_clause_str} AS {self.from_clause.alias}"
        elif isinstance(self.from_clause, MutatedNode) and isinstance(self.from_clause.mutant, Query):
            from_clause_str = f"({from_clause_str})"
            if self.from_clause.mutant.alias is not None:
                from_clause_str = f"{from_clause_str} AS {self.from_clause.mutant.alias}"

        query_str = ''

        if self.cte is not None:
            query_str = f'WITH {list(self.cte.keys())[0]} AS ({list(self.cte.values())[0]}) '

        query_str += f"SELECT {str(self.select_clause)} FROM {from_clause_str}"

        if self.where_clause is not None:
            query_str = f"{query_str} WHERE {self.where_clause}"

        if self.group_by_clause is not None:
            query_str = f"{query_str} GROUP BY {self.group_by_clause}"

        if self.order_by_clause is not None:
            query_str = f"{query_str} ORDER BY {self.order_by_clause}"

        return query_str

    def __repr__(self):
        return f"Query(select_clause={repr(self.select_clause)}, " \
               f"from_clause={repr(self.from_clause)}, " \
               f"where_clause={repr(self.where_clause)}, " \
               f"group_by_clause={repr(self.group_by_clause)}, " \
               f"order_by_clause={repr(self.order_by_clause)}, " \
               f"cte={repr(self.cte)}, " \
               f"alias={repr(self.alias)})"
