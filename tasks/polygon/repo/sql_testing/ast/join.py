from typing import Optional

from sql_testing.ast.expressions.expression import Expression
from sql_testing.ast.expressions.literal import Literal
from sql_testing.ast.node import Node

_type_str_map = {
    'join': 'JOIN',
    'inner join': 'INNER JOIN',
    'left join': 'LEFT JOIN',
    'right join': 'RIGHT JOIN',
    'full join': 'FULL JOIN',
    'left outer join': 'LEFT OUTER JOIN',
    'right outer join': 'RIGHT OUTER JOIN',
    'full outer join': 'FULL OUTER JOIN',
    'natural join': 'NATURAL JOIN',
    'cross join': 'CROSS JOIN',
}


class Join(Node):
    def __init__(self,
                 left,
                 right,
                 join_type: str,
                 condition: Optional[Expression | Literal] = None,
                 using: Optional[Literal] = None,
                 ):
        super().__init__()
        self.left = left
        self.right = right
        self.join_type = join_type
        self.condition = condition
        self.using = using

    def __str__(self):
        from sql_testing.ast.query import Query

        left_str = str(self.left)
        right_str = str(self.right)

        if isinstance(self.left, Query):
            left_str = f"({left_str})"
            if self.left.alias is not None:
                left_str = f"{left_str} AS {self.left.alias}"
        if isinstance(self.right, Query):
            right_str = f"({right_str})"
            if self.right.alias is not None:
                right_str = f"{right_str} AS {self.right.alias}"

        if self.condition is not None:
            return f"{left_str} {_type_str_map[self.join_type]} {right_str} ON {self.condition}"
        elif self.using is not None:
            return f"{left_str} {_type_str_map[self.join_type]} {right_str} USING ({self.using.value})"

        if self.join_type == 'cross join':
            return f"{left_str}, {right_str}"
        return f"{left_str} {_type_str_map[self.join_type]} {right_str}"

    def __repr__(self):
        return f"Join(left={repr(self.left)}, right={repr(self.right)}, join_type={repr(self.join_type)}, condition={repr(self.condition)}, using={repr(self.using)})"
