from typing import List, Optional
from typing_extensions import Self

from sql_testing.ast.expressions.attribute import Attribute
from sql_testing.ast.expressions.function import FunctionCall
from sql_testing.ast.expressions.literal import Literal
from sql_testing.ast.expressions.operator import Operator, _op_callable_map
from sql_testing.ast.node import Node

_op_str_map = {
    'add': '+',
    'sub': '-',
    'mul': '*',
    'div': '/',
    'neg': '-',

    'gt': '>',
    'gte': '>=',
    'lt': '<',
    'lte': '<=',
    'eq': '=',
    'neq': '<>',

    'not': 'NOT',
    'and': 'AND',
    'or': 'OR',

    'in': 'IN',
    'nin': 'NOT IN',
    'is_not_null': 'IS NOT NULL',
    'is_null': 'IS NULL',
    'if': 'IF',
    'distinct': 'DISTINCT',
    'coalesce': 'COALESCE',
    'cast': 'CAST',
}


class Expression(Node):
    def __init__(self,
                 operator: Optional[str],
                 args: List[Literal | Attribute | Self | 'Query'],
                 alias: Optional[str] = None):

        self.operator = operator
        self.args = args
        self.alias = alias

        if operator == 'missing':
            self.operator = 'is_null'
        elif operator == 'exists':
            self.operator = 'is_not_null'

        if operator == 'in' or operator == 'nin':
            if isinstance(self.args[1], Literal | Expression):
                self.args[1] = [self.args[1]]

        if self.operator in _op_callable_map:
            self.operator_callable = Operator(self.operator)
        else:
            self.operator_callable = FunctionCall(self.operator)

    def __str__(self):
        from sql_testing.ast.query import Query

        if isinstance(self.operator_callable, Operator):
            if len(self.args) == 2 or self.operator_callable.name in ['and', 'or']:
                expression_str = f" {_op_str_map[str(self.operator_callable)]} ".join(
                    [str(operand) for operand in self.args])
                if self.operator_callable.name in ['add', 'sub', 'mul', 'div', 'neg', 'and', 'or']:
                    expression_str = f"({expression_str})"
            else:
                expression_str = f"{_op_str_map[str(self.operator_callable)]}({', '.join([str(operand) for operand in self.args])})"
        elif self.operator in ['is_null', 'is_not_null']:
            expression_str = f"{str(self.args[0])} {_op_str_map[self.operator]}"
        else:
            # function call
            if str(self.operator_callable.name) in ['max', 'min', 'sum', 'avg', 'count']:
                if self.args[0]:
                    expression_str = f"{str(self.operator_callable.name).upper()}(DISTINCT {str(self.args[1])})"
                else:
                    expression_str = f"{str(self.operator_callable.name).upper()}({str(self.args[1])})"
            elif str(self.operator_callable.name) == 'in':

                if isinstance(self.args[0], list):
                    expression_str = f"({', '.join(arg.name for arg in self.args[0])}) IN "
                else:
                    expression_str = f"{self.args[0]} IN "

                if isinstance(self.args[1], list):
                    expression_str += f"({', '.join([str(x) for x in self.args[1]])})"
                else:
                    expression_str += f'({self.args[1]})'

            elif str(self.operator_callable.name) == 'nin':

                if isinstance(self.args[0], list):
                    expression_str = f"({', '.join(arg.name for arg in self.args[0])}) NOT IN "
                else:
                    expression_str = f"{self.args[0]} NOT IN "

                if isinstance(self.args[1], list):
                    expression_str += f"({', '.join([str(x) for x in self.args[1]])})"
                else:
                    expression_str += f'({self.args[1]})'

            else:
                expression_str = f"{str(self.operator_callable.name).upper()}({', '.join([f'({arg})' if isinstance(arg, Query) else str(arg) for arg in self.args])})"

        if self.alias is not None:
            return f"{expression_str} AS {self.alias}"
        else:
            return expression_str

    def __repr__(self):
        if self.alias is not None:
            return f"Expression(operator={repr(str(self.operator_callable))}, args={repr(self.args)}, alias={repr(self.alias)})"
        else:
            return f"Expression(operator={repr(str(self.operator_callable))}, args={repr(self.args)})"
