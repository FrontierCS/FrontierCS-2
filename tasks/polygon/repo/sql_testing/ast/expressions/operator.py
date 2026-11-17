import operator

from sql_testing.ast.node import Node
from sql_testing.smt.ast import *

_op_callable_map = {
    'add': operator.add,
    'sub': operator.sub,
    'mul': operator.mul,
    'div': operator.truediv,
    'neg': operator.neg,

    'gt': operator.gt,
    'gte': operator.ge,
    'lt': operator.lt,
    'lte': operator.le,
    'eq': operator.eq,
    'neq': operator.ne,

    'not': Not,
    'and': And,
    'or': Or,
}


class Operator(Node):
    def __init__(self, name: str):
        self.name = name

    def __str__(self):
        return self.name

    def __call__(self, *args, **kwargs):
        if len(args) == 0 and len(kwargs) == 0:
            return _op_callable_map[self.name]
        return _op_callable_map[self.name](list(args))

    def __repr__(self):
        return f"Operator(name={repr(self.name)})"
