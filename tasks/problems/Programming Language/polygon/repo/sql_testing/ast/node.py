from copy import deepcopy
from typing import Optional


class Node:
    def __init__(self):
        self.label = None
        self.parent = []
        self.ctx = {}

    def set_parent(self, parent):
        self.parent = deepcopy(parent)

    def accept(self, visitor):
        method_name = f'visit_{self.__class__.__name__}'
        if hasattr(visitor, method_name):
            visit = getattr(visitor, method_name)
        else:
            visit = getattr(visitor, 'visit')
        return visit(self)


class MutatedNode(Node):
    def __init__(self,
                 original: Node,
                 mutant: Node,
                 rule: str = None,
                 expression_pair: Optional[tuple] = None):
        super().__init__()
        self.rule = rule
        self.original = original
        self.mutant = mutant
        self.expression_pair = expression_pair
        self.str_mutant_only = True

    def set_parent(self, parent):
        self.original.set_parent(parent)
        self.mutant.set_parent(parent)

    def __str__(self):
        if self.str_mutant_only:
            return str(self.mutant)
        return f"[({self.original}) -> ({self.mutant})]"

    def __repr__(self):
        return f"MutatedNode(rule={repr(self.rule)}, original={repr(self.original)}, mutant={repr(self.mutant)})"
