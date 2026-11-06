import functools


class SMTNode:
    def accept(self, visitor):
        method_name = f'visit_{self.__class__.__name__}'
        if hasattr(visitor, method_name):
            visit = getattr(visitor, method_name)
        else:
            visit = getattr(visitor, 'visit')
        return visit(self)

    def __eq__(self, other):
        return Eq(self, other)

    def __ne__(self, other):
        return Neq(self, other)

    def __gt__(self, other):
        return Gt(self, other)

    def __ge__(self, other):
        return Gte(self, other)

    def __lt__(self, other):
        return Lt(self, other)

    def __le__(self, other):
        return Lte(self, other)

    def __add__(self, other):
        return Plus(self, other)

    def __sub__(self, other):
        return Minus(self, other)

    def __mul__(self, other):
        return Mul(self, other)

    def __truediv__(self, other):
        return Div(self, other)

    def __neg__(self):
        return Neg(self)

    def return_type(self):
        pass


class SMTCell(SMTNode):
    def __init__(self, table_id, row_id, column_id):
        self.table_id = table_id
        self.row_id = row_id
        self.column_id = column_id

    def return_type(self):
        return 'Int'

    def __str__(self):
        return f"CELL{self.table_id, self.row_id, self.column_id}"


class SMTNull(SMTNode):
    def __init__(self, table_id, row_id, column_id):
        self.table_id = table_id
        self.row_id = row_id
        self.column_id = column_id

    def return_type(self):
        return 'Bool'

    def __str__(self):
        return f"NULL{self.table_id, self.row_id, self.column_id}"


class SMTGrouping(SMTNode):
    def __init__(self, table_id, tuple_id, group_id):
        self.table_id = table_id
        self.tuple_id = tuple_id
        self.group_id = group_id

    def return_type(self):
        return 'Bool'

    def __str__(self):
        return f"GROUPING{self.table_id, self.tuple_id, self.group_id}"


class Deleted(SMTNode):
    def __init__(self, table_id, tuple_id):
        self.table_id = table_id
        self.tuple_id = tuple_id

    def return_type(self):
        return 'Bool'

    def __str__(self):
        return f"Deleted({self.table_id}, {self.tuple_id})"


class SMTBelongsToGroup(SMTNode):
    def __init__(self, qid, gid):
        self.qid = qid
        self.gid = gid

    def __str__(self):
        return f"BELONGSTOGROUP{self.qid, self.gid}"


class SMTSize(SMTNode):
    def __init__(self, table_id):
        self.table_id = table_id

    def return_type(self):
        return 'Int'

    def __str__(self):
        return f"SIZE({self.table_id})"


class Choice(SMTNode):
    def __init__(self, table_id, bit_id):
        self.table_id = table_id
        self.bit_id = bit_id

    def return_type(self):
        return 'Int'

    def __str__(self):
        return f"Choice({self.table_id}, {self.bit_id})"


class CompNode(SMTNode):
    def __init__(self, a, b):
        assert isinstance(a, SMTNode)
        assert isinstance(b, SMTNode)
        self.a = a
        self.b = b

    def return_type(self):
        return 'Bool'


class And(SMTNode):
    def __init__(self, conjunct):
        assert isinstance(conjunct, list)
        # for x in conjunct:
        #     if not isinstance(x, SMTNode):
        #         raise TypeError
        self.conjunct = conjunct

    def return_type(self):
        return 'Bool'

    def __str__(self):
        return f"And({', '.join([str(x) for x in self.conjunct])})"


class Or(SMTNode):
    def __init__(self, disjunct):
        assert isinstance(disjunct, list)
        self.disjunct = disjunct

    def return_type(self):
        return 'Bool'

    def __str__(self):
        return f"Or({', '.join([str(x) for x in self.disjunct])})"


class Xor(SMTNode):
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def return_type(self):
        return 'Bool'

    def __str__(self):
        return f"{str(self.a)} ⊕ {str(self.b)}"


class Not(SMTNode):
    def __init__(self, node):
        self.node = node

    def return_type(self):
        return 'Bool'

    def __str__(self):
        return f"Not({str(self.node)})"


class Implies(SMTNode):
    def __init__(self, premise, conclusion):
        self.premise = premise
        self.conclusion = conclusion

    def return_type(self):
        return 'Bool'

    def __str__(self):
        return f"{str(self.premise)} => {str(self.conclusion)}"


class If(SMTNode):
    def __init__(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c

    def return_type(self):
        assert self.b.return_type() == self.c.return_type()
        return self.b.return_type()

    def __str__(self):
        return f"ITE({str(self.a)}, {str(self.b)}, {str(self.c)})"


class Gte(CompNode):
    def __init__(self, a, b):
        super().__init__(a, b)

    def __str__(self):
        return f"{str(self.a)} >= {str(self.b)}"


class Gt(CompNode):
    def __init__(self, a, b):
        super().__init__(a, b)

    def __str__(self):
        return f"{str(self.a)} > {str(self.b)}"


class Lte(CompNode):
    def __init__(self, a, b):
        super().__init__(a, b)

    def __str__(self):
        return f"{str(self.a)} <= {str(self.b)}"


class Lt(CompNode):
    def __init__(self, a, b):
        super().__init__(a, b)

    def __str__(self):
        return f"{str(self.a)} < {str(self.b)}"


class Eq(CompNode):
    def __init__(self, a, b):
        super().__init__(a, b)

    def __str__(self):
        return f"{str(self.a)} == {str(self.b)}"


class Neq(CompNode):
    def __init__(self, a, b):
        super().__init__(a, b)

    def __str__(self):
        return f"{str(self.a)} != {str(self.b)}"


class Plus(SMTNode):
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def return_type(self):
        return 'Int'

    def __str__(self):
        return f"{self.a} + {self.b}"


class Minus(SMTNode):
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def return_type(self):
        return 'Int'

    def __str__(self):
        return f"{self.a} - {self.b}"


class Mul(SMTNode):
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def return_type(self):
        return 'Int'

    def __str__(self):
        return f"{self.a} * {self.b}"


class Div(SMTNode):
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def return_type(self):
        return 'Int'

    def __str__(self):
        return f"{self.a} / {self.b}"


class Neg(SMTNode):
    def __init__(self, x):
        self.x = x

    def return_type(self):
        return 'Int'

    def __str__(self):
        return f"-{self.x}"


class Int(SMTNode):
    def __init__(self, x):
        self.x = x

    def return_type(self):
        return 'Int'

    def __str__(self):
        return str(self.x)


class Bool(SMTNode):
    def __init__(self, x):
        self.x = x

    def return_type(self):
        return 'Bool'

    def __str__(self):
        return str(self.x).upper()


# Functions
def Sum(args):
    if not args:
        return Int(0)
    return functools.reduce(lambda x, y: x + y, args)


def EnsureInt(node):
    if node.return_type() == 'Bool':
        return If(node, Int(1), Int(0))
    return node


def EnsureBool(node):
    if node.return_type() == 'Int':
        return node != Int(0)
    return node
