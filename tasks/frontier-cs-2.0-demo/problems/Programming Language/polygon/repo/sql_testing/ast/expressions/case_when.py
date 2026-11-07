from typing import Optional, List, Tuple

from typing_extensions import Self

from sql_testing.ast.expressions.attribute import Attribute
from sql_testing.ast.expressions.expression import Expression
from sql_testing.ast.expressions.literal import Literal
from sql_testing.ast.query import Query


class CaseWhen(Expression):
    def __init__(self,
                 cases: List[Tuple[Expression, Expression]],
                 default: Optional[Expression] = None,
                 alias: Optional[str] = None):
        super().__init__('case', [], alias)
        self.cases = cases
        self.default = default

    def __str__(self):
        case_when_str = f"CASE {''.join([f'WHEN {condition} THEN {result} ' for condition, result in self.cases])}" \
                        f"{f'ELSE {self.default} ' if self.default is not None else ''}" \
                        f"END" \
                        f"{f' AS {self.alias}' if self.alias is not None else ''}"
        return case_when_str

    def __repr__(self):
        if self.alias is not None:
            return f"CaseWhen(cases={repr(self.cases)}, default={repr(self.default)}, alias={repr(self.alias)})"
        else:
            return f"CaseWhen(cases={repr(self.cases)}, default={repr(self.default)})"
