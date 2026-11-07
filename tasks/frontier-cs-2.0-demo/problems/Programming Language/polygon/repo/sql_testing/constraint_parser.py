from datetime import datetime

import yaml
from lark import (
    Lark,
    Transformer,
)


class ConstraintVisitor(Transformer):
    def constraints(self, tree):
        return [node for node in tree if node is not None]

    def membership(self, tree):
        # Column dependence
        # Example: Employee.departmentId <- Department.id
        if isinstance(tree[1], str):
            return {'eq': tree}
        # Column value membership
        # Example: X.Y <- [0,10]
        elif isinstance(tree[1], tuple):
            lower_bound = tree[1][0]
            upper_bound = tree[1][1]
            return {'domain': [tree[0], lower_bound, upper_bound]}
        # Column value membership
        # Example: SurveyLog.action <- {'show', 'answer', 'skip'}
        elif isinstance(tree[1], list):
            enum = []
            for value in tree[1]:
                enum.append(value)
            return {'enum': [str(tree[0]), enum]}
        else:
            raise NotImplementedError

    def inclusion(self, tree):
        # Column value membership
        # Example: X.Y -> [0,10]
        if isinstance(tree[1], tuple):
            lower_bound = tree[1][0]
            upper_bound = tree[1][1]
            return {'inclusion': [{'from': [tree[0], lower_bound]}, {'to': [tree[0], upper_bound]}]}
        # Column value membership
        # Example: SurveyLog.action -> {'show', 'answer', 'skip'}
        elif isinstance(tree[1], list):
            disjunction = []
            for value in tree[1]:
                disjunction.append({'val': [tree[0], value]})
            return {'inclusion1': disjunction}
        else:
            raise NotImplementedError

    def not_null(self, tree):
        # {'not_null': 'Customers.customer_id'}
        attr = tree[0]
        if isinstance(attr, dict):
            attr = attr['value']
        return {'not_null': attr}

    def if_then(self, tree):
        attr = tree[0]
        comp = tree[1]
        attr2 = tree[2]
        attr3 = tree[3]
        comp2 = tree[4]
        attr4 = tree[5]
        return {'if': [{'col1a': attr}, {'comp1': comp}, {'col1b': attr2}, {'col2a': attr3}, {'comp2': comp2},
                       {'col2b': attr4}]}

    def comparison(self, tree):
        operands = [tree[0], tree[2]]
        operator = tree[1]
        return {operator: operands}

    def unique(self, tree):
        return {'distinct': [node for node in tree]}

    def increment(self, tree):
        return {'inc': [node for node in tree]}

    def dec(self, tree):
        return {'dec': [node for node in tree]}

    def consec(self, tree):
        return {'consec': [node for node in tree]}

    def override(self, tree):
        return {'override': tree[0]}

    def op(self, tree):
        operator = str(tree[0])
        if operator == '>':
            return 'gt'
        elif operator == '>=':
            return 'gte'
        elif operator == '<':
            return 'lt'
        elif operator == '<=':
            return 'lte'
        elif operator == '!=':
            return 'neq'
        elif operator == '^':
            return 'join'
        elif operator == '=':
            return 'equal'
        else:
            raise NotImplementedError(operator)

    def attribute(self, tree):
        return f"{tree[0]}.{tree[1]}"

    def NUMBER(self, tree):
        return eval(tree)

    def null(self, tree):
        return None

    def value_range(self, tree):
        return tree[0], tree[1]

    def value_items(self, tree):
        return tree

    def DATE(self, tree):
        return datetime.fromisoformat(tree).date()

    def STRING(self, tree):
        return str(tree.value[1:-1])


class ConstraintParser:
    def __init__(self):
        self._visitor = ConstraintVisitor()
        self._parser = Lark(
            r'''
            constraints: constraint (";" constraint)* [";"]
            ?constraint : membership
                        | inclusion
                        | not_null
                        | comparison
                        | unique
                        | increment
                        | consec
                        | dec
                        | override
                        | if_then
            membership : attribute "<-" attribute
                       | attribute "<-" value_range
                       | attribute "<-" value_items

            inclusion :  attribute "->" value_items
                      | attribute "->" value_range

            if_then:  value op value  "=>"  value op value

            not_null : attribute "!=" NULL
            comparison : value op value
            override: attribute "|" "int+" NULL | attribute "|" "varchar+" NULL
            unique : "unique" "(" [attribute ("," attribute)*] ")"
            increment: "inc" "(" [attribute ("," attribute)*] ")"
            dec: "dec" "(" [attribute ("," attribute)*] ")"
            consec: "consec" "(" [attribute ("," attribute)*] ")"
            ?value : attribute | NUMBER | DATE | STRING | NULL
            !op : ">" | "<" | "!=" | ">=" | "<=" | "^" | "="
            attribute : /[^\W\d]\w*/ "." /[^\W\d]\w*/

            value_range : "[" NUMBER "," NUMBER "]"
                        | "[" DATE "," DATE "]"
            value_items : "{" [value_item ("," value_item)*] "}"
            ?value_item : STRING | NUMBER | NULL

            NULL: "null" | "NULL"
            DATE: NUMBER "-" NUMBER "-" NUMBER
            STRING : "'" /[^']+/ "'" | ESCAPED_STRING | /[^\W\d]\w*/

            %import common.ESCAPED_STRING
            %import common.SIGNED_NUMBER -> NUMBER
            %import common.WS
            %ignore WS
            ''',
            start='constraints')

    def parse(self, constraints):
        return self._visitor.transform(self._parser.parse(constraints))

    def parse_from_file(self, file) -> list:
        out = []
        try:
            with open(file, 'r') as reader:
                try:
                    constraints = yaml.safe_load(reader)
                    if not isinstance(constraints, dict):
                        return out
                    for constraint in constraints.values():
                        try:
                            out.extend(self.parse(constraint))
                        except:
                            print(f"Unknown constraint {constraint} from {file}")
                except:
                    print(SyntaxError(file))
        except FileNotFoundError:
            pass

        return out


if __name__ == '__main__':
    constraint_parser = ConstraintParser()

    file = '595.yml'
    out = constraint_parser.parse_from_file(f"../benchmarks/leetcode/constraints/{file}")
    print(out)

    # constraint = "Employees.manager_id <- Employees.employee_id; Employees.manager_id != NULL"
    # out = parser.parse(constraint)

    # import os
    # files = os.listdir('../data/constraints')
    # for file in files:
    #     print(file)
    #     out = constraint_parser.parse_from_file(f"../data/constraints/{file}")
    #     # debug = Debug()
    #     # debug.check(file, out)
    #     print(out)
