import datetime
import functools
import re
import time
from typing import (
    Dict,
    Union,
)

from mo_parsing import ParseException
from mo_sql_parsing import parse

from sql_testing.ast.expressions.case_when import CaseWhen
from sql_testing.ast.group_by import GroupBy
from sql_testing.ast.order_by import OrderBy
from sql_testing.errors import (
    ParserSyntaxError,
)
from sql_testing.ast.expressions.attribute import Attribute
from sql_testing.ast.expressions.expression import Expression
from sql_testing.ast.expressions.function import FunctionCall
from sql_testing.ast.expressions.literal import Literal
from sql_testing.ast.filter import Filter
from sql_testing.ast.join import Join
from sql_testing.ast.query import Query
from sql_testing.ast.project import Project
from sql_testing.ast.scan import Scan
from sql_testing.ast.union import Union as UnionAST

SQL_NULL = {"null": None}
SPACE_STRING = "__SPACE_STRING__"
TMP_PLACEHOLDER = "TEMPORARY_PLACEHOLDER"
NULL_REGEX = r'([a-zA-Z_\.\$]+\s+(<>|=|!=)\s+NULL)|(NULL\s+(<>|=|!=)\s+[a-zA-Z_\.\$]+)'
FROM_REGEX = r'FROM \([a-zA-Z0-9]+\)'
VALUES_REGEX = r'\(VALUES\s*\(\S+\s*\S*(\s*,\s*\S+\s*\S*)*\)(\s*,\s*\(\S+\s*\S*(\s*,\s*\S+\s*\S*)*\))*\)'
LIMIT_REGEX = r'LIMIT\s+\d+,\s*\d+'


class ValuesTable:
    def __init__(self, name, rows, attributes=['X', 'Y', 'Z']):
        # why set X, Y, Z cuz for calcite dataset
        # only works for VALUES (XXX) in SQL query parsing
        self.name = name
        self.rows = rows
        self.attributes = attributes[:len(rows[0])]

    def __str__(self):
        return f'{self.name}=({self.attributes}, #{len(self.rows)}: {[[list(col.values())[0] for col in row] for row in self.rows]})'

    def __repr__(self):
        return self.__str__()


def is_date_format(date: str):
    return re.match(r'^[0-9]{2,4}[-|_|:|/][0-9]{1,2}[-|_|:|/][0-9]{1,2}(\s+[0-9]{1,2}:[0-9]{1,2}:[0-9]{1,2})?$',
                    date.strip()) is not None


def is_time_format(time: str):
    try:
        datetime.datetime.strptime(time.split('.')[0], '%H:%M:%S')
        return True
    except ValueError:
        return False


def is_VALUES(value):
    # {'value': 'VALUES', 'name': XX}
    return isinstance(value, Dict) and value.get('value', False) == 'VALUES' and value.get('name', False)


def is_NULL(value):
    return value == {'value': TMP_PLACEHOLDER}


def parse_from_values(values, VALUE_TABLES):
    # FROM (VALUES (XX)) AS T
    if isinstance(values['name'], Dict):
        table_name = list(values['name'])[0]
        columns = values['name'][table_name]
        if isinstance(columns, str):
            columns = [columns]
        lines = []
        for row in VALUE_TABLES[0]:
            # use parse to help us to identify value types
            parsed_query = parse(
                f"INSERT INTO {table_name}({', '.join(columns)}) VALUES {row}",
                null=SQL_NULL,
            )
            if isinstance(parsed_query['query']['select'], Dict):
                lines.append([parsed_query['query']['select']])
            else:
                lines.append(parsed_query['query']['select'])
        values_table = ValuesTable(name=table_name, rows=lines, attributes=columns)
    else:
        table_name = values['name']
        lines = []
        for row in VALUE_TABLES[0]:
            # use parse to help us to identify value types
            parsed_query = parse(f"INSERT INTO {table_name} VALUES {row}",
                                 null=SQL_NULL)
            if isinstance(parsed_query['query']['select'], Dict):
                lines.append([parsed_query['query']['select']])
            else:
                lines.append(parsed_query['query']['select'])
        values_table = ValuesTable(name=table_name, rows=lines)
    del VALUE_TABLES[0]
    return values_table


class SQLParser:
    def __init__(self):
        pass

    def parse(self, query: str):
        # preprocessing

        # SELECT ALL -> SELECT
        query = query.replace('SELECT ALL ', 'SELECT ')

        # for uns_key in ['EXISTS', 'GROUPING', 'SUBSTRING', 'LATERAL', 'EXTRACT', 'ROLLUP', 'EXTRACT', 'LIKE', 'TRIM']:
        #     if uns_key in query:
        #         raise NotSupportedError(f"`{uns_key}` is not supported.")

        def _remove_dots(obj):
            if isinstance(obj, dict):
                if len(obj) == 1 and 'literal' in obj and isinstance(obj['literal'], str):
                    if is_time_format(obj['literal']):
                        return {'time': obj['literal'].split('.')[0]}
                    elif is_date_format(obj['literal']):
                        return {'date': obj['literal']}

                for k, v in obj.items():
                    obj[k] = _remove_dots(v)
                    if k == 'literal':
                        obj[k] = v
            elif isinstance(obj, list):
                for i, v in enumerate(obj):
                    obj[i] = _remove_dots(v)
            # elif isinstance(obj, str):
            #     return obj.replace('.', '__')
            # {'literal': 1} -> 1
            # {'literal': [1,2,3]} -> [1,2,3]
            if isinstance(obj, dict) and len(obj) == 1 and 'literal' in obj:
                if isinstance(obj['literal'], Union[int, bool, float]) or \
                        (
                                isinstance(obj['literal'], list) and \
                                all(isinstance(o, Union[int, bool, float]) for o in obj['literal'])
                        ):
                    obj = obj['literal']
                # elif isinstance(obj['literal'], str) and re.match('String_\d+_DAY', obj['literal']):
                #     obj = int(re.findall('\d+', obj['literal'])[0])
                # elif isinstance(obj['literal'], str) and re.match('String_\d+_(YEAR|MONTH|HOUR|MINUTE|SECOND)', v):
                #     raise NotImplementedError("We only support \d+ day.")
            return obj

        try:
            if re.search(VALUES_REGEX, query) is not None:
                sub_queries, VALUE_TABLES = [], []
                re_outs = re.search(VALUES_REGEX, query)
                while re_outs is not None:
                    sub_queries.append(query[:re_outs.span()[0]])
                    VALUE_TABLES.append(query[re_outs.span()[0]:re_outs.span()[-1]])
                    query = query[re_outs.span()[-1]:]
                    re_outs = re.search(VALUES_REGEX, query)
                sub_queries.append(query)

                query = ' VALUES '.join(sub_queries)
                VALUE_TABLES = [
                    re.findall(r'\(.+?\)', values[1:-1].strip()[6:].strip())
                    for values in VALUE_TABLES
                ]
                parsed_query = parse(query, null=SQL_NULL)

                def _f(query):
                    if isinstance(query, list):
                        for idx, subquery in enumerate(query):
                            query[idx] = _f(subquery)
                    elif isinstance(query, Dict):
                        if is_VALUES(query):
                            return parse_from_values(query, VALUE_TABLES)
                        else:
                            for k, v in query.items():
                                query[k] = _f(v)
                    return query

                parsed_query = _f(parsed_query)
            elif 'TIMESTAMP' in query:
                # mo_sql_parsing cannot parse TIMESTAMP(*)
                TIMESTAMP_INDICES = [int(token[10:-1]) for token in re.findall(r'TIMESTAMP\(\d+\)', query)]
                query = re.sub(r'TIMESTAMP\(\d+\)', 'TIMESTAMP', query)
                parsed_query = parse(query, null=SQL_NULL)

                def _f(query):
                    if isinstance(query, list):
                        for subquery in query:
                            _f(subquery)
                    elif isinstance(query, Dict):
                        for k, v in query.items():
                            if k == 'timestamp':
                                query[k] = TIMESTAMP_INDICES[0]
                                del TIMESTAMP_INDICES[0]
                            else:
                                _f(v)

                _f(parsed_query)
            # elif re.search(IS_TRUE_FALSE_REGEX, query) is not None:
            #     sub_queries, IS_TRUE_FALSE_CLAUSES = [], []
            #     re_outs = re.search(IS_TRUE_FALSE_REGEX, query)
            #     while re_outs is not None:
            #         sub_queries.append(query[:re_outs.span()[0]])
            #         clause = query[re_outs.span()[0]:re_outs.span()[-1]]
            #         sub_queries.append(f'IS {TMP_PLACEHOLDER}')
            #         query = query[re_outs.span()[-1]:]
            #         re_outs = re.search(IS_TRUE_FALSE_REGEX, query)
            #     sub_queries.append(query)
            #     parsed_query = parse(''.join(sub_queries), null=SQL_NULL)
            elif re.search(LIMIT_REGEX, query) is not None:
                sub_queries, LIMIT_CLAUSES = [], []
                re_outs = re.search(LIMIT_REGEX, query)
                while re_outs is not None:
                    sub_queries.append(query[:re_outs.span()[0]])
                    clause = query[re_outs.span()[0]:re_outs.span()[-1]]
                    index = clause.find(',')
                    LIMIT_CLAUSES.append(int(clause[index + 1:]))
                    sub_queries.append(clause[:index])
                    query = query[re_outs.span()[-1]:]
                    re_outs = re.search(LIMIT_REGEX, query)
                sub_queries.append(query)
                parsed_query = parse(''.join(sub_queries), null=SQL_NULL)

                def _f(query):
                    if isinstance(query, list):
                        for idx, subquery in enumerate(query):
                            query[idx] = _f(subquery)
                    elif isinstance(query, Dict):
                        for k, v in query.items():
                            if k == 'limit' and isinstance(v, int):
                                index = LIMIT_CLAUSES.pop(0)
                                query[k] = [v, index]
                            else:
                                query[k] = _f(v)
                    return query

                parsed_query = _f(parsed_query)

            elif len(re.findall(FROM_REGEX, query)) > 0:
                from_clauses_with_parenthesis = re.findall(FROM_REGEX, query)
                for clause in from_clauses_with_parenthesis:
                    new_clause = clause.replace('(', '').replace(')', '')
                    query = query.replace(clause, new_clause, 1)
                parsed_query = parse(query, null=SQL_NULL)
            elif re.findall(NULL_REGEX, query) is not None:
                query = query.replace('!=', '<>')
                query = query.replace('||', ' OR ')
                operations = [
                    match
                    for matches in re.findall(NULL_REGEX, query)
                    for match in matches
                    if len(match) > 2  # opd1 op opd2
                ]
                for operation in operations:
                    idx = str.find(query, operation)
                    query = query[:idx] + TMP_PLACEHOLDER + query[idx + len(operation):]
                for idx, operation in enumerate(operations):
                    if '<>' in operation:
                        operands = [opd.strip() for opd in operation.split('<>')]
                        operator = 'neq'
                    else:
                        operands = [opd.strip() for opd in operation.split('=')]
                        operator = 'eq'
                    if operands[0] == 'NULL':
                        null_idx, value_idx = 0, 1
                    else:
                        null_idx, value_idx = 1, 0
                    operands[null_idx] = SQL_NULL
                    if len(operands[value_idx]) > 2 and \
                            (operands[value_idx][0] == operands[value_idx][-1] == '\'' or \
                             operands[value_idx][0] == operands[value_idx][-1] == '\"'):
                        operands[value_idx] = operands[value_idx][1:-1]
                    try:
                        value = float(operands[value_idx])
                        if value == int(operands[value_idx]):
                            value = int(operands[value_idx])
                        operands[value_idx] = {'value': value}
                    except:
                        pass
                    operations[idx] = {operator: operands}
                parsed_query = parse(query, null=SQL_NULL)

                def _f(query):
                    if isinstance(query, list):
                        for idx, subquery in enumerate(query):
                            query[idx] = _f(subquery)
                    elif isinstance(query, Dict):
                        for k, v in query.items():
                            query[k] = _f(v)
                    elif query == TMP_PLACEHOLDER:
                        return operations.pop(0)
                    return query

                parsed_query = _f(parsed_query)
            else:
                parsed_query = parse(query, null=SQL_NULL)
            _remove_dots(parsed_query)
            return parsed_query
        except ParseException as err:
            raise ParserSyntaxError(query)

    def parse_query(self, query: dict) -> Query | UnionAST:
        alias = None
        # from clause is a subquery with renaming
        if 'value' in query and 'name' in query:
            alias = query['name']
            query = query['value']

        with_clause = None
        if 'with' in query:
            with_clause = {}
            cte_list = query['with']
            if not isinstance(cte_list, list):
                cte_list = [cte_list]
            for cte in cte_list:
                cte_name = cte['name']
                cte_query = self.parse_query(cte['value'])
                with_clause[cte_name] = cte_query

        if 'select' in query:
            select_clause = self.parse_select(query['select'])
        elif 'select_distinct' in query:
            select_clause = self.parse_select(query['select_distinct'])
            select_clause.distinct = True
        elif 'union' in query:
            queries = [self.parse_query(q) for q in query['union']]
            return UnionAST(queries, alias=alias)
        elif 'union_all' in query:
            queries = [self.parse_query(q) for q in query['union_all']]
            return UnionAST(queries, allow_duplicates=True, alias=alias)
        else:
            raise NotImplementedError
        from_clause = self.parse_from(query['from'])

        where_clause = None
        if 'where' in query:
            where_clause = self.parse_filter(query['where'])
        # use of HAVING clause without a GROUP BY clause -> filter
        if 'having' in query and 'groupby' not in query:
            if where_clause is None:
                where_clause = self.parse_filter(query['having'])
            else:
                second_where_clause = self.parse_filter(query['having'])
                where_clause.predicate = Expression(operator='and', args=[where_clause.predicate, second_where_clause.predicate])

        group_by_clause = None
        if 'groupby' in query:
            group_by_clause = self.parse_group_by(query['groupby'])
            if 'having' in query:
                group_by_clause.having = self.parse_expression(query['having'])

        order_by_clause = None
        if 'orderby' in query:
            order_by_clause = self.parse_order_by(query['orderby'])
        if 'limit' in query:
            order_by_clause.limit = query['limit']

        ast = Query(select_clause=select_clause,
                    from_clause=from_clause,
                    where_clause=where_clause,
                    group_by_clause=group_by_clause,
                    order_by_clause=order_by_clause,
                    alias=alias,
                    cte=with_clause
                    )
        return ast

    def parse_select(self, select_clause: list | dict | str) -> Project:
        target_list = []
        if isinstance(select_clause, list):
            for attr in select_clause:
                if 'value' in attr:
                    alias = None
                    if 'name' in attr:
                        alias = attr['name']

                    if isinstance(attr['value'], bool | int):
                        target_list.append(Literal(attr['value'], alias=alias))
                    elif isinstance(attr['value'], dict):
                        # select target is an expression
                        exp = self.parse_expression(attr['value'])
                        exp.alias = alias
                        target_list.append(exp)
                    else:
                        # select target is an attribute
                        target_list.append(Attribute(attr['value'], alias=alias))
                else:
                    target_list.append(Attribute(attr))
        elif isinstance(select_clause, str):
            target_list.append(Attribute(select_clause))
        elif isinstance(select_clause, dict):
            if 'value' in select_clause:
                alias = None
                if 'name' in select_clause:
                    alias = select_clause['name']

                if isinstance(select_clause['value'], bool | int):
                    target_list.append(Literal(select_clause['value'], alias=alias))
                elif isinstance(select_clause['value'], dict):
                    # select target is an expression
                    exp = self.parse_expression(select_clause['value'])
                    exp.alias = alias
                    target_list.append(exp)
                else:
                    # select target is an attribute
                    target_list.append(Attribute(select_clause['value'], alias=alias))
        else:
            raise ParserSyntaxError(select_clause)
        return Project(target_list)

    def parse_from(self, from_clause: str | dict) -> Scan | Query | Join:
        if isinstance(from_clause, str):
            return Scan(table=from_clause)
        elif 'value' in from_clause and 'name' in from_clause and isinstance(from_clause['value'], str):
            return Scan(table=from_clause['value'], alias=from_clause['name'])
        # JOIN
        elif isinstance(from_clause, list):
            left = from_clause[0]
            left = self.parse_from(left)

            for table in from_clause[1:]:
                right = table
                join_type = 'cross join'
                if isinstance(right, dict):
                    if 'join' in next(iter(table.items()))[0]:
                        join_type = next(iter(table.items()))[0]
                        right = next(iter(table.items()))[1]
                    right = self.parse_from(right)
                else:
                    right = Scan(table=right)

                condition = None
                using = None
                if isinstance(table, dict):
                    if 'on' in table:
                        condition = self.parse_expression(table['on'])
                    elif 'using' in table:
                        using = self.parse_expression(table['using'])

                left = Join(left=left, right=right, join_type=join_type, condition=condition, using=using)

            return left
        # subquery
        else:
            return self.parse_query(from_clause)

    def parse_filter(self, predicate: dict) -> Filter:
        return Filter(predicate=self.parse_expression(predicate))

    def parse_group_by(self, raw_expressions: list | dict) -> GroupBy:
        if not isinstance(raw_expressions, list):
            raw_expressions = [raw_expressions]

        group_by_expressions = []
        for exp in raw_expressions:
            if isinstance(exp['value'], str):
                group_by_expressions.append(Attribute(exp['value']))
            elif isinstance(exp['value'], int):
                group_by_expressions.append(Literal(exp['value']))
            else:
                group_by_expressions.append(self.parse_expression(exp['value']))
        return GroupBy(group_by_expressions)

    def parse_order_by(self, raw_expressions: dict | list) -> OrderBy:
        if not isinstance(raw_expressions, list):
            raw_expressions = [raw_expressions]

        order_by_expressions = []
        sort_orders = []
        for exp in raw_expressions:
            if isinstance(exp['value'], str):
                order_by_expressions.append(Attribute(exp['value']))
            elif isinstance(exp['value'], int):
                order_by_expressions.append(Literal(exp['value']))
            else:
                order_by_expressions.append(self.parse_expression(exp['value']))
            if 'sort' in exp:
                sort_orders.append(exp['sort'])
            else:
                sort_orders.append('asc')
        return OrderBy(order_by_expressions, sort_orders)

    def parse_expression(self, exp) -> Expression | Literal:
        if isinstance(exp, int | bool | str | float):
            return Literal(exp)

        # special case: DISTINCT aggregate function call
        # TODO: refactor
        if len(exp) == 2 and 'distinct' in exp:
            distinct = exp['distinct']
            agg_fun = list(exp.keys())[1]
            arg = exp[agg_fun]
            if isinstance(arg, dict):
                arg = self.parse_expression(arg)
            else:
                arg = Attribute(arg)
            return Expression(operator=agg_fun, args=[distinct, arg])

        operator = next(iter(exp))

        # cast as type: {'cast': [{'date': '2020-05-03'}, {'date': {}}]}
        if exp[operator] == {}:
            return Literal(operator)

        # special case: the dict encapsulates just a string literal
        if operator == 'literal':
            # value set of IN
            if isinstance(exp['literal'], list):
                return [self.parse_expression(val) for val in exp['literal']]
            return self.parse_expression(exp['literal'])
        elif operator == 'date':
            if 'date' in exp['date']:
                return Literal(datetime.datetime.fromisoformat(exp['date']['date']).date())
            if exp['date'] == {}:
                return None
            return Literal(datetime.datetime.fromisoformat(exp['date']).date())
        elif operator == 'time':
            if 'time' in exp['time']:
                return Literal(time.strptime(exp['time']['time'], '%H:%M:%S'))
            return Literal(time.strptime(exp['time'], '%H:%M:%S'))

        # expression is a CASE WHEN
        if operator == 'case':
            return self.parse_case_when(exp[operator])

        args = []
        if isinstance(exp[operator], list):
            # a list of arguments
            for operand in exp[operator]:
                # arg is a literal
                if isinstance(operand, int | float | bool):
                    args.append(Literal(operand))
                # arg is an attribute
                elif isinstance(operand, str):
                    args.append(Attribute(operand))
                # arg is an expression (or string literal)
                elif isinstance(operand, dict):
                    if 'select' in operand or 'select_distinct' in operand:
                        args.append(self.parse_query(operand))
                    else:
                        args.append(self.parse_expression(operand))
                elif isinstance(operand, list):
                    arg = []
                    for op in operand:
                        if isinstance(op, dict):
                            if 'select' in op or 'select_distinct' in op:
                                arg.append(self.parse_query(op))
                            else:
                                arg.append(self.parse_expression(op))
                        else:
                            arg.append(Attribute(op))
                    args.append(arg)
                else:
                    raise NotImplementedError
        # single argument
        elif isinstance(exp[operator], dict):
            # arg is an expression (or string literal)
            if 'select' in exp[operator] or 'select_distinct' in exp[operator]:
                args.append(self.parse_query(exp[operator]))
            else:
                args.append(self.parse_expression(exp[operator]))
        elif isinstance(exp[operator], int | bool):
            # arg is a literal
            args.append(Literal(exp[operator]))
        elif isinstance(exp[operator], str):
            # arg is an attribute
            args.append(Attribute(exp[operator]))
        elif 'null' in exp:
            return Literal(None)
        else:
            raise NotImplementedError(exp)

        if operator in ['max', 'min', 'sum', 'avg', 'count']:
            args = [False, *args]

        if operator in ['add']:
            return functools.reduce(lambda x, y: Expression(operator, [x, y]), args)
        return Expression(operator, args)

    def parse_case_when(self, exp) -> CaseWhen:
        cases = []
        default = None
        if isinstance(exp, dict):
            exp = [exp]
        if not isinstance(exp[-1], dict) or ('when' not in exp[-1] and 'then' not in exp[-1]):
            if isinstance(exp[-1], str):
                default = Attribute(exp[-1])
            else:
                default = self.parse_expression(exp[-1])
            exp = exp[:-1]

        for case in exp:
            if isinstance(case['when'], str):
                when = Attribute(case['when'])
            else:
                when = self.parse_expression(case['when'])
            if isinstance(case['then'], str):
                then = Attribute(case['then'])
            else:
                then = self.parse_expression(case['then'])
            cases.append((when, then))
        return CaseWhen(cases, default)

    def table_names_used(self, ast):
        def extract_table_name(ast_json):
            if isinstance(ast_json, list):
                for table in ast_json:
                    for i in extract_table_name(table):
                        yield from extract_table_name(i)
                    #     pass
                    #     # print (i)
                    # yield from extract_table_name(table)
            elif isinstance(ast_json, dict):
                if 'value' in ast_json:
                    yield ast_json['value']
                elif 'inner join' in ast_json:
                    yield from extract_table_name(ast_json['inner join'])
                elif 'join' in ast_json:
                    yield from extract_table_name(ast_json['join'])
                elif 'left join' in ast_json:
                    yield from extract_table_name(ast_json['left join'])
                elif 'right join' in ast_json:
                    yield from extract_table_name(ast_json['right join'])
                elif 'full join' in ast_json:
                    yield from extract_table_name(ast_json['full join'])
                elif 'from' in ast_json:
                    yield from extract_table_name(ast_json['from'])
                else:
                    raise NotImplementedError(ast_json)
            else:
                yield ast_json

        for k, v in ast.items():
            if k == "from":
                yield from extract_table_name(v)
            elif isinstance(v, dict):
                for id_val in self.table_names_used(v):
                    yield id_val

    def __str__(self):
        return self.__class__.__name__

    def __repr__(self):
        return self.__str__()


if __name__ == '__main__':
    parser = SQLParser()
    query = "SELECT BUYER_ID FROM SALES ORDER BY a, B DESC LIMIT 1"

    q_json = parser.parse(query)
    print(q_json)

    q_ast = parser.parse_query(q_json)
    print(repr(q_ast))
    print(q_ast)
