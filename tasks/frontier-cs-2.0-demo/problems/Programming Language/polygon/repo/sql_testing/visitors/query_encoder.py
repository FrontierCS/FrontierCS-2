from typing import Tuple

from sql_testing.ast.expressions.literal import Literal
from sql_testing.ast.filter import Filter
from sql_testing.ast.group_by import GroupBy
from sql_testing.ast.join import Join
from sql_testing.ast.node import MutatedNode
from sql_testing.ast.order_by import OrderBy
from sql_testing.ast.project import Project
from sql_testing.ast.query import Query
from sql_testing.ast.scan import Scan
from sql_testing.formulas.distinct import FDistinct
from sql_testing.formulas.filter import FFilter
from sql_testing.formulas.group_by import FGroupBy
from sql_testing.formulas.join import *
from sql_testing.formulas.mutation import FMutation
from sql_testing.formulas.order_by import FOrderBy
from sql_testing.formulas.project import FProject
from sql_testing.formulas.scan import FScan
from sql_testing.formulas.union import FUnion
from sql_testing.schemas import TableSchema
from sql_testing.smt.ast import Bool
from sql_testing.utils import copy_group_by_mapping
from sql_testing.visitors.initializer import Initializer


class QueryEncoder:
    def __init__(self, env, outer_tuple_id=None):
        self.env = env
        self.output_table = None
        self.query_initialized = False
        self.outer_tuple_id = outer_tuple_id

    def visit_Query(self, node: Query) -> Tuple[TableSchema, TableSchema]:
        if not node.initialized:
            initializer = Initializer(self.env)
            node.accept(initializer)
            node.initialized = True

        # WITH clause
        if node.cte:
            for cte_name, cte_query in node.cte.items():
                table = cte_query.accept(self)
                table.table_name = cte_name
                for column in table:
                    column.table_name = cte_name

        # Query order of execution:

        # 1. FROM
        self.output_table = node.from_clause.accept(self)

        # 2. WHERE
        if node.where_clause is not None:
            self.output_table = node.where_clause.accept(self)

        # 3. GROUP BY / HAVING
        if node.group_by_clause is not None:
            node.group_by_clause.ctx['select_list'] = node.select_clause.target_list
            self.output_table = node.group_by_clause.accept(self)

        # 4. SELECT
        self.output_table = node.select_clause.accept(self)

        # 5. ORDER BY
        # if node.order_by_clause is not None:
        #     self.output_table = node.order_by_clause.accept(self)

        # alias
        if node.alias is not None:
            self.output_table = self.output_table.as_aliased(node.alias, self.env)

        return self.output_table

    def visit_Union(self, node) -> Tuple[TableSchema, TableSchema]:
        if not self.env.initialized and not node.initialized:
            initializer = Initializer(self.env)
            node.accept(initializer)
            node.initialized = True

        output_tables = []
        for query in node.queries:
            subquery_visitor = QueryEncoder(self.env)
            output_table = query.accept(subquery_visitor)
            output_tables.append(output_table)

        k = self.env.formulas.under_config[node.label]

        f = FUnion(
            output_tables,
            node,
            self.env,
            k
        )

        self.output_table = f.output

        # alias
        if node.alias is not None:
            self.output_table = self.output_table.as_aliased(node.alias, self.env)

        return self.output_table

    def visit_Scan(self, node: Scan) -> Tuple[TableSchema, TableSchema]:
        output_table = FScan(node.table, self.env).output_table

        # alias
        if node.alias is not None:
            output_table = output_table.as_aliased(node.alias, self.env)

        return output_table

    def visit_Join(self, node: Join) -> Tuple[TableSchema, TableSchema]:
        left = node.left.accept(self)
        right = node.right.accept(self)

        match node.join_type:
            case 'join' | 'inner join':
                formula_name = 'FInnerJoin'
            case 'left join' | 'left outer join':
                formula_name = 'FLeftJoin'
            case 'right join' | 'right outer join':
                # formula_name = 'FRightJoin'
                left, right = right, left
                formula_name = 'FLeftJoin'
            case 'full join' | 'full outer join':
                formula_name = 'FFullJoin'
            case 'cross join':
                formula_name = 'FProduct'
            case _:
                raise NotImplementedError(f"Join type {node.join_type} not supported")

        under_vector_size = self.env.formulas.under_config[node.label]

        if node.condition is None:
            node.condition = Literal(True)

        f = globals()[formula_name](
            left, right,
            node,
            self.env,
            under_vector_size
        )
        return f.output

    def visit_GroupBy(self, node: GroupBy) -> Tuple[TableSchema, TableSchema]:
        # k, branches = self.env.formulas.under_config[node.label]

        f = FGroupBy(self.output_table, node, self.env, self.output_table.bound)
        f.output.ctx['groups_considered'] = [self.output_table.bound]
        return f.output

    def visit_Filter(self, node: Filter) -> Tuple[TableSchema, TableSchema]:
        k = self.env.formulas.under_config[node.label]
        f = FFilter(self.output_table, node, self.env, k, self.outer_tuple_id)
        return f.output

    def visit_Project(self, node: Project) -> Tuple[TableSchema, TableSchema]:
        k = self.env.formulas.under_config[node.label]
        f = FProject(self.output_table, node, self.env, k)
        if node.distinct:
            f = FDistinct(f.output, node, self.env)
        return f.output

    def visit_OrderBy(self, node: OrderBy) -> Tuple[TableSchema, TableSchema]:
        k = self.env.formulas.under_config[node.label]

        f = FOrderBy(self.output_table, node, self.env, k)
        return f.output
