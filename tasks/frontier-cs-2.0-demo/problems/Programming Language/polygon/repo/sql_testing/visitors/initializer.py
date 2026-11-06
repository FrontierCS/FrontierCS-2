from sql_testing.ast.expressions.expression import Expression
from sql_testing.ast.filter import Filter
from sql_testing.ast.group_by import GroupBy
from sql_testing.ast.join import Join
from sql_testing.ast.node import MutatedNode
from sql_testing.ast.order_by import OrderBy
from sql_testing.ast.project import Project
from sql_testing.ast.query import Query
from sql_testing.ast.scan import Scan


class Initializer:
    def __init__(self, env):
        self.env = env
        self.default_k = env.default_k

    def visit_Query(self, node: Query):
        # WITH clause
        if node.cte:
            for cte_name, cte_query in node.cte.items():
                cte_query.accept(self)

        # Query order of execution:

        # 1. FROM
        nodes = node.from_clause.accept(self)

        # 2. WHERE
        if node.where_clause is not None:
            node.where_clause.set_parent(nodes)
            nodes = node.where_clause.accept(self)

        # 3. GROUP BY / HAVING
        if node.group_by_clause is not None:
            node.group_by_clause.set_parent(nodes)
            nodes = node.group_by_clause.accept(self)

        # 4. SELECT
        node.select_clause.set_parent(nodes)
        nodes = node.select_clause.accept(self)

        # 5. ORDER BY
        if node.order_by_clause is not None:
            node.order_by_clause.set_parent(nodes)
            nodes = node.order_by_clause.accept(self)

        return nodes

    def visit_Union(self, node):
        for query in node.queries:
            nodes = query.accept(self)
            node.parent.extend(nodes)

        # register node in formula manager
        label_id = self.env.formulas.next_node_label()
        label = f'union${label_id}'
        node.label = label
        self.env.formulas.label_to_node[label] = node

        self.env.formulas.under_config[label] = self.default_k['union']

        if not node.allow_duplicates:
            label = f'distinct${label_id}'
            node.distinct_label = label
            self.env.formulas.label_to_node[label] = node

        return [node]

    def visit_Scan(self, node: Scan):
        node.label = f'size_{node.table.lower()}'
        return [node]

    def visit_Join(self, node: Join):
        left = node.left.accept(self)
        right = node.right.accept(self)
        node.set_parent([*left, *right])

        match node.join_type:
            case 'join' | 'inner join':
                label = f'inner_join${self.env.formulas.next_node_label()}'
                k = self.default_k['inner join']
            case 'left join' | 'left outer join':
                label = f'left_join${self.env.formulas.next_node_label()}'
                # k = 16
                k = self.default_k['left join']
            case 'right join' | 'right outer join':
                label = f'right_join${self.env.formulas.next_node_label()}'
                k = self.default_k['right join']
            case 'full join' | 'full outer join':
                label = f'full_join${self.env.formulas.next_node_label()}'
                k = self.default_k['full join']
            case 'cross join':
                label = f'product${self.env.formulas.next_node_label()}'
                # k = 16
                k = self.default_k['product']
            case _:
                raise NotImplementedError(f"Join type {node.join_type} not supported")

        node.label = label
        self.env.formulas.label_to_node[label] = node
        self.env.formulas.under_config[label] = k

        return [node]

    def visit_GroupBy(self, node: GroupBy):
        k = (4, 10)

        # register node in formula manager
        label = f'group_by${self.env.formulas.next_node_label()}'
        node.label = label
        self.env.formulas.label_to_node[label] = node
        self.env.formulas.under_config[label] = k

        return [node]

    def visit_Filter(self, node: Filter):
        # if isinstance(node.predicate, Expression):
        #     if node.predicate.operator in ['and', 'or']:
        #         for exp in node.predicate.args:
        #             if isinstance(exp, Expression):
        #                 if exp.operator in ['in', 'nin'] and isinstance(exp.args[1], Query):
        #                     subquery_visitor = Initializer(self.env)
        #                     subquery_visitor.cur_label = self.cur_label
        #
        #                     nodes = exp.args[1].accept(subquery_visitor)
        #                     node.parent.extend(nodes)
        #
        #                     self.cur_label = subquery_visitor.cur_label
        #                 elif exp.operator in ['is_null', 'is_not_null'] and isinstance(exp.args[0], Query):
        #                     subquery_visitor = Initializer(self.env)
        #                     subquery_visitor.cur_label = self.cur_label
        #
        #                     nodes = exp.args[1].accept(subquery_visitor)
        #                     node.parent.extend(nodes)
        #
        #                     self.cur_label = subquery_visitor.cur_label
        #     elif node.predicate.operator in ['in', 'nin'] and isinstance(node.predicate.args[1], Query):
        #         subquery_visitor = Initializer(self.env)
        #         subquery_visitor.cur_label = self.cur_label
        #
        #         nodes = node.predicate.args[1].accept(subquery_visitor)
        #         node.parent.extend(nodes)
        #
        #         self.cur_label = subquery_visitor.cur_label
        #     elif node.predicate.operator in ['is_null', 'is_not_null'] and isinstance(node.predicate.args[0], Query):
        #         subquery_visitor = Initializer(self.env)
        #         subquery_visitor.cur_label = self.cur_label
        #
        #         nodes = node.predicate.args[0].accept(subquery_visitor)
        #         node.parent.extend(nodes)
        #
        #         self.cur_label = subquery_visitor.cur_label

        # register node in formula manager
        label = f'filter${self.env.formulas.next_node_label()}'
        node.label = label
        self.env.formulas.label_to_node[label] = node
        self.env.formulas.under_config[label] = self.default_k['filter']

        return [node]

    def visit_Project(self, node: Project):
        # register node in formula manager
        label_id = self.env.formulas.next_node_label()
        label = f'project${label_id}'
        node.label = label
        self.env.formulas.label_to_node[label] = node

        # self.env.formulas.under_config[label] = 16
        self.env.formulas.under_config[label] = self.default_k['project']

        if node.distinct:
            label = f'distinct${label_id}'
            node.distinct_label = label
            self.env.formulas.label_to_node[label] = node

        return [node]

    def visit_OrderBy(self, node: OrderBy):
        # register node in formula manager
        label = f'order_by${self.env.formulas.next_node_label()}'
        node.label = label
        self.env.formulas.label_to_node[label] = node
        self.env.formulas.under_config[label] = self.default_k['order by']

        return [node]

    # def visit_MutatedNode(self, node: MutatedNode):
    #     original = node.original.accept(self)
    #     mutant = node.mutant.accept(self)
    #
    #     return [*original, *mutant]
