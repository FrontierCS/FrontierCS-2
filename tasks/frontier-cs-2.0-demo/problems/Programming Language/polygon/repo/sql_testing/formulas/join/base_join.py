import itertools
import math
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import List

from sql_testing.ast.expressions.attribute import Attribute
from sql_testing.ast.expressions.expression import Expression
from sql_testing.schemas import TableSchema


class FJoin(ABC):
    def __init__(self,
                 left_input_table: TableSchema,
                 right_input_table: TableSchema,
                 node,
                 env,
                 k=2,
                 ):
        self.left = left_input_table
        self.right = right_input_table
        self.node = node
        self.condition = node.condition
        self.env = env
        self.under_vector_size = k
        self.mapping = {}  # maps new column id to the column id in the input

        if node.using is not None:
            self.condition = Expression(operator='eq', args=[
                Attribute(name=f'{self.left.table_name}.{node.using.value}'),
                Attribute(name=f'{self.right.table_name}.{node.using.value}')
            ])

        # precise output
        output = self.create_output_table(self.left, self.right)
        output.node = node
        self.output = output

        if self.under_vector_size < output.bound:
            # under output
            self.approximated_output = deepcopy(output)
            self.approximated_output.table_id = self.env.next_table_id()
            self.approximated_output.bound = self.under_vector_size
            self.env.db.add_table(self.approximated_output)
            self.env.formulas.under_table_id_to_original[self.approximated_output.table_id] = output.table_id
            self.output = self.approximated_output
        else:
            self.approximated_output = None

        self.semantics(self.left, self.right, output)

    def create_output_table(self, left_table: TableSchema, right_table: TableSchema) -> TableSchema:
        left_table_id, left_table_name = left_table.get_info()
        right_table_id, right_table_name = right_table.get_info()

        output_table_id = self.env.next_table_id()
        match self.node.join_type:
            case 'join' | 'inner join':
                precise_bound = self.left.bound * self.right.bound
            case 'left join' | 'left outer join':
                precise_bound = self.left.bound * self.right.bound + self.left.bound
            case 'right join' | 'right outer join':
                precise_bound = self.left.bound * self.right.bound + self.right.bound
            case 'full join' | 'full outer join':
                precise_bound = self.left.bound * self.right.bound + self.left.bound + self.right.bound
            case 'cross join':
                precise_bound = self.left.bound * self.right.bound
            case _:
                raise NotImplementedError(f"Join type {self.node.join_type} not supported")

        new_table = TableSchema(
            output_table_id,
            f"!{left_table_name}_JOIN_{right_table_name}!",
            precise_bound
        )

        column_id = 0
        for column in left_table:
            output_column = deepcopy(column)
            output_column.column_id = column_id
            new_table.append(output_column)
            self.mapping[column_id] = (left_table.table_id, column.column_id)
            column_id += 1
        for column in right_table:
            output_column = deepcopy(column)
            output_column.column_id = column_id
            new_table.append(output_column)
            self.mapping[column_id] = (right_table.table_id, column.column_id)
            column_id += 1

        new_table.lineage = f"Joined from T{left_table_id} and T{right_table_id} on ({self.condition})"
        new_table.ancestors.extend([left_table, right_table])
        self.env.db.add_table(new_table)

        return new_table

    @abstractmethod
    def semantics(self, left_table, right_table, output):
        ...

    @staticmethod
    def _gen_size(input_schemas: List[TableSchema], output_schema: TableSchema):
        combs = (list(range(input_schema.bound + 1)) for input_schema in input_schemas)
        combs = itertools.product(*combs)

        outputs = []
        for comb in combs:
            comb = list(comb)
            if math.prod(comb) <= output_schema.bound:
                outputs.append(comb)
        return outputs
