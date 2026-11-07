from collections import defaultdict

from sql_testing.schemas import TableSchema, ColumnSchema
from sql_testing.smt.ast import SMTCell, SMTNull


class Variable:
    def __init__(self, table_id, row_id, column_id, column_name, column_type):
        self.table_id = table_id
        self.row_id = row_id
        self.column_id = column_id
        self.identity = (table_id, row_id, column_id)
        self.column_name = column_name
        self.type = column_type

        self.VAL = SMTCell(table_id, row_id, column_id)
        self.NULL = SMTNull(table_id, row_id, column_id)

    def __eq__(self, other):
        return self.identity == other.identity

    def __str__(self):
        return f"T{self.table_id}[{self.row_id}].{self.column_id}"

    def accept(self, visitor):
        method_name = f'visit_{self.__class__.__name__}'
        visit = getattr(visitor, method_name)
        return visit(self)

    def __hash__(self):
        return hash(self.identity)


class Size(Variable):
    def __init__(self, table_id):
        super().__init__(table_id, -1, -1, 'SIZE', 'int')

    def __str__(self):
        return f"SIZE(T{self.table_id})"


class Database:
    def __init__(self):
        self.tables = defaultdict(list)
        self.table_size = {}
        self.schemas = {}

    def add_table(self, schema: TableSchema):
        table_id = schema.table_id
        self.table_size[table_id] = Size(table_id)
        for tuple_idx in range(schema.bound):
            columns = [None for _ in range(len(schema))]
            for column in schema:
                column_id, column_name, column_type, _ = column.get_info()
                columns[column_id] = Variable(table_id, tuple_idx, column_id, column_type, column_name)
            if len(columns) > 0:
                self.tables[table_id].append(columns)
        self.schemas[table_id] = schema

    def find_table_by_name(self, name: str, env) -> TableSchema:
        name = name.lower()
        for table in self.schemas:
            if self.schemas[table].table_name.lower() == name.lower():
                if self.schemas[table].scope is None or self.schemas[table].scope == env.curr_query_id:
                    return self.schemas[table]

        raise SyntaxError(f"Table '{name}' does not exist")

    def __getitem__(self, identity):
        table_id, row_id, column_id = identity
        try:
            return self.tables[table_id][row_id][column_id]
        except IndexError:
            raise IndexError(str(self.schemas[table_id]), table_id, row_id, column_id)

    def __str__(self):
        output_str = ""
        for table in self.tables:
            # output_str += f"{str(self.schemas[table])}\n"
            output_str += f"T{table} ({self.schemas[table].table_name}): {self.schemas[table].lineage}\n"
            for row in self.tables[table]:
                output_str += f"{', '.join([str(cell) for cell in row])}\n"
        return output_str
