from copy import deepcopy
from functools import cache
from typing import Optional

from sql_testing.smt.ast import SMTSize, Choice, Deleted


class TableSchema:
    def __init__(self, table_id: int, table_name: str, bound: int, lineage: str = None):
        self.table_id = table_id
        self.table_name = table_name
        self.columns = []
        self.bound = bound
        self.ancestors: [TableSchema] = []
        self.lineage = lineage
        self.ctx = {
            'groups_considered': None
        }
        self.node = None

        self.scope = None  # belongs to which query

        self.SIZE = SMTSize(table_id)

    def __eq__(self, other):
        if not isinstance(other, TableSchema):
            return False

        if 'Renamed from' in self.lineage:
            return self.ancestors == other.ancestors

        if self.table_id == other.table_id and self.table_name == other.table_name and len(self.columns) == len(other.columns):
            for idx, column in enumerate(self.columns):
                if column != other.columns[idx]:
                    return False
            return True
        return False

    def is_same(self, other):
        if self.table_id == other.table_id:
            return True
        return False

    def get_info(self):
        return self.table_id, self.table_name

    def as_aliased(self, alias, env):
        if 'Scan' in self.lineage:
            new_table_id = env.next_table_id()
            aliased_table = deepcopy(self)
            aliased_table.table_id = new_table_id
            aliased_table.table_name = alias
            for column in aliased_table.columns:
                column.table_name = alias
            aliased_table.lineage = self.lineage
            aliased_table.ancestors = self.ancestors
            aliased_table.scope = env.curr_query_id

            env.db.tables[new_table_id] = env.db.tables[self.table_id]
            env.db.schemas[new_table_id] = aliased_table

            for tuple_id in range(self.bound):
                env.formulas.append(Choice(new_table_id, tuple_id) == Choice(self.table_id, tuple_id), label=f'scan_{self.table_name}')
                env.formulas.append(Deleted(new_table_id, tuple_id) == Deleted(self.table_id, tuple_id), label=f'scan_{self.table_name}')

            return aliased_table
        else:
            self.table_name = alias
            self.scope = env.curr_query_id
            for column in self.columns:
                column.table_name = alias
            return self

    # find attribute in table by id/name
    @cache
    def __getitem__(self, item):
        if isinstance(item, int):
            return self.columns[item]
        else:
            item = item.lower()
            if '.' in item:
                name = item.split('.')
                table = name[0]
                attr = name[1]
            else:
                table = None
                attr = item

            for column in self.columns:
                if (table is None or column.table_name is None or table.lower() == column.table_name.lower()) and column.column_name.lower() == attr.lower():
                    return column
                # order by
                if column.name_before_project is not None and column.name_before_project.lower() == item:
                    return column
            raise SyntaxError(f"Attribute '{item}' does not exist in T{self.table_id} ({self.table_name})\n{self}")

    def __setitem__(self, key, value):
        if not isinstance(value, ColumnSchema):
            raise ValueError(f"{value} is not a Column_Schema type.")
        self.columns[key] = value

    def append(self, column):
        self.columns.append(column)

    def __len__(self):
        return len(self.columns)

    def __iter__(self):
        for column in self.columns:
            yield column

    def __str__(self):
        output_str = f"T{self.table_id}({self.table_name}):["
        column_str = []
        for column in self.columns:
            column_str.append(str(column))
        column_str = ",".join(column_str)
        output_str += column_str + "]"
        return output_str + f", size={self.bound}"

    def __hash__(self):
        return hash(self.table_id)


class ColumnSchema:
    def __init__(self,
                 column_id: int,
                 column_name: str,
                 column_type: Optional[str],
                 table_name: Optional[str] = None,
                 name_before_project: Optional[str] = None
                 ):
        self.column_id = column_id
        self.column_name = column_name
        self.column_type = column_type
        self.table_name = table_name
        self.name_before_project = name_before_project
        self.table_aliases = []
        if self.table_name is not None:
            self.table_aliases.append(table_name.lower())

    def __eq__(self, other):
        return self.column_name == other.column_name and \
            self.column_type == other.column_type and \
            self.table_name == other.table_name

    def get_info(self):
        return self.column_id, self.column_name, self.column_type, self.table_name

    def __str__(self):
        return f"c{self.column_id}({self.table_name}.{self.column_name}):{self.column_type}"
