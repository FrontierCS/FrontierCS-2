import datetime
import traceback
import multiprocess.pool
import numpy as np

from functools import cache
from typing import Tuple

from sql_testing.ast.expressions.attribute import Attribute
from sql_testing.ast.expressions.literal import Literal
from sql_testing.formulas.integrity_constraint import encode_integrity_constraints
from sql_testing.logger import logger
from sql_testing.schemas import *
from sql_testing.smt.ast import *
from sql_testing.smt.formula import FormulaManager
from sql_testing.smt.provers.smtlibv2 import SMTLIBv2
from sql_testing.sql_parser import SQLParser
from sql_testing.utils import create_empty_table
from sql_testing.variables import *
from sql_testing.visitors.expression_encoder import ExpressionEncoder
from sql_testing.visitors.initializer import Initializer
from sql_testing.visitors.query_encoder import QueryEncoder
from sql_testing.visitors.underapproximator import Underapproximator


class Environment:
    def __init__(self, schema, constraints, bound=2, time_budget=60, default_k=None):
        self.db = Database()
        self.bound_size = bound
        self.table_id_counter = -1

        self.schema = schema
        self.constraints = constraints

        self.time_budget = time_budget
        self.integrity_constraints = []
        self.formulas = FormulaManager(self)
        self.formulas.timeout = self.time_budget

        # z3 variables
        self.cell = SMTCell
        self.null = SMTNull
        self.size = SMTSize
        self.grouping = SMTGrouping

        self.string_hash_table = {}
        self.hash_string_table = {}
        self.curr_query_id = None

        self.load_schema(schema)

        self.underapproximator = Underapproximator(self)
        self.initialized = False
        if default_k is not None:
            self.default_k = default_k
        else:
            self.default_k = {
                'filter': 99,
                'project': 99,
                'union': 99,
                'inner join': 99,
                'left join': 99,
                'right join': 99,
                'full join': 99,
                'product': 99,
                'order by': 99
            }

        self.formulas.append(encode_integrity_constraints(self.constraints, self), label='ic')

    def next_table_id(self) -> int:
        self.table_id_counter += 1
        return self.table_id_counter

    def string_hash(self, string: str) -> int:
        if not isinstance(string, str):
            string = str(string)

        if string in self.string_hash_table:
            return self.string_hash_table[string]

        try:
            h = int(string)
        except:
            h = hash(string)
        self.string_hash_table[string] = h
        self.hash_string_table[h] = string

        return h

    def lookup_string(self, string_hash: int) -> str:
        if string_hash in self.hash_string_table:
            return self.hash_string_table[string_hash]

        candidates = list(self.hash_string_table.values())
        if len(candidates) > 0:
            return candidates[string_hash % len(candidates)] + str(string_hash)
        return str(string_hash)

    def load_schema(self, schema: list):
        self.integrity_constraints = []
        enum_constraints = []
        for idx, table in enumerate(schema):
            table_id = self.next_table_id()
            table_name = table["TableName"].lower()
            table_bound = self.bound_size
            table_schema = TableSchema(table_id, table_name, table_bound)
            column_id = 0
            for col in table['PKeys']:
                column_name = col['Name'].lower()
                data_type = col['Type'].split(',')[0]
                if data_type == 'enum':
                    enum = col['Type'].split(',')[1:]
                    enum_constraints.append({'enum': [f'{table_name}.{column_name}', enum]})
                    data_type = 'varchar'
                column_schema = ColumnSchema(column_id, column_name, data_type, table_name=table_name)
                column_id += 1
                table_schema.append(column_schema)
            for col in table['FKeys']:
                column_name = col['FName'].lower()
                has_column = False
                for column in table_schema.columns:
                    if column.column_name == column_name:
                        has_column = True
                        break
                if has_column:
                    continue
                p_table = int(col["PTable"])
                p_name = col["PName"]
                p_cols = schema[p_table]["PKeys"]
                data_type = None
                for p_col in p_cols:
                    if p_col["Name"] == p_name:
                        data_type = p_col["Type"].split(',')[0]
                        break
                if data_type == 'enum':
                    enum = col['Type'].split(',')[1:]
                    enum_constraints.append({'enum': [f'{table_name}.{column_name}', enum]})
                    data_type = 'varchar'
                column_schema = ColumnSchema(column_id, column_name, data_type, table_name=table_name)
                if column_schema in table_schema:
                    continue
                column_id += 1
                table_schema.append(column_schema)
            for col in table['Others']:
                column_name = col['Name'].lower()
                has_column = False
                for column in table_schema.columns:
                    if column.column_name == column_name:
                        has_column = True
                        break
                if has_column:
                    continue
                data_type = col['Type'].split(',')[0]
                if data_type == 'enum':
                    enum = col['Type'].split(',')[1:]
                    enum_constraints.append({'enum': [f'{table_name}.{column_name}', enum]})
                    data_type = 'varchar'
                column_schema = ColumnSchema(column_id, column_name, data_type, table_name=table_name)
                if column_schema in table_schema:
                    continue
                column_id += 1
                table_schema.append(column_schema)
            self.db.add_table(table_schema)
            # self.formulas.append(And([self.size(table_id) >= Int(1), self.size(table_id) <= Int(table_bound)]), label=f'size_{table_name}')
            self.constraints.extend(enum_constraints)

        # add integrity constraints from schema
        for idx, table in enumerate(schema):
            table_name = table["TableName"].lower()
            if len(table['PKeys']) > 0:
                self.constraints.append(
                    {'primary': [f"{table_name}.{column['Name'].lower()}" for column in table['PKeys']]}
                )

            for col in table['FKeys']:
                column_name = col['FName'].lower()
                p_table_name = schema[int(col["PTable"])]["TableName"].lower()
                p_name = col["PName"].lower()
                self.constraints.append(
                    {'eq': [f"{table_name}.{column_name}", f"{p_table_name}.{p_name}"]}
                )

    def check(self, q1, q2, use_precise_encoding=False):
        # FRONTIER_EDIT_BEGIN: counterexample discovery
        raise NotImplementedError(
            "implement the SQL-to-counterexample discovery pipeline"
        )
        # FRONTIER_EDIT_END: counterexample discovery

    def disambiguate(self, queries, group_range=0, use_precise_encoding=False):
        # FRONTIER_EDIT_BEGIN: query disambiguation
        raise NotImplementedError(
            "implement the multi-query counterexample discovery pipeline"
        )
        # FRONTIER_EDIT_END: query disambiguation

    def o1_eq_o2(self, o1, o2):
        def f_multiplicity(r, t):
            indicators = []
            for tuple_idx in range(r.bound):
                tuple_eq = [Not(Deleted(r.table_id, tuple_idx))]
                for column_idx, column in enumerate(r):
                    tuple_eq.append(Or([
                        And([self.null(r.table_id, tuple_idx, column.column_id), self.null(*t[column_idx])]),
                        And([
                            Not(Or([self.null(r.table_id, tuple_idx, column.column_id), self.null(*t[column_idx])])),
                            self.cell(r.table_id, tuple_idx, column.column_id) == self.cell(*t[column_idx])
                        ])
                    ]))
                indicators.append(If(And(tuple_eq), Int(1), Int(0)))
            return Sum(indicators)

        o1_size = Sum([If(Not(Deleted(o1.table_id, tuple_id)), Int(1), Int(0)) for tuple_id in range(o1.bound)])
        o2_size = Sum([If(Not(Deleted(o2.table_id, tuple_id)), Int(1), Int(0)) for tuple_id in range(o2.bound)])

        if len(o1.columns) == len(o2.columns):
            lateral_bag_eq = []
            for tuple_id in range(o1.bound):
                lateral_bag_eq.append(
                    Implies(
                        Not(Deleted(o1.table_id, tuple_id)),
                        f_multiplicity(o1, [(o1.table_id, tuple_id, column.column_id) for column in o1]) ==
                        f_multiplicity(o2, [(o1.table_id, tuple_id, column.column_id) for column in o1])
                    )
                )
            f = [o1_size == o2_size, And(lateral_bag_eq)]

            # sorted columns are equivalent under list semantics
            if o1.lineage is not None and o2.lineage is not None and 'Sorted' in o1.lineage and 'Sorted' in o2.lineage:
                sorted_columns_list_eq = []
                o1_encoder = ExpressionEncoder(o1, self)
                o2_encoder = ExpressionEncoder(o2, self)
                for tuple_id in range(min(o1.bound, o2.bound)):
                    for expression_idx, expression in enumerate(o1.node.expressions):
                        if isinstance(expression, Literal):
                            if isinstance(expression.value, bool):
                                continue
                            o1_cell = self.db[o1.table_id, tuple_id, expression.value - 1]
                            o1_cell = o1_cell.VAL, o1_cell.NULL
                            o2_cell = self.db[o2.table_id, tuple_id, expression.value - 1]
                            o2_cell = o2_cell.VAL, o2_cell.NULL
                        else:
                            if isinstance(expression, Attribute) and '.' in expression.name:
                                expression = Attribute(expression.name.split('.')[1])
                            o1_cell = o1_encoder.expression_for_tuple(expression, tuple_id)
                            o2_cell = o2_encoder.expression_for_tuple(expression, tuple_id)

                        VAL, NULL = 0, 1
                        sorted_columns_list_eq.append(
                            Implies(
                                Not(Deleted(o1.table_id, tuple_id)),
                                Or([
                                    And([o1_cell[NULL], o2_cell[NULL]]),
                                    And([
                                        Not(Or([o1_cell[NULL], o2_cell[NULL]])),
                                        o1_cell[VAL] == o2_cell[VAL]
                                    ])
                                ])
                            )
                        )
                f.append(And(sorted_columns_list_eq))
            return And(f)
        else:
            return Or([o1_size > Int(0), o2_size > Int(0)])

    def copy_cell(self, original_cell, new_cell):
        return And([
            new_cell.NULL == original_cell.NULL,
            new_cell.VAL == original_cell.VAL,
        ])

    def clear(self):
        self.db = Database()
        self.table_id_counter = -1
        self.string_hash_table = {}
        self.hash_string_table = {}

        self.formulas = FormulaManager(self)

        self.load_schema(self.schema)
        self.formulas.append(encode_integrity_constraints(self.constraints, self), label='ic')
        # self.formulas.append(And(self.integrity_constraints), label='ic')

        self.underapproximator = Underapproximator(self)
