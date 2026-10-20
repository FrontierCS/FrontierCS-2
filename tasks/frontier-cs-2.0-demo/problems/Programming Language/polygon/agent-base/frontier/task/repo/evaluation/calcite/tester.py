import re

import mysql.connector
import operator
import os
import pandas as pd
import ujson

from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime
from queue import PriorityQueue
from tqdm import tqdm

from sql_testing.logger import logger


def type_string(column):
    extra_info = ''
    if 'char' in column['Type'].lower():
        column['Type'] = 'varchar'
    match column['Type'].lower():
        case 'numeric':
            data_type = 'float'
        case 'varchar' | 'text':
            data_type = 'varchar'
            extra_info = '(255)'
        case 'date':
            data_type = 'date'
        case _:
            if column['Type'].startswith('enum'):
                data_type = 'varchar'
                extra_info = '(255)'
            else:
                data_type = 'bigint'
    return data_type + extra_info


def compare_results(result1, result2, consider_order=False):
    if consider_order:
        return result1 == result2
    else:
        return Counter(result1) == Counter(result2)


def verieql_preprocessing(query: str):
    query = query.strip().upper().replace('`', ' ')
    if (query[0] == '\'' and query[-1] == '\'') or (query[0] == '\"' and query[-1] == '\"'):
        query = query[1:-1]
    alias_names = re.findall(r'[a-zA-Z0-9_)]+\s*""\s*[a-zA-Z0-9_]+\s*""', query)
    if len(alias_names) > 0:
        for name in alias_names:
            if str.startswith(name, 'WHEN') or str.startswith(name, 'WHERE') \
                    or re.match(r'\d+\s*""\s*AND\s*""', name) or re.match(r'\d+\s*""\s*OR\s*""', name):
                continue
            index = query.find(name)
            query = query[:index] + name.replace('""', ' ') + query[index + len(name):]
    alias_names = re.findall(r'[a-zA-Z0-9_)]+\s*"\s*[a-zA-Z0-9_]+\s*"', query)
    if len(alias_names) > 0:
        for name in alias_names:
            if str.startswith(name, 'WHEN') or str.startswith(name, 'WHERE') \
                    or re.match(r'\d+\s*"\s*AND\s*"', name) or re.match(r'\d+\s*"\s*OR\s*"', name):
                continue
            index = query.find(name)
            query = query[:index] + name.replace('"', ' ') + query[index + len(name):]
    alias_names = re.findall(r"[a-zA-Z0-9_)]+\s*'\s*[a-zA-Z0-9_]+\s*'", query)
    if len(alias_names) > 0:
        for name in alias_names:
            if str.startswith(name, 'WHEN') or str.startswith(name, 'WHERE') \
                    or re.match(r"\d+\s*'\s*AND\s*'", name) or re.match(r"\d+\s*'\s*OR\s*'", name):
                # WHEN 'RED'
                # '2019-01-01' AND '2019-03-31'
                continue
            index = query.find(name)
            query = query[:index] + name.replace("'", ' ') + query[index + len(name):]
    query = query.replace('""', '\'').replace('\"', '\'')
    query = re.sub(r'\s+', ' ', query).strip()
    return query


class MySQLTester:
    def __init__(self,
                 config,
                 schema,
                 db_predix='testing_calcite',
                 ):
        self.cnx = mysql.connector.connect(**config)
        self.cursor = self.cnx.cursor(buffered=True)

        self.cursor.execute("SET GLOBAL sql_mode=(SELECT REPLACE(@@sql_mode,'ONLY_FULL_GROUP_BY',''));")

        self.schema = schema
        self.db_prefix = db_predix

        self.db_created = PriorityQueue()
        self.groundtruth_result = {}

        self.databases = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, tb):
        while not self.db_created.empty():
            self.drop_database(self.db_created.get()[1])
        self.cursor.close()
        self.cnx.close()

    def create_all_databases(self, databases, constraints=None):
        self.databases = databases

        for idx, database in enumerate(databases):
            try:
                self.create_test_database(idx, database, constraints)
            except Exception as e:
                logger.critical(f'Error when creating database: {database}, error: {e}')
                continue

    def create_test_database(self, idx, database, constraints):
        self.drop_database(idx)

        db_name = f'{self.db_prefix}_{idx}'
        self.cursor.execute(f'CREATE DATABASE {db_name}')
        self.cursor.execute(f'USE {db_name}')

        for table in self.schema:
            table_name = table['TableName'].lower()
            fields = []
            col_names = []

            for col in table['PKeys']:
                fields.append(f"{col['Name'].lower()} {type_string(col)}")
                col_names.append(col['Name'].lower())

            for col in table['FKeys']:
                if col['FName'].lower() in col_names:
                    continue
                p_table = int(col['PTable'])
                p_name = col['PName']
                p_cols = self.schema[p_table]['PKeys'] + self.schema[p_table]['Others']
                data_type = None
                for p_col in p_cols:
                    if p_col['Name'] == p_name:
                        data_type = p_col['Type']
                        break
                if 'char' in data_type.lower() or 'text' in data_type.lower():
                    data_type = 'varchar(255)'
                fields.append(f"`{col['FName'].lower()}` {data_type}")
                col_names.append(col['FName'].lower())

            for col in table['Others']:
                if col['Name'].lower() in col_names:
                    continue
                fields.append(f"`{col['Name'].lower()}` {type_string(col)}")

            create_statement = f"CREATE TABLE {table_name} ({', '.join(fields)});"
            self.cursor.execute(create_statement)

            table_data = database.get(table_name, [])[1:]
            if len(table_data) > 0:
                rows = [tuple(row) for row in table_data]
                insert_statement = f"INSERT INTO {table_name} VALUES ({', '.join(['%s' for _ in range(len(table_data[0]))])});"

                self.cursor.executemany(insert_statement, rows)
                self.cnx.commit()

        # print(self.check_database_integrity(idx, constraints))
        if not self.check_database_integrity(idx, constraints):
            logger.critical(f'check DB integrity: {database} (IC: {constraints})')
        self.db_created.put((0, idx))

    def test_pair(self, groundtruth, query):
        new_queue = []
        while not self.db_created.empty():
            priority, idx = self.db_created.get()
            self.cursor.execute(f'USE {self.db_prefix}_{idx}')
            try:
                self.cursor.execute(groundtruth)
            except mysql.connector.errors.ProgrammingError as e:
                logger.error(e)
                return False
            self.groundtruth_result[idx] = self.cursor.fetchall()
            new_queue.append((priority, idx))
        for x in new_queue:
            self.db_created.put(x)

        if 'ORDER BY' in groundtruth.upper():
            consider_order = True
        else:
            consider_order = False

        rejected = False
        new_queue = []
        while not self.db_created.empty():
            priority, idx = self.db_created.get()
            self.cursor.execute(f'USE {self.db_prefix}_{idx}')
            try:
                self.cursor.execute(query)
                q2_result = self.cursor.fetchall()

                if not compare_results(self.groundtruth_result[idx], q2_result, consider_order):
                    # logger.info(self.databases[idx])
                    logger.info(f'Q1: {self.groundtruth_result[idx]}')
                    logger.info(f'Q2: {q2_result}')
                    rejected = True
                    new_queue.append((priority - 1, idx))

                    while not self.db_created.empty():
                        new_queue.append(self.db_created.get())
                    break
                else:
                    logger.warning(groundtruth)
                    logger.warning(query)
                    logger.warning(f'{self.groundtruth_result[idx]}, {q2_result}')
            except Exception as e:
                logger.error(f'{query}, {e}')
                pass

            new_queue.append((priority, idx))

        for x in new_queue:
            self.db_created.put(x)

        return rejected

    def mutants_statistics(self, mutants, unsat_mutants):
        idx = 0
        all_mutants = 0
        killed_mutants = 0
        for mutation_type in mutants:
            for mutated in mutants[mutation_type]:
                if str(mutated) in unsat_mutants:
                    continue
                all_mutants += 1

                try:
                    self.cursor.execute(f'USE {self.db_prefix}_{idx}')

                    self.cursor.execute(str(mutated))

                    if not compare_results(self.groundtruth_result[idx], self.cursor.fetchall()):
                        killed_mutants += 1
                    else:
                        # print(self.databases_df[idx])
                        logger.debug("Not killed:", mutation_type, str(mutated))
                except Exception as e:
                    all_mutants -= 1
                    # print(str(mutated), e)
                    pass
                idx += 1
        return all_mutants, killed_mutants

    def check_database_integrity(self, idx, constraints: list[dict]) -> bool:
        self.cursor.execute(f'USE {self.db_prefix}_{idx}')

        operator_map = {
            'gt': operator.gt,
            'gte': operator.ge,
            'lt': operator.lt,
            'lte': operator.le,
            'neq': operator.ne,
            'eq': operator.eq,
        }

        if constraints is None:
            return True

        for c in constraints:
            key = next(iter(c))
            match key:
                case 'primary':
                    columns = c[key]
                    if not isinstance(columns, list):
                        columns = [columns]
                    table = columns[0].split('.')[0]
                    columns = [col.split('.')[1] for col in columns]

                    self.cursor.execute(f"SELECT * FROM {table}")
                    header = [a[0].lower() for a in self.cursor.description]
                    rows = self.cursor.fetchall()
                    pk_indices = [header.index(col.lower()) for col in columns]
                    visited = []
                    for row in rows:
                        row_values = [row[index] for index in pk_indices]
                        if row_values in visited or None in row_values:
                            return False
                        visited.append(row_values)
                case 'distinct':
                    columns = c[key]
                    if not isinstance(columns, list):
                        columns = [columns]
                    table = columns[0].split('.')[0]
                    columns = [col.split('.')[1] for col in columns]

                    self.cursor.execute(f"SELECT * FROM {table}")
                    header = [a[0].lower() for a in self.cursor.description]
                    rows = self.cursor.fetchall()
                    pk_indices = [header.index(col.lower()) for col in columns]
                    visited = []
                    for row in rows:
                        row_values = [row[index] for index in pk_indices]
                        if row_values in visited:
                            return False
                        visited.append(row_values)
                case 'foreign':
                    table = c[key][0].split('.')[0]
                    column = c[key][0].split('.')[1]
                    references_table = c[key][1].split('.')[0]
                    references_column = c[key][1].split('.')[1]

                    self.cursor.execute(f"SELECT * FROM {references_table}")
                    header = [a[0].lower() for a in self.cursor.description]
                    fk_references_index = header.index(references_column.lower())
                    rows = self.cursor.fetchall()
                    foreign_values = [row[fk_references_index] for row in rows]

                    self.cursor.execute(f"SELECT * FROM {table}")
                    header = [a[0].lower() for a in self.cursor.description]
                    fk_index = header.index(column.lower())
                    rows = self.cursor.fetchall()
                    for row in rows:
                        if row[fk_index] not in foreign_values:
                            return False
                case 'between':
                    table = c[key][0].split('.')[0]
                    column = c[key][0].split('.')[1]
                    lower_bound = c[key][1]
                    upper_bound = c[key][2]

                    self.cursor.execute(f"SELECT * FROM {table}")
                    header = [a[0].lower() for a in self.cursor.description]
                    index = header.index(column.lower())
                    rows = self.cursor.fetchall()
                    for row in rows:
                        if row[index] is None or not (lower_bound <= row[index] <= upper_bound):
                            return False
                case 'in':
                    table = c[key][0].split('.')[0]
                    column = c[key][0].split('.')[1]
                    subset = []
                    for val in c[key][1]:
                        if isinstance(val, dict):
                            subset.append(val['literal'])
                        else:
                            subset.append(val)

                    self.cursor.execute(f"SELECT * FROM {table}")
                    header = [a[0].lower() for a in self.cursor.description]
                    index = header.index(column.lower())
                    rows = self.cursor.fetchall()
                    for row in rows:
                        if not row[index] in subset:
                            return False
                case 'not_null':
                    table = c[key].split('.')[0]
                    column = c[key].split('.')[1]

                    self.cursor.execute(f"SELECT * FROM {table}")
                    header = [a[0].lower() for a in self.cursor.description]
                    rows = self.cursor.fetchall()
                    not_null_index = header.index(column.lower())
                    for row in rows:
                        if row[not_null_index] is None:
                            return False
                case 'eq' if isinstance(c[key][1], str) and '.' in c[key][1]:
                    table = c[key][0].split('.')[0]
                    column = c[key][0].split('.')[1]
                    references_table = c[key][1].split('.')[0]
                    references_column = c[key][1].split('.')[1]

                    self.cursor.execute(f"SELECT * FROM {references_table}")
                    header = [a[0].lower() for a in self.cursor.description]
                    fk_references_index = header.index(references_column.lower())
                    rows = self.cursor.fetchall()
                    foreign_values = [row[fk_references_index] for row in rows]

                    self.cursor.execute(f"SELECT * FROM {table}")
                    header = [a[0].lower() for a in self.cursor.description]
                    fk_index = header.index(column.lower())
                    rows = self.cursor.fetchall()
                    for row in rows:
                        if row[fk_index] is not None and row[fk_index] not in foreign_values:
                            return False
                case 'gt' | 'gte' | 'lt' | 'lte' | 'eq' | 'neq' if not (isinstance(c[key][1], str) and '.' in c[key][1]):
                    table = c[key][0].split('.')[0]
                    column = c[key][0].split('.')[1]

                    self.cursor.execute(f"SELECT * FROM {table}")
                    header = [a[0].lower() for a in self.cursor.description]
                    index = header.index(column.lower())
                    rows = self.cursor.fetchall()
                    for row in rows:
                        if isinstance(c[key][1], str) and '.' in c[key][1]:
                            rhs_column = c[key][1].split('.')[1]
                            rhs_index = header.index(rhs_column.lower())
                            compare_to = row[rhs_index]
                        else:
                            compare_to = c[key][1]
                        if row[index] is not None and not operator_map[key](row[index], compare_to):
                            return False
                case 'inc':
                    table = c[key]['value'].split('__')[0]
                    column = c[key]['value'].split('__')[1]

                    self.cursor.execute(f"SELECT * FROM {table}")
                    header = [a[0] for a in self.cursor.description]
                    rows = self.cursor.fetchall()
                    inc_index = header.index(column)
                    inc_values = [row[inc_index] for row in rows]
                    if not all(
                            i is not None and j is not None and j == i + 1 for i, j in zip(inc_values, inc_values[1:])):
                        return False
                case _:
                    pass
                    # raise Exception(f'Unknown integrity constraint type: {key}')
        return True

    def drop_database(self, idx):
        db_name = f'{self.db_prefix}_{idx}'
        self.cursor.execute(f'DROP DATABASE IF EXISTS {db_name}')
