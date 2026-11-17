from sql_testing.smt.ast import *


class Underapproximator:
    def __init__(self, env):
        self.env = env
        self.underapproximation_constraints = []

    def encode(self, operator: str, *args):
        method_name = f'encode_{operator}'
        if hasattr(self, method_name):
            return getattr(self, method_name)(*args)
        else:
            raise NotImplementedError(f'{operator}({args})')

    def encode_not_like(self, cell, pattern: str):
        cell_val, cell_null = cell

        considered_cases = []

        # cases for val to be true
        val_true_cases = []
        null = cell_null

        # underapproximate certain instances
        if not pattern.startswith('%'):
            case = cell_val == Int(self.env.string_hash(f"1{pattern.replace('%', '')}"))
            val_true_cases.append(case)
            considered_cases.append(case)

        if not pattern.endswith('%'):
            case = cell_val == Int(self.env.string_hash(f"{pattern.replace('%', '')}1"))
            val_true_cases.append(case)
            considered_cases.append(case)

        case = cell_val == Int(self.env.string_hash(pattern.replace('%', '1')))
        considered_cases.append(case)

        if len(pattern) > 0:
            case = cell_val == Int(self.env.string_hash(''))
            val_true_cases.append(case)
            considered_cases.append(case)

        self.underapproximation_constraints.append(Or(considered_cases))

        return Or(val_true_cases), null

    def encode_like(self, cell, pattern: str):
        cell_val, cell_null = cell

        considered_cases = []

        # cases for val to be true
        val_true_cases = []
        null = cell_null

        # underapproximate certain instances
        if not pattern.startswith('%'):
            case = cell_val == Int(self.env.string_hash(f"1{pattern.replace('%', '')}"))
            considered_cases.append(case)

        if not pattern.endswith('%'):
            case = cell_val == Int(self.env.string_hash(f"{pattern.replace('%', '')}1"))
            considered_cases.append(case)

        if pattern.find('%') == -1:
            case = cell_val == Int(self.env.string_hash(pattern))
            val_true_cases.append(case)
            considered_cases.append(case)

        case = cell_val == Int(self.env.string_hash(pattern.replace('%', '1')))
        val_true_cases.append(case)
        considered_cases.append(case)

        self.underapproximation_constraints.append(Or(considered_cases))

        return Or(val_true_cases), null
