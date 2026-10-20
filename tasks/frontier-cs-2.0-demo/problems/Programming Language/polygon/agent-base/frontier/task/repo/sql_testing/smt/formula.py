import itertools
import random

from bidict import bidict
from copy import deepcopy

from sql_testing.logger import logger
from sql_testing.smt.ast import *
from sql_testing.smt.knowledgebase import KnowledgeBase
from sql_testing.smt.provers.smtlibv2 import SMTLIBv2Visitor, SMTLIBv2


class FormulaManager:
    def __init__(self, env):
        self.formulas = {}
        self.label_to_node = {}
        self.next_label = 0
        self.node_cur_label = 0
        self.env = env
        self.kb = KnowledgeBase()
        self.timeout = 120

        self.label_to_table_id = None
        self.current_under = {}
        self.labels_considered = set()

        self.under_table_id_to_original = {}
        self.under_config = {}
        self.table_bound = {}

        self.visited_formula_cache = {}

        self.ret = None

    def append(self, f: SMTNode, label: str = None):
        if label is None:
            label = f'f_{self.next_label}'
            self.next_label += 1

        if label in self.formulas:
            # label = f'{label}_{self.next_label}'
            # self.formulas[label] = f
            self.formulas[label] = And([self.formulas[label], f])
            # self.next_label += 1
        else:
            self.formulas[label] = f

    def init_label_table_id_bidict(self):
        label_to_table_id = {}
        for table in self.env.db.schemas.values():
            # print(table.table_id, table.node, table.node.label if table.node is not None else None)
            if table.lineage is not None:
                if 'Duplicate' in table.lineage:
                    assert table.node is not None
                    table_id = table.table_id
                    if table_id in self.under_table_id_to_original:
                        table_id = self.under_table_id_to_original[table.table_id]

                    label_to_table_id[table.node.distinct_label] = table_id
                elif 'Group' in table.lineage:
                    assert table.node is not None
                    table_id = table.table_id
                    if table_id in self.under_table_id_to_original:
                        table_id = self.under_table_id_to_original[table.table_id]

                    label_to_table_id[table.node.label] = table_id
                    label_to_table_id[table.node.group_by_label] = table_id
                elif table.node is not None:
                    table_id = table.table_id
                    if table_id in self.under_table_id_to_original:
                        table_id = self.under_table_id_to_original[table.table_id]

                    label_to_table_id[table.node.label] = table_id

        self.label_to_table_id = label_to_table_id
        logger.debug(self.label_to_table_id)

    def search_naive(self, outputs, ret):
        # FRONTIER_EDIT_BEGIN: exhaustive baseline
        raise NotImplementedError("implement an exhaustive baseline if useful")
        # FRONTIER_EDIT_END: exhaustive baseline

    def search(self, outputs, ret):
        # FRONTIER_EDIT_BEGIN: search policy
        raise NotImplementedError("implement the under-approximation search policy")
        # FRONTIER_EDIT_END: search policy

    def backtrack(self, unsat_core, ret):
        # FRONTIER_EDIT_BEGIN: recovery policy
        raise NotImplementedError("implement search recovery")
        # FRONTIER_EDIT_END: recovery policy

    def cover_ua(self, node_label, left_tops=None, right_tops=None, left_right_tops=None, tops_ratio=None):
        table = self.env.db.schemas[self.label_to_table_id[node_label]]
        vec_size = table.bound

        if 'Sorted' in table.lineage:
            yield ['T'] * vec_size
        elif left_right_tops is not None and left_right_tops * 2 < vec_size:
            for cover in [list(x) for x in itertools.product([0, 1], repeat=vec_size - left_right_tops * 2)]:
                vec = ['T'] * left_right_tops + cover + ['T'] * left_right_tops
                if table.lineage is not None and 'Grouped' in table.lineage:
                    vec = vec * 2
                yield vec
        elif left_tops is not None and left_tops < vec_size:
            for cover in [list(x) for x in itertools.product([0, 1], repeat=vec_size - left_tops)]:
                vec = ['T'] * left_tops + cover
                if table.lineage is not None and 'Grouped' in table.lineage:
                    vec = vec * 2
                yield vec
        elif right_tops is not None and right_tops < vec_size:
            for cover in [list(x) for x in itertools.product([0, 1], repeat=vec_size - right_tops)]:
                vec = cover + ['T'] * right_tops
                if table.lineage is not None and 'Grouped' in table.lineage:
                    vec = vec * 2
                yield vec
        elif tops_ratio is not None:
            random.seed(123456)
            num_tops = int(vec_size * tops_ratio)
            top_indices = random.sample(range(0, vec_size), num_tops)
            for cover in [list(x) for x in itertools.product([0, 1], repeat=vec_size - num_tops)]:
                cover = list(cover)
                next_cover_idx = 0
                vec = []
                for i in range(vec_size):
                    if i in top_indices:
                        vec.append('T')
                    else:
                        vec.append(cover[next_cover_idx])
                        next_cover_idx += 1
                if table.lineage is not None and 'Grouped' in table.lineage:
                    vec = vec * 2
                yield vec
        else:
            yield ['T'] * vec_size

    def add_kb(self, unsat_core):
        # FRONTIER_EDIT_BEGIN: search memory
        raise NotImplementedError("implement search-state recording")
        # FRONTIER_EDIT_END: search memory

    def encode_current_under(self):
        if not self.current_under:
            if 'under' in self.formulas:
                del self.formulas['under']
            return
        f = []
        for table_id, vec in self.current_under.items():
            for bit_id, bit_val in enumerate(vec):
                if bit_val != 'T':
                    f.append(Choice(table_id, bit_id) == Int(bit_val))
        self.formulas['under'] = And(f)

    def solve_precise(self, ret):
        # FRONTIER_EDIT_BEGIN: precise fallback
        raise NotImplementedError("implement a precise fallback if useful")
        # FRONTIER_EDIT_END: precise fallback

    def dump(self):
        visitor = SMTLIBv2Visitor()
        out = ''
        for label, formula in self.formulas.items():
            if '$' in label and label not in self.labels_considered:
                continue
            # print(label)

            # cache operator encodings since they will not change
            if '$' in label or 'scan' in label or label in ['ic', 'neq', 'disambiguation']:
                if label not in self.visited_formula_cache:
                    self.visited_formula_cache[label] = formula.accept(visitor)
                formula_smt_lib = self.visited_formula_cache[label]
            else:
                formula_smt_lib = formula.accept(visitor)
            out = f'{out}\n(assert (! {formula_smt_lib} :named {label}))'

        # FRONTIER_EDIT_BEGIN: additional search constraints
        pass
        # FRONTIER_EDIT_END: additional search constraints

        return out

    def next_node_label(self):
        self.node_cur_label += 1
        return self.node_cur_label
