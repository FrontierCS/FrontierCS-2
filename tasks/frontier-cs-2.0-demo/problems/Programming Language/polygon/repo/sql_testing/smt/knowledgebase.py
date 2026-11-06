from collections import defaultdict

from sql_testing.smt.ast import *


class KnowledgeBase:
    def __init__(self):
        # FRONTIER_EDIT_BEGIN: search state
        pass
        # FRONTIER_EDIT_END: search state

    def add_conflict(self, conflict: dict, labels):
        # FRONTIER_EDIT_BEGIN: search-state update
        raise NotImplementedError("implement search-state updates")
        # FRONTIER_EDIT_END: search-state update

    def get_block_formula(self):
        pass
