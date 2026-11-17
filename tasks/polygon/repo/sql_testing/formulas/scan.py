from sql_testing.smt.ast import *


class FScan:
    def __init__(self, table_name: str, env):
        self.output_table = env.db.find_table_by_name(table_name, env)
        if self.output_table.lineage is None:
            self.output_table.lineage = f"Scanned from initial schema"

        output_table_id = self.output_table.table_id

        cases = []
        choice_constraints = []
        for tuple_idx in range(self.output_table.bound):
            choice_constraints.append(Or([
                Choice(output_table_id, tuple_idx) == Int(1),
                Choice(output_table_id, tuple_idx) == Int(0)
            ]))

            cases.append(
                Implies(
                    Choice(output_table_id, tuple_idx) == Int(1),
                    Not(Deleted(output_table_id, tuple_idx))
                )
            )
            cases.append(
                Implies(
                    Choice(output_table_id, tuple_idx) == Int(0),
                    Deleted(output_table_id, tuple_idx)
                )
            )

        # env.formulas.append(And([*cases, *choice_constraints]), label=f'scan_{table_name}')
