import datetime

from sql_testing.logger import logger
from sql_testing.smt.ast import *

cmp_op_map = {
    'gt': Gt,
    'gte': Gte,
    'lt': Lt,
    'lte': Lte,
    'neq': Neq,
}


def encode_integrity_constraints(constraints, env):
    f = []

    for constraint in constraints:
        key = next(iter(constraint))
        match key:
            case 'primary' | 'distinct':
                f.append(encode_primary(constraint, env))
            case 'eq':
                f.append(encode_foreign_key(constraint, env))
            case 'enum':
                f.append(encode_enum(constraint, env))
            case 'gt' | 'gte' | 'lt' | 'lte' | 'neq':
                f.append(encode_cmp(constraint, cmp_op_map[key], env))
            case 'domain':
                f.append(encode_value_domain(constraint, env))
            case 'inclusion1':
                f.append(encode_inclusion1(constraint, env))
            case 'not_null':
                f.append(encode_not_null(constraint, env))
            case _:
                logger.warning(f"{constraint} is not implemented")

    return And(f)


def encode_primary(constraint, env):
    f = []

    key = next(iter(constraint))
    table_name = constraint[key][0].split('.')[0]
    table = env.db.find_table_by_name(table_name, env)
    distinct_attr = []
    for attr in constraint[key]:
        attr = attr.split('.')[1].lower()
        distinct_attr.append(attr)

    def _f_distinct_from(tuple_idx, others):
        _f = []
        for other_idx in others:
            tuple_distinct = []
            for column in table:
                if column.column_name not in distinct_attr:
                    continue
                column_id = column.column_id
                tuple_distinct.append(Or([
                    env.null(table.table_id, tuple_idx, column_id) != env.null(table.table_id, other_idx, column_id),
                    And([
                        And([
                            Not(env.null(table.table_id, tuple_idx, column_id)),
                            Not(env.null(table.table_id, other_idx, column_id))
                        ]),
                        env.cell(table.table_id, tuple_idx, column_id) !=
                        env.cell(table.table_id, other_idx, column_id)
                    ]),
                ]))
            _f.append(
                Implies(
                    And([Not(Deleted(table.table_id, tuple_idx)), Not(Deleted(table.table_id, other_idx))]),
                    Or(tuple_distinct)
                )
            )
        return And(_f)

    for tuple_id in range(table.bound):
        f.append(_f_distinct_from(tuple_id, [i for i in range(table.bound) if i < tuple_id]))
        if key == 'primary':
            for column in table:
                if column.column_name not in distinct_attr:
                    continue
                column_id = column.column_id
                f.append(Not(env.null(table.table_id, tuple_id, column_id)))

    return And(f)


def encode_enum(constraint, env):
    f = []

    key = next(iter(constraint))
    attr = constraint[key][0].split('.')
    enum = [env.string_hash(val) if isinstance(val, str) else val for val in constraint[key][1]]
    table = env.db.find_table_by_name(attr[0], env)

    for tuple_idx in range(table.bound):
        for column in table:
            if column.column_name == attr[1].lower():
                f.append(
                    Implies(
                        Not(Deleted(table.table_id, tuple_idx)),
                        Or([
                            env.cell(table.table_id, tuple_idx, column.column_id) == Int(val)
                            for val in enum
                        ])
                    )
                )

    return And(f)


def encode_cmp(constraint, op, env):
    f = []

    key = next(iter(constraint))

    if isinstance(constraint[key][1], str) and '.' in constraint[key][1]:
        lhs = constraint[key][0].split('.')
        rhs = constraint[key][1].split('.')
        lhs_table = env.db.find_table_by_name(lhs[0], env)
        rhs_table = env.db.find_table_by_name(rhs[0], env)
        assert lhs_table.is_same(rhs_table)

        lhs_column_id = None
        rhs_column_id = None
        for column in lhs_table:
            if column.column_name.lower() == lhs[1].lower():
                lhs_column_id = column.column_id
            if column.column_name.lower() == rhs[1].lower():
                rhs_column_id = column.column_id

        assert lhs_column_id is not None and rhs_column_id is not None

        for tuple_idx in range(lhs_table.bound):
            f.append(
                Implies(
                    Not(Deleted(lhs_table.table_id, tuple_idx)),
                    And([
                        # Not(Or([
                        #     self.null(lhs_table.table_id, tuple_idx, lhs_column_id),
                        #     self.null(lhs_table.table_id, tuple_idx, rhs_column_id),
                        # ])),
                        op(
                            env.cell(lhs_table.table_id, tuple_idx, lhs_column_id),
                            env.cell(lhs_table.table_id, tuple_idx, rhs_column_id)
                        )
                    ])
                )
            )
    else:
        attr = constraint[key][0].split('.')
        val = constraint[key][1]
        if isinstance(val, datetime.date):
            first = datetime.date(1000, 1, 1)
            diff = val - first
            val = diff.days
        table = env.db.find_table_by_name(attr[0], env)
        for tuple_idx in range(table.bound):
            for column in table:
                if column.column_name == attr[1].lower():
                    f.append(
                        Implies(
                            Not(Deleted(table.table_id, tuple_idx)),
                            And([
                                # Not(env.null(table.table_id, tuple_idx, column.column_id)),
                                op(
                                    env.cell(table.table_id, tuple_idx, column.column_id),
                                    Int(val)
                                )
                            ])
                        )
                    )

    return And(f)


def encode_value_domain(constraint, env):
    f = []

    key = next(iter(constraint))
    attr = constraint[key][0].split('.')
    lower_bound = constraint[key][1]
    upper_bound = constraint[key][2]
    table = env.db.find_table_by_name(attr[0], env)
    for tuple_idx in range(table.bound):
        for column in table:
            if column.column_name.lower() == attr[1].lower():
                f.append(
                    Implies(
                        Not(Deleted(table.table_id, tuple_idx)),
                        And([
                            # Not(env.null(table.table_id, tuple_idx, column.column_id)),
                            env.cell(table.table_id, tuple_idx, column.column_id) >= Int(lower_bound),
                            env.cell(table.table_id, tuple_idx, column.column_id) <= Int(upper_bound),
                        ])
                    )
                )

    return And(f)

def encode_foreign_key(constraint, env):
    f = []

    key = next(iter(constraint))

    # foreign key
    # t1.a <- t2.b
    if isinstance(constraint[key][1], str) and '.' in constraint[key][1]:
        lhs = constraint[key][0].split('.')
        rhs = constraint[key][1].split('.')
        lhs_table = env.db.find_table_by_name(lhs[0], env)
        rhs_table = env.db.find_table_by_name(rhs[0], env)

        env.integrity_constraints.append(
            Implies(
                Sum([If(Not(Deleted(lhs_table.table_id, tuple_id)), Int(1), Int(0)) for tuple_id in range(lhs_table.bound)]),
                Sum([If(Not(Deleted(rhs_table.table_id, tuple_id)), Int(1), Int(0)) for tuple_id in range(rhs_table.bound)]),
            )
        )

        for column in rhs_table:
            if column.column_name.lower() == rhs[1].lower():
                rhs_column_id = column.column_id

        for lhs_idx in range(lhs_table.bound):
            for column in lhs_table:
                if column.column_name.lower() == lhs[1].lower():
                    f.append(
                        Implies(
                            Not(Deleted(lhs_table.table_id, lhs_idx)),
                            Or([
                                env.null(lhs_table.table_id, lhs_idx, column.column_id),
                                And([
                                    Not(env.null(lhs_table.table_id, lhs_idx, column.column_id)),
                                    Or([
                                        And([
                                            Not(Deleted(rhs_table.table_id, rhs_idx)),
                                            Not(env.null(rhs_table.table_id, rhs_idx, rhs_column_id)),
                                            env.cell(lhs_table.table_id, lhs_idx, column.column_id) ==
                                            env.cell(rhs_table.table_id, rhs_idx, rhs_column_id)
                                        ])
                                        for rhs_idx in range(rhs_table.bound)
                                    ])
                                ])
                            ])
                        )
                    )
    else:
        # value equality
        raise NotImplementedError(f"{constraint} is not implemented")

    return And(f)


def encode_not_null(constraint, env):
    f = []
    key = next(iter(constraint))
    attr = constraint[key].split('.')
    table = env.db.find_table_by_name(attr[0], env)
    for tuple_idx in range(table.bound):
        for column in table:
            if column.column_name.lower() == attr[1].lower():
                f.append(
                    Implies(
                        Not(Deleted(table.table_id, tuple_idx)),
                        Not(env.null(table.table_id, tuple_idx, column.column_id))
                    )
                )
    return And(f)


def encode_inclusion1(constraint, env):
    key = next(iter(constraint))

    vals = []
    attr = constraint[key][0]['val'][0].split('.')
    for pair in constraint[key]:
        if isinstance(pair['val'][1], str):
            vals.append(env.string_hash(pair['val'][1]))
        else:
            vals.append(pair['val'][1])
    table = env.db.find_table_by_name(attr[0], env)

    for column in table:
        if column.column_name.lower() == attr[1].lower():
            column_id = column.column_id

    f = Or([
            And([
                Not(Deleted(table.table_id, tuple_idx)),
                Or([
                    And([
                        Not(env.null(table.table_id, tuple_idx, column_id)),
                        env.cell(table.table_id, tuple_idx, column_id) == Int(val),
                    ])
                    for val in vals
                ])
            ])
            for tuple_idx in range(table.bound)
        ])

    return f
