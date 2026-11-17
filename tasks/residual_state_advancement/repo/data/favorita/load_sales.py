#!/usr/bin/env python3
"""Load a development fact table the way the judge loads its workloads.

The four `sales-dev-<i>.parquet` files in this directory are the development
workloads' fact rows: real Corporacion Favorita sales rows dated 2017-02-02 to
2017-04-30, keyed to the dimension tables beside them. `load_sales` below is the
judge's loader, verbatim: it adds the constant `onpromotion` code and the target
`Y`, and returns the fact table together with the materialized join.

    python data/favorita/load_sales.py --fact data/favorita/sales-dev-0.parquet --out sales.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

FULL_FEATURES = [
    "htype", "locale", "locale_name", "transferred", "f2",
    "dcoilwtico", "f3", "transactions", "f5", "city", "state",
    "stype", "cluster", "f4", "family", "class", "perishable", "f1",
]


def _lookup(frame: pd.DataFrame, key: str, column: str) -> np.ndarray:
    keys = frame[key].to_numpy()
    if len(np.unique(keys)) != len(keys) or keys.min() < 0:
        raise RuntimeError("dimension key is not unique")
    table = np.zeros(int(keys.max()) + 1, dtype=frame[column].dtype)
    table[keys] = frame[column].to_numpy()
    return table


def load_sales(public_data: Path, fact_file: Path, limit: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fact table of real Favorita sales rows over the shipped dimension tables.

    ``fact_file`` is a Parquet slice of Corporacion Favorita sales rows with the
    columns ``item_nbr`` and ``tid`` (the keys of ``items.csv`` and
    ``transactions.csv``) and the real ``unit_sales``. Every row is a distinct
    (item_nbr, tid) pair, i.e. a distinct item, store and date. ``onpromotion``
    is not available for these rows and is set to 2, the code the artifact uses
    for a promotion flag that was not recorded. ``Y`` is the target formula of
    the upstream JoinBoost artifact (exact on its 10,000-row sample):
    f1 + log2(f2) - 10*f3 - 10*f4 + f5*log2(f5). ``limit`` keeps the first rows
    in file order. Returns the fact table and the aligned joined feature table.
    """
    items = pd.read_csv(public_data / "items.csv")
    transactions = pd.read_csv(public_data / "transactions.csv")
    stores = pd.read_csv(public_data / "stores.csv")
    holidays = pd.read_csv(public_data / "holidays.csv")
    oil = pd.read_csv(public_data / "oil.csv")
    query = "SELECT item_nbr, tid, unit_sales FROM read_parquet(?)"
    if limit is not None:
        query += f" LIMIT {int(limit)}"
    reader = duckdb.connect(database=":memory:")
    try:
        reader.execute("SET threads=1")
        columns = reader.execute(query, [str(fact_file)]).fetchnumpy()
    finally:
        reader.close()
    item_nbr = np.asarray(columns["item_nbr"]).astype(np.int64)
    tid = np.asarray(columns["tid"]).astype(np.int64)
    rows = len(item_nbr)
    if rows == 0 or len(np.unique(item_nbr * (int(tid.max()) + 1) + tid)) != rows:
        raise RuntimeError("fact rows are not unique")

    date = _lookup(transactions, "tid", "date")[tid]
    store = _lookup(transactions, "tid", "store_nbr")[tid]
    joined = {}
    for frame, key, index in ((items, "item_nbr", item_nbr), (transactions, "tid", tid),
                              (stores, "store_nbr", store), (holidays, "date", date), (oil, "date", date)):
        for column in frame.columns:
            if column not in ("item_nbr", "tid", "store_nbr", "date"):
                joined[column] = _lookup(frame, key, column)[index]
    target = (joined["f1"] + np.log2(joined["f2"]) - 10.0 * joined["f3"] - 10.0 * joined["f4"]
              + joined["f5"] * np.log2(joined["f5"]))
    sales = pd.DataFrame({
        "item_nbr": item_nbr,
        "unit_sales": np.asarray(columns["unit_sales"], dtype=float),
        "onpromotion": np.full(rows, 2, dtype=np.int64),
        "tid": tid,
        "Y": target,
    })
    features = pd.DataFrame({name: joined[name].astype(np.int32) for name in FULL_FEATURES})
    features["Y"] = target
    return sales, features


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fact", required=True, type=Path, help="a sales-dev-<i>.parquet file")
    parser.add_argument("--limit", type=int, default=None, help="keep only the first rows in file order")
    parser.add_argument("--out", required=True, type=Path, help="fact table CSV to write")
    parser.add_argument("--joined-out", type=Path, default=None, help="optional materialized-join CSV")
    args = parser.parse_args()
    sales, joined = load_sales(Path(__file__).resolve().parent, args.fact, args.limit)
    sales.to_csv(args.out, index=False)
    if args.joined_out is not None:
        joined.to_csv(args.joined_out, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
