"""Split dataset into scalpel-succeeded and -failed ones"""

import argparse
from dataclasses import asdict
import json
import logging
import os
from pathlib import Path
import random

from filter_vulnerable_patterns import check_vulnerablity
from parse_redos_hunter import Output as HunterData
from parse_regex_scalpel import Output as ScalpelData

# (
#     pattern,
#     (hunter_raw, hunter),
#     (scalpel_raw, scalpel),
#     repaired_pattern,
#     (hunter_scalpel-raw, hunter_scalpel),
# )
EntryType = tuple[
    str,
    tuple[str, HunterData],
    tuple[str, ScalpelData],
    str,
    tuple[str, HunterData],
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("hunter_raw", type=Path)
    parser.add_argument("hunter", type=Path)
    parser.add_argument("scalpel_raw", type=Path)
    parser.add_argument("scalpel", type=Path)
    parser.add_argument("hunter_scalpel_raw", type=Path)
    parser.add_argument("hunter_scalpel", type=Path)

    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--num-test", type=int, default=500)

    args = parser.parse_args()

    succeeded_dataset: list[EntryType] = []
    failed_dataset: list[EntryType] = []

    with (
        open(args.hunter_raw, encoding="utf-8") as hunter_raw_file,
        open(args.hunter, encoding="utf-8") as hunter_file,
        open(args.scalpel_raw, encoding="utf-8") as scalpel_raw_file,
        open(args.scalpel, encoding="utf-8") as scalpel_file,
        open(
            args.hunter_scalpel_raw, encoding="utf-8"
        ) as hunter_scalpel_raw_file,
        open(args.hunter_scalpel, encoding="utf-8") as hunter_scalpel_file,
    ):
        hunter_raw_dataset = map(json.loads, hunter_raw_file)
        hunter_dataset = map(HunterData.loads, hunter_file)
        scalpel_raw_dataset = map(json.loads, scalpel_raw_file)
        scalpel_dataset = map(ScalpelData.loads, scalpel_file)
        hunter_scalpel_raw_dataset = map(json.loads, hunter_scalpel_raw_file)
        hunter_scalpel_dataset = map(HunterData.loads, hunter_scalpel_file)

        for (
            hunter_raw_entry,
            hunter_entry,
            scalpel_raw_entry,
            scalpel_entry,
            hunter_scalpel_raw_entry,
            hunter_scalpel_entry,
        ) in zip(
            hunter_raw_dataset,
            hunter_dataset,
            scalpel_raw_dataset,
            scalpel_dataset,
            hunter_scalpel_raw_dataset,
            hunter_scalpel_dataset,
            strict=True,
        ):
            if hunter_entry.error is not None:
                continue
            if scalpel_entry.error is not None:
                continue
            if hunter_scalpel_entry.error is not None:
                continue

            assert hunter_entry.result is not None
            assert scalpel_entry.result is not None
            assert hunter_scalpel_entry.result is not None

            assert hunter_entry.pattern == scalpel_entry.pattern
            assert (
                scalpel_entry.result.final_regex
                == hunter_scalpel_entry.pattern
            )
            pattern = hunter_entry.pattern
            assert pattern is not None
            repaired_pattern = hunter_scalpel_entry.pattern

            if len(pattern) > args.max_length:
                continue

            if not check_vulnerablity(hunter_entry.result):
                continue

            if not check_vulnerablity(hunter_scalpel_entry.result):
                dataset = succeeded_dataset
            else:
                dataset = failed_dataset
            entry = (
                pattern,
                (hunter_raw_entry, hunter_entry),
                (scalpel_raw_entry, scalpel_entry),
                repaired_pattern,
                (hunter_scalpel_raw_entry, hunter_scalpel_entry),
            )
            dataset.append(entry)

    rand = random.Random(args.seed)
    rand.shuffle(succeeded_dataset)
    rand.shuffle(failed_dataset)

    patterns_dir = args.output_dir / "patterns"
    raw_outputs_dir = args.output_dir / "raw-ouputs"
    results_dir = args.output_dir / "results"
    repaired_patterns_dir = args.output_dir / "repaired-patterns"

    os.makedirs(patterns_dir, exist_ok=True)
    os.makedirs(raw_outputs_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(repaired_patterns_dir, exist_ok=True)

    logging.warning(len(succeeded_dataset))
    logging.warning(len(failed_dataset))

    for succeeded, full_dataset in [
        ("succeeded", succeeded_dataset),
        ("failed", failed_dataset),
    ]:
        for test in ["test", "train"]:
            if test == "test":
                dataset = full_dataset[: args.num_test]
            else:
                dataset = full_dataset[args.num_test :]

            patterns_path = patterns_dir / f"{succeeded}_{test}.txt"
            hunter_raw_path = (
                raw_outputs_dir / f"redos_hunter-{succeeded}_{test}.jsonl"
            )
            hunter_path = (
                results_dir / f"redos_hunter-{succeeded}_{test}.jsonl"
            )
            scalpel_raw_path = (
                raw_outputs_dir / f"regex_scalpel-{succeeded}_{test}.jsonl"
            )
            scalpel_path = (
                results_dir / f"regex_scalpel-{succeeded}_{test}.jsonl"
            )
            repaired_patterns_path = (
                repaired_patterns_dir / f"regex_scalpel-{succeeded}_{test}.txt"
            )
            hunter_scalpel_raw_path = (
                raw_outputs_dir
                / f"redos_hunter-regex_scalpel-{succeeded}_{test}.jsonl"
            )
            hunter_scalpel_path = (
                results_dir
                / f"redos_hunter-regex_scalpel-{succeeded}_{test}.jsonl"
            )

            with (
                open(patterns_path, "w", encoding="utf-8") as patterns_file,
                open(
                    hunter_raw_path, "w", encoding="utf-8"
                ) as hunter_raw_file,
                open(hunter_path, "w", encoding="utf-8") as hunter_file,
                open(
                    scalpel_raw_path, "w", encoding="utf-8"
                ) as scalpel_raw_file,
                open(scalpel_path, "w", encoding="utf-8") as scalpel_file,
                open(
                    repaired_patterns_path, "w", encoding="utf-8"
                ) as repaired_patterns_file,
                open(
                    hunter_scalpel_raw_path, "w", encoding="utf-8"
                ) as hunter_scalpel_raw_file,
                open(
                    hunter_scalpel_path, "w", encoding="utf-8"
                ) as hunter_scalpel_file,
            ):
                for (
                    pattern,
                    (hunter_raw, hunter),
                    (scalpel_raw, scalpel),
                    repaired_pattern,
                    (hunter_scalpel_raw, hunter_scalpel),
                ) in dataset:
                    print(pattern, file=patterns_file)
                    print(json.dumps(hunter_raw), file=hunter_raw_file)
                    print(json.dumps(asdict(hunter)), file=hunter_file)
                    print(json.dumps(scalpel_raw), file=scalpel_raw_file)
                    print(json.dumps(asdict(scalpel)), file=scalpel_file)
                    print(repaired_pattern, file=repaired_patterns_file)
                    print(
                        json.dumps(hunter_scalpel_raw),
                        file=hunter_scalpel_raw_file,
                    )
                    print(
                        json.dumps(asdict(hunter_scalpel)),
                        file=hunter_scalpel_file,
                    )


if __name__ == "__main__":
    main()
