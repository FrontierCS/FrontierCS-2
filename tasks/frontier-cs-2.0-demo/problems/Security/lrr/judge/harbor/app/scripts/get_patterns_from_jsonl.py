"""Get patterns from jsonl"""

import argparse
import json
import logging
import sys
from typing import Any


def extract_from_closed_llm(entry: dict[str, Any], i: int = -1) -> str | None:
    answer: str | None = entry.get("answer", None)

    if answer is None:
        logging.warning("Line %d: No pattern in object", i)
        return ""

    return answer.split("```")[-2].strip("\r\n")


def extract_from_sl_regex(entry: dict[str, Any], i: int = -1) -> str | None:
    pattern: str | None = entry.get("pattern", None)

    if pattern is None:
        logging.warning("Line %d: No pattern in object", i)
        return ""

    if "\n" in pattern:
        logging.warning("Line %d: Exclude pattern with new-line", i)
        return ""

    return pattern


def extract_from_regex_scalpel(
    entry: dict[str, Any], i: int = -1
) -> str | None:
    logging.warning(entry)
    result = entry.get("result", None)

    if result is None:
        return None

    pattern: str | None = result.get("final_regex", None)

    if pattern is None:
        return None

    if "\n" in pattern:
        logging.warning("Line %d: Exclude pattern with new-line", i)
        pattern = None

    return pattern


def main() -> None:
    parser = argparse.ArgumentParser()

    format_to_extract = {
        "sl_regex": extract_from_sl_regex,
        "regex_scalpel": extract_from_regex_scalpel,
        "closed_llm": extract_from_closed_llm,
    }

    parser.add_argument(
        "--format", choices=list(format_to_extract), default="sl_regex"
    )
    args = parser.parse_args()
    extract = format_to_extract[args.format]

    dataset = map(json.loads, sys.stdin)
    for i, entry in enumerate(dataset):
        logging.warning(entry)
        line = extract(entry, i)
        print(line, flush=True)


if __name__ == "__main__":
    main()
