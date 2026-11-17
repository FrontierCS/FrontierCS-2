"""Filter vulnerable patterns from ReDoSHunter outputs"""

import sys

from parse_redos_hunter import Output as Input
from parse_redos_hunter import Result


def check_vulnerablity(result: list[Result]) -> bool:
    try:
        return any(e.attack_success for e in result)
    except Exception:  # pylint: disable=broad-exception-caught
        return False


def main() -> None:
    dataset = map(Input.loads, sys.stdin)
    for entry in dataset:
        if entry.result is None:
            continue
        if check_vulnerablity(entry.result):
            print(entry.pattern, flush=True)


if __name__ == "__main__":
    main()
