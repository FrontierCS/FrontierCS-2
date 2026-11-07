"""Parse Remedy results"""

import sys

from run_redos_hunter import Output as Input


def parse(text: str) -> str | None:
    try:
        return text.splitlines()[-1]
    except Exception:  # pylint: disable=broad-exception-caught
        return None


def main() -> None:
    for entry in map(Input.loads, sys.stdin):
        pattern: str | None
        if entry.result is None:
            pattern = None
        else:
            pattern = parse(entry.result)

        if pattern is None:
            print(flush=True)
        else:
            print(pattern, flush=True)


if __name__ == "__main__":
    main()
