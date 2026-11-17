"""Get RegexScalpel results"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from dataclasses import dataclass
from itertools import tee
import json
import logging
import subprocess
import sys
from typing import Iterable

from tqdm import tqdm
from utils import retry_on_blockingio


@dataclass
class Output:
    pattern: str | None
    result: str | None = None
    error: str | None = None

    def __str__(self) -> str:
        return json.dumps(asdict(self))


@retry_on_blockingio()
def job(pattern: str, timeout: int) -> str | Exception:
    cmd = ["java", "-jar", "target/RedosDetector-1.0-SNAPSHOT.jar", pattern]
    cwd = "third-party/RegexScalpel/"
    try:
        return subprocess.run(
            cmd,
            capture_output=True,
            check=True,
            cwd=cwd,
            encoding="utf-8",
            text=True,
            timeout=timeout,
        ).stdout
    except Exception as e:  # pylint: disable=broad-exception-caught
        return e


def main() -> None:
    max_workers = 4
    timeout = 60

    logging.debug("Number of workers: %d", max_workers)

    patterns: Iterable[str] = map(lambda e: e.rstrip("\r\n"), sys.stdin)
    patterns, forked_patterns = tee(patterns, 2)

    with ThreadPoolExecutor(max_workers) as executor:
        completed_processes = executor.map(
            lambda e: job(e, timeout), forked_patterns
        )
        for i, (pattern, maybe_result) in tqdm(
            enumerate(zip(patterns, completed_processes), 1)
        ):
            if not pattern:
                output = Output(pattern=None, error="Empty pattern")
            if isinstance(maybe_result, Exception):
                error = str(maybe_result)
                logging.warning("Line %d: %s", i, error)
                output = Output(pattern, error=error)
            else:
                output = Output(pattern, result=maybe_result)
            print(str(output), flush=True)


if __name__ == "__main__":
    main()
