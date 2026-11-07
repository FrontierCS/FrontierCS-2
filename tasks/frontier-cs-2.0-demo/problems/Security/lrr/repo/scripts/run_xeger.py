import argparse
from dataclasses import asdict
from dataclasses import dataclass
import json
import logging
import re
import sys

import timeout_decorator  # type: ignore
from tqdm import tqdm
from xeger import Xeger  # type: ignore


@dataclass
class Output:
    pattern: str | None
    texts: list[str] | None


def safe_match(pattern: str, text: str, timeout: int) -> bool | None:
    timeout_match = timeout_decorator.timeout(timeout, TimeoutError)(
        re.fullmatch
    )
    try:
        return timeout_match(pattern, text) is not None
    except Exception:  # pylint: disable=broad-exception-caught
        return None


def safe_search(pattern: str, text: str, timeout: int) -> bool | None:
    timeout_match = timeout_decorator.timeout(timeout, TimeoutError)(re.search)
    try:
        return timeout_match(pattern, text) is not None
    except Exception:  # pylint: disable=broad-exception-caught
        return None


def get_partial_pattern(pattern: str) -> str:
    partial_pattern = f"(?:{pattern})"
    if pattern[0] != "^":
        partial_pattern = ".*" + partial_pattern
    if pattern[-1] != "$":
        partial_pattern += ".*"
    return partial_pattern


def main() -> None:
    # total_timeout: timeout per pattern
    total_timeout = 60
    # total_timeout: timeout per text
    partial_timeout = 1
    max_num_texts = 100

    parser = argparse.ArgumentParser()
    parser.add_argument("--partial", action="store_true")
    args = parser.parse_args()

    patterns = map(lambda e: e.rstrip("\r\n"), sys.stdin)
    x = Xeger(seed=42)
    for i, pattern in tqdm(enumerate(patterns, 1)):
        if not pattern:
            output = Output(None, None)
        else:
            try:
                if args.partial:
                    pattern_for_text = get_partial_pattern(pattern)
                    test_func = safe_search
                else:
                    pattern_for_text = pattern
                    test_func = safe_match

                untested_texts = [
                    x.xeger(pattern_for_text) for _ in range(max_num_texts)
                ]
                # XXX: We treat a language as a multi-set
                # untested_texts = list(set(untested_texts))

                remaining_total_timeout = total_timeout
                cache: dict[str, bool | None] = dict()
                texts = []
                for text in untested_texts:
                    if text in cache:
                        test_result = cache[text]
                    else:
                        timeout = min(remaining_total_timeout, partial_timeout)
                        test_result = None
                        if timeout > 0:
                            test_result = test_func(pattern, text, timeout)
                        if test_result is None:
                            remaining_total_timeout -= timeout

                    if test_result is True:
                        texts.append(text)

                if not texts:
                    logging.warning("Line %d: empty test cases", i)
            except Exception:  # pylint: disable=broad-exception-caught
                texts = None
            output = Output(pattern, texts)
        print(json.dumps(asdict(output)), flush=True)


if __name__ == "__main__":
    main()
