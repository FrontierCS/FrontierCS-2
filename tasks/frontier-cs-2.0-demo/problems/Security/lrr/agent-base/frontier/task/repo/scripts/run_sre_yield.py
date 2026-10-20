import argparse
from dataclasses import asdict
from dataclasses import dataclass
from itertools import islice
import json
import logging
import sys

from run_xeger import safe_match
from run_xeger import safe_search
import sre_yield  # type: ignore
import timeout_decorator  # type: ignore
from tqdm import tqdm


@dataclass
class Output:
    pattern: str | None
    texts: list[str] | None


def safe_sre_yield(pattern: str, n: int, timeout: int) -> list[str]:
    texts: list[str] = []
    it_texts = sre_yield.AllStrings(pattern)

    @timeout_decorator.timeout(0.001)
    def timeout_sre_yield() -> None:
        for text in islice(it_texts, n):
            texts.append(text)

    try:
        timeout_sre_yield()
    except Exception:
        logging.warning("?")
        pass
    return texts


def main() -> None:
    timeout = 1
    max_num_texts = 100

    parser = argparse.ArgumentParser()
    parser.add_argument("--partial", action="store_true")
    args = parser.parse_args()

    patterns = map(lambda e: e.rstrip("\r\n"), sys.stdin)
    for i, pattern in tqdm(enumerate(patterns, 1)):
        if not pattern:
            output = Output(None, None)
        else:
            try:
                if args.partial:
                    texts = safe_sre_yield(
                        f".*(:?{pattern}).*", max_num_texts, timeout
                    )
                else:
                    texts = safe_sre_yield(pattern, max_num_texts, timeout)
                texts = list(set(texts))
                if args.partial:
                    texts = [
                        text
                        for text in texts
                        if safe_search(pattern, text, timeout) is True
                    ]
                else:
                    texts = [
                        text
                        for text in texts
                        if safe_match(pattern, text, timeout) is True
                    ]
                if not texts:
                    logging.warning("Line %d: empty test cases", i)
            except Exception:  # pylint: disable=broad-exception-caught
                texts = None
            output = Output(pattern, texts)
        print(json.dumps(asdict(output)), flush=True)


if __name__ == "__main__":
    main()
