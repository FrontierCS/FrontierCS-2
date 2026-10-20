import argparse
from dataclasses import asdict
from dataclasses import dataclass
import json
from pathlib import Path

from run_xeger import Output as TestCaseInput
from run_xeger import safe_match
from run_xeger import safe_search
from tqdm import tqdm


@dataclass
class Output:
    pattern: str | None
    texts_from: str | None
    texts: list[str] | None
    results: list[bool | None] | None
    error: str | None = None


def main() -> None:
    total_timeout = 60  # timeout per pattern
    partial_timeout = 1  # timeout per text

    parser = argparse.ArgumentParser()
    parser.add_argument("--partial", action="store_true")
    parser.add_argument("pattern", type=Path)
    parser.add_argument("test_case", type=Path)
    args = parser.parse_args()

    with (
        open(args.pattern, encoding="utf-8") as pattern_file,
        open(args.test_case, encoding="utf-8") as test_case_file,
    ):
        patterns = (e.rstrip("\r\n") for e in pattern_file)
        dataset = (TestCaseInput(**json.loads(e)) for e in test_case_file)
        for _, (pattern, entry) in tqdm(
            enumerate(zip(patterns, dataset, strict=True), 1)
        ):
            if not pattern:
                output = Output(
                    None, entry.pattern, entry.texts, None, error="no patter"
                )
            if entry.texts is None:
                output = Output(
                    pattern,
                    entry.pattern,
                    entry.texts,
                    None,
                    error="no test cases",
                )
            else:
                if args.partial:
                    test_func = safe_search
                else:
                    test_func = safe_match
                results = []

                remaining_total_timeout = total_timeout
                cache: dict[str, bool | None] = dict()
                for text in entry.texts:
                    if text in cache:
                        test_result = cache[text]
                    else:
                        timeout = min(remaining_total_timeout, partial_timeout)
                        test_result = None
                        if timeout > 0:
                            test_result = test_func(pattern, text, timeout)
                        if test_result is None:
                            remaining_total_timeout -= timeout
                    results.append(test_result)
                output = Output(pattern, entry.pattern, entry.texts, results)
            print(json.dumps(asdict(output)), flush=True)


if __name__ == "__main__":
    main()
