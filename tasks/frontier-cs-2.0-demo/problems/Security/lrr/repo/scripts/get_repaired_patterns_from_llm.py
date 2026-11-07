"""Get patterns from LLM ouptuts"""

import json
import logging
import sys

from inference import Output as Input


def extract_fixed_pattern_from_answer(answer: str) -> str | None:
    blocks = answer.split("```")
    if len(blocks) < 2:
        return None
    block = blocks[-2]
    for pattern in reversed(block.split("\n")):
        if len(pattern) > 0:
            return pattern
    return None


def main() -> None:
    dataset = (Input(**json.loads(e)) for e in sys.stdin)
    for i, entry in enumerate(dataset):
        fixed_pattern = extract_fixed_pattern_from_answer(entry.answer)
        if fixed_pattern is None:
            logging.warning("Line %d: failed to parse model output", i)
            print(flush=True)
        elif "\n" in fixed_pattern:
            logging.warning("Line %d: pattern with newline", i)
            print(flush=True)
        else:
            print(fixed_pattern, flush=True)


if __name__ == "__main__":
    main()
