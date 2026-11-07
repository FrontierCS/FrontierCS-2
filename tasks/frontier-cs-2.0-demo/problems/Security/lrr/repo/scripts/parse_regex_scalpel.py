"""Parse RegexScalpel results"""

from dataclasses import asdict
from dataclasses import dataclass
import json
import logging
import re
import sys
from typing import Self


@dataclass(frozen=True)
class Input:
    pattern: str | None
    result: str | None = None
    error: str | None = None

    @classmethod
    def loads(cls, text: str) -> Self:
        return cls(**json.loads(text))


@dataclass(frozen=True)
class RepairStep:
    """Repair Step"""

    nested_quantifier_counts: int | None
    overlapping_disjunction_counts: int | None
    overlapping_adjacent_counts: int | None
    large_quantifier_start_counts: int | None
    subpatterns: list[str] | None
    nested_quantifier_result: bool | None
    overlapping_disjunction_result: bool | None
    overlapping_adjacent_result: bool | None
    large_quantifier_start_result: bool | None
    repaired: str | None
    overlapping_adjacent_repair: list[str] | None
    large_quantifier_start_repair: list[str] | None


@dataclass(frozen=True)
class Result:
    regex: str
    final_regex: str
    repairs: list[RepairStep]


@dataclass(frozen=True)
class Output:
    """ScalpelData"""

    pattern: str | None
    result: Result | None = None
    error: str | None = None

    @classmethod
    def loads(cls, text: str) -> Self:
        parsed = json.loads(text)
        pattern = parsed["pattern"]
        error = parsed["error"]
        if parsed["result"] is None:
            result = None
        else:
            regex = parsed["result"]["regex"]
            final_regex = parsed["result"]["final_regex"]
            repairs = [
                RepairStep(**repair) for repair in parsed["result"]["repairs"]
            ]
            result = Result(
                regex=regex,
                final_regex=final_regex,
                repairs=repairs,
            )

        return cls(pattern=pattern, result=result, error=error)


def find_all_subpatterns(content: str) -> list[str]:
    """Finds all subpatterns marked with ▶▷ and ◁◀ in a string."""
    subpatterns = []
    stack = []
    i = 0
    while i < len(content):
        if content[i : i + 2] == "▶▷":
            stack.append(i + 2)
            i += 2
        elif content[i : i + 2] == "◁◀":
            if stack:
                start_index = stack.pop()
                if (
                    not stack
                ):  # Only add the subpattern if we are at the top level
                    subpatterns.append(content[start_index:i])
            i += 2
        else:
            i += 1
    return subpatterns


def parse_block(block: str) -> RepairStep:
    lines = block.splitlines()

    nested_quantifier_counts: int | None = None
    overlapping_disjunction_counts: int | None = None
    overlapping_adjacent_counts: int | None = None
    large_quantifier_start_counts: int | None = None
    subpatterns: list[str] | None = None
    nested_quantifier_result: bool | None = None
    overlapping_disjunction_result: bool | None = None
    overlapping_adjacent_result: bool | None = None
    large_quantifier_start_result: bool | None = None
    repaired: str | None = None
    overlapping_adjacent_repair: list[str] | None = None
    large_quantifier_start_repair: list[str] | None = None

    # Extract counts
    nested_quantifier_counts = int(lines[0].removeprefix('NQ\u68c0\u6d4b\u7ed3\u679c\u6570\u91cf\uff1a'))

    qod_counts_match = re.search('QOD\u68c0\u6d4b\u7ed3\u679c\u6570\u91cf\uff1a(\\d+)', block)
    if qod_counts_match:
        overlapping_disjunction_counts = int(qod_counts_match.group(1))

    qoa_counts_match = re.search('QOA\u68c0\u6d4b\u7ed3\u679c\u6570\u91cf\uff1a(\\d+)', block)
    if qoa_counts_match:
        overlapping_adjacent_counts = int(qoa_counts_match.group(1))

    slq_counts_match = re.search('SLQ\u68c0\u6d4b\u7ed3\u679c\u6570\u91cf\uff1a(\\d+)', block)
    if slq_counts_match:
        large_quantifier_start_counts = int(slq_counts_match.group(1))

    # Extract subpatterns
    subpattern_line = next((l for l in lines if "▶▷" in l and "◁◀" in l), None)
    if subpattern_line:
        subpatterns = find_all_subpatterns(subpattern_line)

    nq_result_match = re.search(r"NQResult: (true|false)", block)
    if nq_result_match:
        nested_quantifier_result = nq_result_match.group(1) == "true"

    qod_result_match = re.search(r"QODResult: (true|false)", block)
    if qod_result_match:
        overlapping_disjunction_result = qod_result_match.group(1) == "true"

    qoa_result_match = re.search(r"QOAResult: (true|false)", block)
    if qoa_result_match:
        overlapping_adjacent_result = qoa_result_match.group(1) == "true"

    slq_result_match = re.search(r"SLQResult: (true|false)", block)
    if slq_result_match:
        large_quantifier_start_result = slq_result_match.group(1) == "true"

    repaired_match = re.search(r"repaired: (.*)", block)
    if repaired_match:
        repaired = repaired_match.group(1).strip()

    qoa_repair_match = re.search(r"QOARepair: \[(.*)\]", block, re.DOTALL)
    if qoa_repair_match:
        repair_str = qoa_repair_match.group(1)
        items = [r.split("=r")[0].strip() for r in repair_str.split(",")]
        if items:
            items[0] = items[0].lstrip("^")
        overlapping_adjacent_repair = items

    slq_repair_match = re.search(r"SLQRepair: \[(.*)\]", block, re.DOTALL)
    if slq_repair_match:
        repair_str = slq_repair_match.group(1)
        items = [r.split("=r")[0].strip() for r in repair_str.split(",")]
        if items:
            items[0] = items[0].lstrip("^")
        large_quantifier_start_repair = items

    return RepairStep(
        nested_quantifier_counts=nested_quantifier_counts,
        overlapping_disjunction_counts=overlapping_disjunction_counts,
        overlapping_adjacent_counts=overlapping_adjacent_counts,
        large_quantifier_start_counts=large_quantifier_start_counts,
        subpatterns=subpatterns,
        nested_quantifier_result=nested_quantifier_result,
        overlapping_disjunction_result=overlapping_disjunction_result,
        overlapping_adjacent_result=overlapping_adjacent_result,
        large_quantifier_start_result=large_quantifier_start_result,
        repaired=repaired,
        overlapping_adjacent_repair=overlapping_adjacent_repair,
        large_quantifier_start_repair=large_quantifier_start_repair,
    )


def parse(text: str) -> Result:
    lines = text.splitlines()

    regex = lines[1].removeprefix("regex: ")
    final_regex = lines[-1].removeprefix("final regex: ")

    blocks = "\n".join(lines[4:-1]).split("\n\n")
    repairs = []
    for block in blocks:
        if not block.strip():
            continue
        repair = parse_block(block)
        repairs.append(repair)
    return Result(regex=regex, final_regex=final_regex, repairs=repairs)


def verify_and_print_results(entry: Input) -> None:
    if not entry.result:
        return

    result_text = entry.result
    lines = result_text.splitlines()

    original_regex_line = next(
        (l for l in lines if l.startswith("regex:")), None
    )
    if not original_regex_line:
        return

    original_regex = original_regex_line.removeprefix("regex: ").strip()
    print(f"\n--- Entry: Original Regex: `{original_regex}` ---")

    current_regex_context = original_regex
    repair_blocks = result_text.split("\n\n")

    for block_num, block in enumerate(repair_blocks, 1):
        block_lines = block.splitlines()
        subpattern_line = next(
            (l for l in block_lines if "▶▷" in l and "◁◀" in l), None
        )

        if subpattern_line:
            print(f"  [Block {block_num}]")
            reconstructed_regex = subpattern_line.replace("▶▷", "").replace(
                "◁◀", ""
            )

            if reconstructed_regex.strip() != current_regex_context.strip():
                print(  # pylint: disable-next=line-too-long
                    "    Context Mismatch: The reconstructed regex from this step does not match the expected context."
                )
                print(f"      Expected: `{current_regex_context.strip()}`")
                print(f"      Found:    `{reconstructed_regex.strip()}`")

            subpatterns = find_all_subpatterns(subpattern_line)
            if not subpatterns:
                print(  # pylint: disable-next=line-too-long
                    "    No subpatterns found in this line, though markers were present."
                )

            for subpattern in subpatterns:
                is_substring = subpattern in reconstructed_regex
                verification_status = "OK" if is_substring else "FAIL"
                print(  # pylint: disable-next=line-too-long
                    f"    - Extracted: '{subpattern}' -> Verification: {verification_status}"
                )

        repaired_match = re.search(r"repaired: (.*)", block)
        if repaired_match:
            current_regex_context = repaired_match.group(1).strip()


def main() -> None:
    verify = "--verify" in sys.argv
    if verify:
        # Can't read from stdin if we might need it twice
        stdin_lines = sys.stdin.readlines()
        for i, line in enumerate(stdin_lines, 1):
            try:
                entry = Input.loads(line)
                verify_and_print_results(entry)
            except Exception as e:  # pylint: disable=broad-exception-caught
                logging.warning("Line %d: %s", i, e)
        return

    for i, entry in enumerate(map(Input.loads, sys.stdin), 1):
        try:
            if entry.result is None:
                raise ValueError("Result is None")
            output = Output(pattern=entry.pattern, result=parse(entry.result))
        except Exception as e:  # pylint: disable=broad-exception-caught
            logging.warning("Line %d: %s", i, e)
            output = Output(pattern=entry.pattern, error=str(e))
        print(json.dumps(asdict(output)), flush=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
