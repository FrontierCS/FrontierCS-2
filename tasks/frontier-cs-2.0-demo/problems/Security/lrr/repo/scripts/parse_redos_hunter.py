"""Parse ReDoSHunter results"""

from dataclasses import asdict
from dataclasses import dataclass
import json
import logging
import re
import sys
from typing import Self

from run_redos_hunter import Output as Input
from utils import batched


@dataclass(frozen=True)
class Result:
    attack_success: bool
    attack_time: int
    vulnerability_position: str
    attack_string: tuple[str, str, str]
    vulnerability_source: str
    vulnerability_degree: str


@dataclass(frozen=True)
class Output:
    pattern: str | None
    result: list[Result] | None = None
    error: str | None = None

    @classmethod
    def loads(cls, text: str) -> Self:
        parsed = json.loads(text)
        pattern = parsed["pattern"]
        error = parsed["error"]
        if parsed["result"] is None:
            result = None
        else:
            result = list(Result(**e) for e in parsed["result"])
        return cls(pattern=pattern, result=result, error=error)


def parse_attack_string(attack_string: str) -> tuple[str, str, str]:
    compiled = re.compile(r'"(.*?)"\+"(.*?)"\*(?:\d+)\+"(.*)"')
    matched = compiled.fullmatch(attack_string)
    if matched is None:
        raise ValueError(f"{attack_string} is invalid attack string")
    prefix = matched[1]
    pump = matched[2]
    suffix = matched[3]
    return (prefix, pump, suffix)


def parse_batch(batch: tuple[str, str, str, str, str, str]) -> Result:
    attack_success = batch[0].removeprefix("Is attack success: ") == "true"
    attack_time = int(
        batch[1].removeprefix("Attack time: ").removesuffix(" (ms)")
    )
    vulnerability_position = batch[2].removeprefix("Vulnerability Position: ")
    attack_string = parse_attack_string(
        batch[3].removeprefix("Attack String: ")
    )
    vulnerability_source = batch[4].removeprefix("Vulnerability Source: ")
    vulnerability_degree = batch[5]

    return Result(
        attack_success=attack_success,
        attack_time=attack_time,
        vulnerability_position=vulnerability_position,
        attack_string=attack_string,
        vulnerability_source=vulnerability_source,
        vulnerability_degree=vulnerability_degree,
    )


def parse(text: str) -> list[Result]:
    result = []
    lines = text.splitlines()
    for batch in batched(lines[3:], 6):
        result.append(parse_batch(batch))  # type: ignore

    return result


def main() -> None:
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
    main()
