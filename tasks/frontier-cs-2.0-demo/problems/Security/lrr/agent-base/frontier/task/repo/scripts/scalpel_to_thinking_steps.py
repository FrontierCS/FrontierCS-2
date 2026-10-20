"""Public policy hooks used by the upstream prompt constructor."""

from parse_regex_scalpel import Result as ScalpelResult


def subpattern_to_string(subpattern: str) -> str:
    """Render one first-step symbolic localization for the repair prompt."""
    raise NotImplementedError("implement the localization policy")


def generate_cot_prompt(result: ScalpelResult) -> tuple[str, list[str]]:
    """Render one localization trajectory without exposing the upstream policy."""
    raise NotImplementedError("implement the localization policy")
