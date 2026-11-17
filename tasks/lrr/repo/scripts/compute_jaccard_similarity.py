"""Compute Jaccard distance"""

from dataclasses import dataclass
import json
from pathlib import Path

from compute_vulnerability_ratio import check_compilability


@dataclass(frozen=True)
class Result:
    attack_success: bool
    attack_time: int
    vulnerability_position: str
    attack_string: tuple[str, str, str]
    vulnerability_source: str
    vulnerability_degree: str


@dataclass(frozen=True)
class Input:
    pattern: str | None
    texts_from: str | None
    texts: list[str] | None
    results: list[bool | None] | None
    error: str | None = None


def load_scores(
    target_over_reference_path: Path, reference_over_target_path: Path
) -> tuple[list[float | None], list[float | None], list[float | None]]:
    with (
        open(target_over_reference_path, encoding="utf-8") as target_file,
        open(reference_over_target_path, encoding="utf-8") as reference_file,
    ):
        targets = (Input(**json.loads(e)) for e in target_file)
        references = (Input(**json.loads(e)) for e in reference_file)

        jaccard_scores = []
        precision_scores = []
        recall_scores = []
        for target, reference in zip(targets, references, strict=True):
            if target.pattern is None:
                jaccard_score = None
                precision_score = None
                recall_score = None
            elif not check_compilability(target.pattern):
                jaccard_score = None
                precision_score = None
                recall_score = None
            else:
                if target.results is None:
                    target_results = []
                else:
                    target_results = [
                        e for e in target.results if e is not None
                    ]
                if reference.results is None:
                    reference_results = []
                else:
                    reference_results = [
                        e for e in reference.results if e is not None
                    ]
                jaccard_score, precision_score, recall_score = (
                    compute_jaccard_precision_recall(
                        target_results, reference_results
                    )
                )
            jaccard_scores.append(jaccard_score)
            precision_scores.append(precision_score)
            recall_scores.append(recall_score)
    return jaccard_scores, precision_scores, recall_scores


def compute_jaccard_precision_recall(
    target_results: list[bool], reference_results: list[bool]
) -> tuple[float, float, float]:
    num_test_cases_in_union = len(target_results) + len(reference_results)
    num_test_cases_in_target = len(reference_results) + sum(target_results)
    num_test_cases_in_reference = len(target_results) + sum(reference_results)
    num_test_cases_in_intersection = sum(reference_results) + sum(
        target_results
    )

    if num_test_cases_in_union == 0:
        jaccard_score = 1.0
    else:
        jaccard_score = (
            num_test_cases_in_intersection / num_test_cases_in_union
        )

    if num_test_cases_in_target == 0:
        precision_score = 1.0
    else:
        precision_score = (
            num_test_cases_in_intersection / num_test_cases_in_target
        )
    if num_test_cases_in_reference == 0:
        recall_score = 1.0
    else:
        recall_score = (
            num_test_cases_in_intersection / num_test_cases_in_reference
        )
    return (jaccard_score, precision_score, recall_score)
