import argparse
import json
import os
from pathlib import Path

from compute_edit_distance import load_scores as load_syntactic_scores
from compute_jaccard_similarity import load_scores as load_semantic_scores
from compute_vulnerability_ratio import load_scores as load_vulnerability_scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("method", type=str)
    parser.add_argument("output", type=Path)
    parser.add_argument("--data-dir", type=Path, default="new-splitted-data")
    parser.add_argument("--dataset-name", type=str, default="failed_test")
    parser.add_argument("--label", type=str, default=None)
    args = parser.parse_args()

    hunter_path = (
        args.data_dir
        / "results"
        / f"redos_hunter-{args.method}-{args.dataset_name}.jsonl"
    )
    source_pattern_path = (
        args.data_dir / "patterns" / f"{args.dataset_name}.txt"
    )
    target_pattern_path = (
        args.data_dir
        / "repaired-patterns"
        / f"{args.method}-{args.dataset_name}.txt"
    )
    source_matching_path = (
        args.data_dir
        / "full-matching"
        / "reference-patterns"
        / f"xeger-{args.method}-{args.dataset_name}.jsonl"
    )
    target_matching_path = (
        args.data_dir
        / "full-matching"
        / "reference-test-cases"
        / f"xeger-{args.method}-{args.dataset_name}.jsonl"
    )
    with (
        open(target_pattern_path, encoding="utf-8") as target_file,
        open(source_pattern_path, encoding="utf-8") as source_file,
    ):
        source_patterns = [e.rstrip("\r\n") for e in source_file]
        target_patterns = [e.rstrip("\r\n") for e in target_file]

    jaccard, precision, recall = load_semantic_scores(
        target_matching_path, source_matching_path
    )
    vulnerability = load_vulnerability_scores(hunter_path)
    compilability, _, _, edit_distance = load_syntactic_scores(
        target_pattern_path, source_pattern_path
    )

    os.makedirs(args.output.parent, exist_ok=True)
    if os.path.exists(args.output):
        with open(args.output, encoding="utf-8") as fp:
            dataset = json.load(fp)
    else:
        dataset = dict()

    normalized_edit_distance = [
        d / max(len(s), len(t)) if c else 1.0
        for c, d, s, t in zip(
            compilability,
            edit_distance,
            source_patterns,
            target_patterns,
            strict=True,
        )
    ]
    length_ratio = [
        len(t) / len(s) if c else 1.0
        for c, s, t in zip(
            compilability, source_patterns, target_patterns, strict=True
        )
    ]
    jaccard = [
        e if c else 0.0 for c, e in zip(compilability, jaccard, strict=True)
    ]
    precision = [
        e if c else 0.0 for c, e in zip(compilability, precision, strict=True)
    ]
    recall = [
        e if c else 0.0 for c, e in zip(compilability, recall, strict=True)
    ]
    update = {
        "original-pattern": source_patterns,
        "repaired-pattern": target_patterns,
        "jaccard": jaccard,
        "precision": precision,
        "recall": recall,
        "vulnerability": vulnerability,
        "compilability": compilability,
        "edit-distance": edit_distance,
        "normalized-edit-distance": normalized_edit_distance,
        "length_ratio": length_ratio,
    }
    label = args.method if args.label is None else args.label
    dataset.update({label: update})
    with open(args.output, "w", encoding="utf-8") as fp:
        json.dump(dataset, fp, indent=2)


if __name__ == "__main__":
    main()
