from pathlib import Path

from compute_vulnerability_ratio import check_compilability
import editdistance  # type: ignore


def load_scores(
    target_path: Path,
    reference_path: Path,
) -> tuple[list[bool], list[int], list[int], list[int]]:
    compilabilities: list[bool] = []
    target_lengths: list[int] = []
    reference_lengths: list[int] = []
    edit_distances: list[int] = []
    with (
        open(target_path, encoding="utf-8") as target_file,
        open(reference_path, encoding="utf-8") as reference_file,
    ):
        targets = (e.rstrip("\r\n") for e in target_file)
        references = (e.rstrip("\r\n") for e in reference_file)
        for target, reference in zip(targets, references, strict=True):
            if not target:
                compilability = False
            else:
                compilability = check_compilability(target)

            compilabilities.append(compilability)
            target_lengths.append(len(target))
            reference_lengths.append(len(reference))
            edit_distances.append(editdistance.distance(target, reference))
    return compilabilities, target_lengths, reference_lengths, edit_distances
