"""Judge-owned DBSCAN label validation and permutation-invariant agreement."""
from collections import Counter
from math import isfinite


def validate_labels(labels: list[int], count: int) -> None:
    if not isinstance(labels, list) or len(labels) != count:
        raise ValueError("DBSCAN must return one label per input point")
    if any(type(label) is not int or label < -1 for label in labels):
        raise ValueError("DBSCAN labels must be integers >= 0 or noise -1")


def adjusted_rand_index(left: list[int], right: list[int]) -> float:
    """Match the legacy ARI, including treating noise as one ordinary group.

    Sparse contingency counts avoid quadratic storage when many labels differ.
    The noise marker's semantic identity is separate from label permutation.
    """
    validate_labels(left, len(right))
    validate_labels(right, len(left))
    if len(left) < 2:
        return 1.0
    choose2 = lambda count: count * (count - 1) // 2
    joint = sum(choose2(count) for count in Counter(zip(left, right)).values())
    rows = sum(choose2(count) for count in Counter(left).values())
    columns = sum(choose2(count) for count in Counter(right).values())
    pairs = choose2(len(left))
    # Integer arithmetic avoids cancellation for identical near-degenerate partitions.
    numerator = 2 * (joint * pairs - rows * columns)
    denominator = (rows + columns) * pairs - 2 * rows * columns
    result = numerator / denominator if denominator else 1.0
    if not isfinite(result) or not -1.0 <= result <= 1.0:
        raise ValueError("invalid adjusted Rand index")
    return result


def noise_mask_agreement(left: list[int], right: list[int]) -> float:
    validate_labels(left, len(right))
    validate_labels(right, len(left))
    if not left:
        return 1.0
    return sum((a == -1) == (b == -1) for a, b in zip(left, right)) / len(left)


def noise_mask_mismatch(left: list[int], right: list[int]) -> float:
    validate_labels(left, len(right))
    validate_labels(right, len(left))
    if not left:
        return 0.0
    return sum((a == -1) != (b == -1) for a, b in zip(left, right)) / len(left)
