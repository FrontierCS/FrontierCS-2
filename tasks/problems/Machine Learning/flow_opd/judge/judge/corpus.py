"""Real upstream corpus hygiene; no sampling of final workloads at import time."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path


def prompt_key(prompt):
    return ' '.join(prompt.lower().split())


def target_key(prompt):
    parts = prompt.split('"')
    if len(parts) < 3:
        raise ValueError('missing quoted OCR target')
    target = parts[1].replace(' ', '').lower()
    if not target:
        raise ValueError('empty OCR target')
    return target


def clean(lines):
    seen, result = set(), []
    for line in lines:
        text = line.rstrip('\r\n')
        try:
            target_key(text)
        except ValueError:
            continue
        key = prompt_key(text)
        if key not in seen:
            result.append(text)
            seen.add(key)
    return result


def load_corpus(root):
    root = Path(root)
    train = clean((root/'train.txt').read_text().splitlines())
    test = clean((root/'test.txt').read_text().splitlines())
    held_prompts = {prompt_key(p) for p in test}
    held_targets = {target_key(p) for p in test}
    train = [p for p in train if prompt_key(p) not in held_prompts and target_key(p) not in held_targets]
    return train, test


def stratum(group):
    # Challenge strata, not paper constants. Whole target groups cannot cross splits.
    t = target_key(group[0])
    length_bucket = min(len(t)//8, 8)
    words_bucket = min(len(group[0].split('"')[1].split()), 5)
    context_bucket = min(max(len(p.split()) for p in group)//16, 8)
    return (length_bucket, words_bucket, context_bucket, any(c.isdigit() for c in t), not t.isascii())


def partition_test(prompts, seed):
    """Candidate partitioning utility; run/freeze only after qualification.

    Balances groups within each difficulty stratum. Sparse strata and actual
    distribution balance must be audited before choosing suite cardinalities.
    No reward-dependent selection is permitted.
    """
    groups = defaultdict(list)
    for prompt in prompts:
        groups[target_key(prompt)].append(prompt)
    buckets = defaultdict(list)
    for target, group in groups.items():
        buckets[stratum(group)].append((target, group))
    splits = {name: [] for name in ('development', 'train_grading', 'final')}
    counts = Counter()
    for key in sorted(buckets):
        local = Counter()
        items = sorted(buckets[key], key=lambda x: hashlib.sha256(f'{seed}\0{x[0]}'.encode()).digest())
        for _, group in items:
            name = min(splits, key=lambda n: (local[n], counts[n], n))
            splits[name].extend(group)
            local[name] += len(group)
            counts[name] += len(group)
    return splits


def audit(root):
    root = Path(root)
    raw_train = (root/'train.txt').read_text().splitlines()
    raw_test = (root/'test.txt').read_text().splitlines()
    train, test = load_corpus(root)
    return {'raw_train_rows':len(raw_train), 'raw_test_rows':len(raw_test),
            'unique_train_prompts':len(clean(raw_train)), 'unique_test_prompts':len(test),
            'eligible_train_prompts':len(train), 'heldout_target_groups':len({target_key(p) for p in test}),
            'training_prompt_overlap':len({prompt_key(p) for p in train}&{prompt_key(p) for p in test}),
            'training_target_overlap':len({target_key(p) for p in train}&{target_key(p) for p in test}),
            'frozen_suite':False}

if __name__ == '__main__':
    print(json.dumps(audit(Path(__file__).parent/'corpus'), indent=2))
