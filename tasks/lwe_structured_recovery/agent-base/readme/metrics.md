- `instance_count`: Number of instances in the public catalog.
- `solved_count`: Distinct catalog instances whose unique admissible ledger record passes every
  validity predicate.
- `submitted_count`: Ledger records read, including records rejected as malformed.
- `invalid_count`: Rejected record occurrences: malformed, duplicated, unknown or failing a
  predicate.
- `duplicate_count`: Distinct syntactically valid instance IDs that occur more than once in the
  ledger.
- `conflict_count`: Distinct instance IDs that carry more than one distinct syntactically valid
  vector.
- `unknown_count`: Distinct syntactically valid instance IDs that are not in the public catalog.
- `rejection_code_counts`: Number of rejected record occurrences per stable rejection code.
