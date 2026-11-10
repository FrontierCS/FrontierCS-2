# Platform contract fixture

`run-metrics-v1.schema.json` is copied verbatim from
the platform source's `docs/contract/schemas/run-metrics-v1.schema.json` at
commit `c71e0adce851b73e3c442f319fc52282a1ba3f1e`. It is a platform contract
fixture, not a task-authored approximation. Update it only when intentionally
adopting a changed platform contract, then review the closed-envelope regression
tests and the Driver runtime's stricter integer-representation validation.
