# Round 4 repair report

The homogeneous six-configuration benchmark was replaced by six distinct tests from byte-identical upstream `tests/math.rs`: `math_ematching_bench`, `diff_power_harder`, `math_associate_adds`, `integ_part2`, `math_simplify_root`, and `math_simplify_factor`. They cover multi-expression e-matching, `ConstantFold`, unit analysis, explanations, differentiation, association, integration, radical simplification, and factorization. Development and final selections are disjoint.

Samples now last at least 4.0 seconds. Three complete paired evaluations span 0.062420 score points on development and 0.478837 on final, compared with the approximately 1.85-point effect of a uniform 5% speedup. Exact measurements are in `evidence/preflight.md`.

This changes what the next trial measures by broadening score-bearing coverage across authentic upstream workload families. The formula and strict 5% conjunction are unchanged.

Round 5 added independent method-separation evidence without changing the evaluator. A valid worklist/fixed-point implementation scored 49.420472--49.660391 across three runs; a valid correctness-first global repair scored 62.314854--62.393526. The closest cross-method observations remain 12.654463 points apart, while the larger within-method span is 0.239918. A third independent reconstruction failed the correctness boundary and was excluded. See `evidence/trajectories.md`.
