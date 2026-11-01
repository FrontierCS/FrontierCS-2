# Localized Regex Repair

The task uses a 100-case subset of authentic Polyglot patterns, balanced 50/50 by RegexScalpel outcome. Development has 20 records; final includes all 100. Fixed original-order prefixes select 10 development and 40 held-out records per stratum, without inspecting model performance. The paper uses 1,000 cases; this adaptation does not claim to reproduce its reported numbers.

The retained constructor consumes solver-supplied localization instructions and prompt templates. Candidate and reference share fixed Qwen3-8B Q4_K_M bytes, greedy non-thinking decoding, seed 42, 512 output tokens, and a matching llama.cpp CUDA runtime on one Driver-managed Modal A100-80GB. All eight metric comparisons remain unchanged. GPU runtime and platform acceptance are pending. Per-verification timeout is 3600 seconds; total GPU budget 7200 seconds; training pool 5400 seconds; answering window 18000 seconds; call cap 100.

Continuous score: 100 times the mean over all frozen regexes of q = credited x min(kept, admitted), where kept and admitted are the two directional Xeger acceptance rates (the artifact's Jaccard estimate is their mean). This replaces the plain repair rate, which a broad safe rewrite such as `[\s\S]*` could drive to 90 on development. The eight-metric beat-spec gate is unchanged.
