# Localized Regex Repair

Write the instruction policy that lets a fixed language model repair regular expressions with
catastrophic backtracking (ReDoS). Maximize the mean, over all frozen regexes, of a credited
repair's language preservation: a repair counts only if it compiles, survives the ReDoS detector
and keeps a nonzero overlap with the original language. A valid submission keeps the constructor
working: it returns one well-formed chat per input without changing the original pattern.

## Task

The system (paper: *Repairing Regex Vulnerabilities via Localization-Guided Instructions*) gives a
fixed Qwen3-8B model a prompt built from a vulnerable Polyglot regex and its first-step RegexScalpel
symbolic localization, and asks for one complete repaired regex. The localization-to-instruction
policy of the original prompt constructor is removed (`NotImplementedError`); supply it. You
control the instructions, not the model weights or decoding.

## What you may change

- `scripts/scalpel_to_thinking_steps.py`: `subpattern_to_string` (also `generate_cot_prompt`,
  which the scored path does not call)
- `prompts/localization.txt`, `prompts/5-shot-localization-prompt.txt`, `prompts/1-shot-prompt.txt`,
  `prompts/5-shot-prompt.txt`
- new files under `frontier_solution/` (reusable source); editable files may also be deleted

Everything else is fixed, including `scripts/inference.py`, its parser and `prompts/prompt.txt`.
The policy may use only the original regex and the first symbolic localization the constructor
passes in (not RegexScalpel's repaired output, later steps or earlier model answers). Model outputs
must be complete regexes, not fragments.

## Interface

`subpattern_to_string(subpattern: str) -> str` returns the text inserted for one symbolic
subpattern. The templates are `string.Template` files and must stay valid for
`string.Template.substitute`: the localization templates receive exactly `pattern` and
`subpatterns`, the ordinary ones `pattern`. The scored path is `five_shot_localization`
(`prompts/5-shot-localization-prompt.txt`); inputs without a localization use the ordinary
`prompts/5-shot-prompt.txt` instead. Renderers return strings and may not alter the retained constructor or parser.

Example: an input with pattern `^(a+)+$` and first-step localization `(a+)+`:

```python
make_chat = get_line_to_chat_and_pattern("five_shot_localization")
chat, pattern = make_chat(input_json_line, None)
pattern  # "^(a+)+$"
chat     # [{"role": "user", "content": "<your rendered instruction>"}]
```

The judge then runs the equivalent of

```text
python scripts/inference.py --task five_shot_localization --dataset-path <frozen-jsonl> --output-dir <results> --model-name Qwen/Qwen3-8B --max-answer-tokens 512
```

through the retained constructor and the fixed offline model. A model answer is usable only if it
contains exactly one fenced, single-line regex.

## Correctness

The submission scores 0 when the constructor fails to import, raises, exceeds 600 seconds per
workload, changes the original pattern, returns `None` for an input, or returns a malformed chat
or the wrong number of chats.
The constructor runs in its own process per workload with no network and no access to the
evaluator's files; every process it starts is ended when it returns.

Per input (counted as a failed repair, still in the denominator): a chat the model server rejects
(for example over the 8,192-token context), an answer cut off at the 512-token limit, empty output, several fenced answers, a line break
inside the answer, invalid syntax, detector or Java parser errors, tool timeouts, and missing,
incomplete or timed-out semantic samples. A failure of the evaluator or the fixed model is an
unscored failed verification.

## Workload

One workload is one stratum of one split: train has the two development strata (10 regexes
each); final has those two plus the two held-out strata (40 each). The 100 frozen regexes are unique, originally
detector-confirmed vulnerable Polyglot regexes shorter than 1,024 characters that compile in
Python 3.12 (no CR, LF, NUL or invalid UTF-8): 50 whose RegexScalpel output has a completed
detector result with no successful attack, and 50 whose RegexScalpel output stays vulnerable or
times out.

| suite | regexes | workloads |
|---|---|---|
| train (development) | 20 (10 per stratum) | 2 |
| final | 100: the 20 plus 40 held-out per stratum | 4 |

Membership is withheld. Inputs without a localization (all in the unresolved stratum) are 3 of
the 20 train regexes and 24 of the 100 final ones. Inference is greedy, non-thinking, seed 42, at most 512 output tokens;
detection is ReDoSHunter (Java 25); semantics use 100 Xeger samples per direction, matched in
Python. Execution order may vary; time is not scored.

## Scoring

```text
score = 100 * mean over all regexes of the suite of q
q        = credited * min(kept, admitted)
credited = compiles AND completed detector result with no successful attack
           AND complete bidirectional samples AND jaccard > 0
kept     = fraction of the original's 100 Xeger samples the repair matches
admitted = fraction of the repair's 100 Xeger samples the original matches
jaccard  = (kept + admitted) / 2
```

The score is in `[0, 100]`; one more exact-language repair adds 5 points on train (20 regexes) and 1
on final (100). A superset rewrite has `jaccard ≥ 0.5` but `q = admitted`. Also measured per
workload: vulnerability elimination rate (completed detector results with no successful attack,
regardless of semantics), repair rate, well-formed rate, RLI `(len(repair) - len(original)) /
len(original)`, NLS `1 - edit_distance / max(len(original), len(repair))`, and Xeger Jaccard,
precision and recall; invalid repairs count 0 for the semantic metrics and NLS and 0 as RLI.

**Beats the reference** requires, on every workload, vulnerability elimination rate, repair rate,
well-formed rate, NLS, Jaccard, precision and recall each strictly higher than the reference's
(equal passes only when both are 1), and RLI strictly lower (no tolerance). It does not change the score.

## Running locally

`bash frontier-build.sh` (or `frontier-bench.sh`) in `/workspace/repo` byte-compiles
`scripts/inference.py`, `scripts/parse_regex_scalpel.py` and `scripts/scalpel_to_thinking_steps.py`.
The model, ReDoSHunter and the scorers are only in the Verifier.

## Environment

Verification: Qwen3-8B Q4_K_M (llama.cpp CUDA, one NVIDIA A100 80GB), ReDoSHunter on Temurin 25,
Python 3.12.
