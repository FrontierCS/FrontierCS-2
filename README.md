# FrontierCS tasks for Pumpkin

This branch holds 47 of the 48 tasks that [FrontierCS](https://github.com/FrontierCS/FrontierCS-2)
publishes, all but `vllm_llm_serving_optimization`, each as a workflow that
[Pumpkin](https://github.com/FrontierCS/Pumpkin/tree/ad8556425e08a190b2800562dfbb99b4b1092dd1)
runs. In a run, an agent works on the task's statement and starter, may submit to the task's
verifier for scored train calls, and submits once for the final call, whose score is the run's.
Each role, the agent and the two judges that run the verifier, states what its sandbox needs. The
agent is a harness, one of the coding agents that FrontierCS evaluates, such as Codex or Claude
Code, on a model of your choice, or `reference`, which submits the task's reference solution.

## What you need

- Linux on x86_64 with `git`, `git lfs`, `curl`, `python3` and a C compiler. `./pumpkin` builds
  the Pumpkin that `.pumpkinversion` pins on its first run, which downloads about 250 MB, takes a
  few minutes and leaves about 1.5 GB in `.pumpkin/`, and names any package that is missing.
- Docker, which builds the tasks' images and runs their sandboxes.
- For an agent that works on a model, a key for a service that serves the model.

## Check and run a task

Clone with `git lfs` installed, so that the clone fetches its Git LFS files, about 2.8 GB, and write
`~/.config/pumpkin/pumpkin.toml` so that Pumpkin runs every role on this machine's Docker engine:

```toml
[docker]
context = "default"

[providers.local.docker]
```

The reference agent submits the task's reference solution and needs no model, so it is the cheap
way to see a task end to end; only `progent_privilege_control`'s judges call a model (`--arg
task_model`), so even its reference run needs an upstream (below) that serves it, with its prices:

```console
$ git clone --single-branch --branch pumpkin https://github.com/FrontierCS/FrontierCS-2.git fcs-tasks
$ cd fcs-tasks
$ ./ci.py                    # builds Pumpkin on its first run, then checks every task
...
==> pumpkin check, 47 tasks, --arg=agent=codex --arg=model=gpt-6.1-terra
ci.py: all checks passed
$ ./pumpkin check tasks/binary_fuse_filter/task.star --arg agent=reference   # silent if sound
$ ./pumpkin run tasks/binary_fuse_filter/task.star --arg agent=reference     # prints the run's id
...
20261117-040053-978f3ae9
$ ./pumpkin wait 978f        # follows the run, named by a prefix of its last part, to its end
...
pumpkin: run 20261117-040053-978f3ae9 finished with a result (score=50.02 status="passed" beats_reference=0 summary="valid; strict FAIL" ...), after the work phase ended when the agent exited with status 0; its record is /home/alice/fcs-tasks/.pumpkin/runs/20261117-040053-978f3ae9
...
$ ./pumpkin wait --outcome 978f > outcome.json
$ ./pumpkin ls               # the runs of the last day
20261117-040053-978f3ae9 finished      1m56s  -               -  frontier-cs/binary_fuse_filter: score=50.02 status="passed" beats_reference=0 ...
...
```

`run` builds the task's images, starts the run and returns; the run goes on without it. A run's
record is `.pumpkin/runs/<id>/`, and `wait --outcome` prints its `outcome.json` once the run has
ended: `state` is `finished` for a run that ended with a verdict, and `verdict.result.values` lists
the final call's values, each a `name` and a `value`: `score`, `status` (`passed` or `failed`),
`beats_reference`, then the verifier's `summary` and metrics.

A harness works on a model that an upstream serves: an `[upstreams.<name>]` table of the same
`pumpkin.toml` names the upstream's URL, the file that holds its key and the models it serves, by
the `name` that `--arg model` gives; `./pumpkin help config` says every key, with an example
([models](https://github.com/FrontierCS/Pumpkin/blob/ad8556425e08a190b2800562dfbb99b4b1092dd1/docs/models.md)):

```console
$ ./pumpkin run tasks/binary_fuse_filter/task.star --arg agent=codex --arg model=gpt-6.1-terra
```

`--arg agent` names one of `HARNESSES` in `lib/fcs.star`; `check` says what a harness lacks:
under Codex's default, `--arg tools=functions`, it refuses a model outside the catalog that
Pumpkin's Codex carries and names the models it does carry (`TOOLS` in `lib/fcs.star` says why
`functions` is the default). The `arg()` calls at the top of `lib/fcs.star` declare what every
task takes, such as `deadline_seconds`, the work phase, five hours by default.
[Running](https://github.com/FrontierCS/Pumpkin/blob/ad8556425e08a190b2800562dfbb99b4b1092dd1/docs/running.md)
goes through each command and `pumpkin view`, the runs' page;
[records](https://github.com/FrontierCS/Pumpkin/blob/ad8556425e08a190b2800562dfbb99b4b1092dd1/docs/record.md)
says what a record holds.

## A task that needs a GPU or another provider

The local Docker engine offers this machine's CPUs, memory and CPU flags, but no GPU and no CPU
model, so the 17 tasks that state either (`grep --files-with-matches --extended-regexp
'"(gpu|cpu_model)"' tasks/*/task.star`) run on EC2 or Modal, each of which needs the AWS envelope
that Pumpkin's `deploy/aws` makes. `run` refuses a role that no class of the configured providers
offers, saying what each class lacks, before anything is reserved.
[Providers](https://github.com/FrontierCS/Pumpkin/blob/ad8556425e08a190b2800562dfbb99b4b1092dd1/docs/providers.md)
says what EC2, Modal and a controller, which builds the images and drives the runs there, each
need, and where each is set up.

## What is in the repository

- `tasks/<id>/`: a task each. `task.star`; the published task's `judge/`, `datasets/` and `repo/`;
  `generated/`, the configuration that FrontierCS's build derives from the task; the recipes of its
  images (`agent-base/`, `verifier/`, and `runtime/` where they share one); and this branch's own
  judge files: `eval/`, the evaluator of the 10 tasks whose row below names one, and `train/`, what
  the train judge's image holds or runs in place of what it leaves out.
- `lib/`: what every task shares: `fcs.star`, FrontierCS's conventions and `fcs_task`, which each
  `task.star` calls; the platform's part of each agent's README; `frontier-gpu`, the agent's client
  for its GPU runners.
- `shared/`: the components that FrontierCS shares among its tasks, which the recipes copy, and
  `verifier-lease/`, a judge's GPU adapter with FrontierCS's interface, over runners that Pumpkin
  leases to the judge.
- `fetch/`: the helpers that recipes run to fetch what a digest pins.
- `smoke/`: two workflows that check a deployment, not a task: `lease/` leases GPU and CPU
  runners, and `routes/` sends model requests from every role; no task loads them.
- `PUMPKIN.star`, `pumpkin`, `.pumpkinversion` and `ci.py`: the root of the `//` labels, the
  script that builds and runs the pinned Pumpkin, the pin, and the checks.

## How the ports differ

A task here is a port of the published one: its `judge/`, `datasets/` and `repo/` are the files of
FrontierCS at commit [`2fc5ff7c`](https://github.com/FrontierCS/FrontierCS-2/tree/2fc5ff7c6e43e11c64d7d644b9808b010b3814b5), unchanged; its `task.star` and recipes restate the published
`task.toml`, `README.md` and `docker/` recipes for Pumpkin, and its agent submits with Pumpkin's
`sandboxctl` in place of `frontier-submit`. A port departs from how FrontierCS runs the task in
three ways: in 10 tasks the judges' images hold the port's `eval/evaluator.py` in place of
`judge/evaluator.py`, so that the score no longer rests on what the submission's code prints or
leaves behind; in 43 the train judge's image leaves out what only the final evaluation reads, so
that a train call cannot reach it; and in 10 a value or a line of the judge differs, where Pumpkin's
sandboxes or the port's evaluator need another. The table lists the 45 tasks that depart;
`lwe_structured_recovery` and `quarter_pir` depart in none.

| Task | The port's evaluator | Left out of the train judge | Other differences |
| --- | --- | --- | --- |
| `approximate_mips_global_probing` | recomputes the results' inner products (`eval/inner_products.cpp`) | the second split's seed and the ground truth of its queries |  |
| `approximate_mips_query_stopping` | recomputes the results' inner products (`eval/inner_products.cpp`) | the final-only heap capacities and the ground truth of the second split's queries |  |
| `babble_anti_unification` |  | the final corpora |  |
| `batch_local_counting` |  | the final's case lists |  |
| `bbq_packet_scheduling` | takes a testbench's PASS line only with a random nonce | the held-out cases |  |
| `binary_fuse_filter` |  | the final seed and the final workloads' calibration |  |
| `bounded_evm_path_prediction` | in the final call: replays the predictor's logged answers in its own build (`eval/seer/`) | the blocks the train judge never replays |  |
| `ckks_composable_noise` |  | the final's frozen bands |  |
| `clusterwise_spgemm` |  | the final suite's archive, matrix list and member list |  |
| `cute_crest` |  | the final programs |  |
| `dbscan_gpu_kernel_optimization` |  | the hidden workloads' shapes and seed |  |
| `detok_latent_denoising` |  | the final suite | 1024 commands on the agent's GPU runners, not 32 |
| `diffusion_policy_cpu` |  | the final suite's dataset |  |
| `dynamic_tanh` |  | the final workloads' photographs, models, batch sizes and seeds | 1024 commands on the agent's GPU runners, not 32 |
| `edlib_exact_alignment` |  | the held-out workloads and the qualification record | both judges accept a self-reported time up to 0.1 s over the measured one |
| `egraph_invariant_restoration` | checks what a timed invocation leaves behind |  |  |
| `evm_state_checkpoints` |  | the final suite's seeds |  |
| `fast_kv_compaction` |  | the final suite's articles |  |
| `flow_opd` |  | the final suite | 1024 commands on the agent's GPU runners, not 32 |
| `glenside` |  | the final workloads' seeds |  |
| `graph_mining_method_selection` |  | the final suite's rows |  |
| `hypergraph_core_numbers` |  | the graded graphs and the manifest |  |
| `ivf_pq_gpu_kernel_optimization` |  | the hidden workloads' shapes and seed |  |
| `join_tree_rewriting` |  | the final queries' plans |  |
| `kami_fp64_block_gemm` |  | the hidden workloads |  |
| `kmeans_gpu_kernel_optimization` | computes the quality metrics from the returned labels and centroids | the hidden workloads' shapes and seed |  |
| `knn_gpu_kernel_optimization` | computes the recalls and distance errors from the returned results | the hidden workloads' datasets, sizes and seed |  |
| `lejepa_world_model` | recomputes the statistics from the saved encoder weights | the held-out seeds |  |
| `lrr` |  | the final suite |  |
| `metaopt_combined_te_gap_search` |  | the final suite's key |  |
| `mujoco_playground_cpu` |  | the final suite's seeds | 1024 commands on the agent's GPU runners, not 32 |
| `nano_world_model` |  | the final suite's manifest, frames and public data, but not the tensors that hold every episode | 1024 commands on the agent's GPU runners, not 32 |
| `parallel_dbscan` |  | the final suite's seeds |  |
| `parallel_radix_sort` |  | the final workloads' hidden rows |  |
| `parbutterfly` |  | the hidden graphs |  |
| `persistent_treap_update_engine` |  | the final suite's seed material |  |
| `polar_express` |  | the final seed |  |
| `polygon` |  | the final workloads |  |
| `progent_privilege_control` |  |  | `MODEL_SESSIONS` 7, not 8; the task model (`--arg task_model`) through each role's model route; FrontierCS's three key budgets as the routes' budgets, at the configuration's prices |
| `ragen_multi_turn_rl` |  | the final suite's seed and group selection | 1024 commands on the agent's GPU runners, not 32 |
| `representation_frechet_loss` |  | the final workloads' seeds | 1024 commands on the agent's GPU runners, not 32 |
| `residual_state_advancement` |  | the final workloads' data |  |
| `runlts_predictor` | in the final call: replays the predictor's recorded directions in its own simulator | the 81 final traces, and the final rows of the reference table | the final call's timeout 12,000 s, not 7,200 s |
| `shockwave_dynamic_gpu_allocation` | replays the policy's rounds in its own simulator | the final suite's two traces |  |
| `sparse_video_attention_policy` |  | the graded suite's cached outputs |  |
