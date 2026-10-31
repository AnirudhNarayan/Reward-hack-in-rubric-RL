# Implementation plan: RP-OPD (stage 1 of "OPD Before RL")

Paper: Wang, Shi, Chen, Zontak, He, Pang. *OPD Before RL: Warm-Starting Rubric-Based RL with
On-Policy Distillation*, arXiv:2610.02781v1 (NYU / Meta). PDF in `../papers/OPD before RL.pdf`.

**Scope.** This plan covers only stage 1: rubric-privileged on-policy distillation (RP-OPD). Stage 2
(Rubric-RL / GRPO after OPD) is out of scope. RP-OPD writes HF-format checkpoints, so the existing
verl GRPO pipeline can start from them with `POLICY=<run>/checkpoints/hf/step_XXXXXX bash
scripts/train_grpo.sh` (see "Stage-2 hand-off" below).

Every choice the paper leaves open is recorded in `DECISIONS.md` and given an ID (D1, D2, …),
which the code cites.

## 1. Paper → code map

| Paper element | What it says | Where it is implemented |
|---|---|---|
| §2 Eq. (1) rubric score | R = Σ_j w_j g_j / Σ_{w_j>0} w_j. Met pitfalls subtract. | `rrh.grading.reward.normalized_rubric_score` |
| §2 Eq. (3) OPD objective | Σ_t D(π_T(·\|x,y<t), π_θ(·\|x,y<t)), y ~ π_θ | `rrh.opd.trainer.OPDTrainer.train_step` (on-policy loop) |
| §3 Eq. (4) RP-OPD | Teacher sees (x, r, y<t). Student sees (x, y<t). Forward KL. | contexts: `rrh.opd.contexts.{student_messages,teacher_messages}`; loss: `rrh.opd.loss.topk_forward_kl` |
| §3 "top-K, K = 256" | Forward KL over the teacher's top-256 probabilities | `rrh.opd.loss.teacher_topk` + `topk_forward_kl` (`loss.topk: 256`) |
| App. B.4 reverse-KL ablation | k3 estimator, backprop through student log-probs (veRL OPD) | `rrh.opd.loss.reverse_kl_k3` (`loss.mode: reverse_kl_k3`) |
| App. A.2 teacher system prompt | "Your next response will be evaluated using the rubric below …" (verbatim) | `rrh.grading.opd_prompts.TEACHER_SYSTEM_PROMPT`, `rrh.opd.contexts.format_rubric_for_teacher` |
| App. A.2 rubric-blind control | The teacher's rubric message is omitted | `teacher.rubric_in_context: false` → `rrh.opd.contexts.teacher_messages` |
| §4 Models | Llama-3.1-8B-Instruct student, Llama-3.1-70B-Instruct teacher | `configs/opd/base.yaml` (`student.path`, `teacher.path`) |
| Table 2 (RP-OPD column) | lr 1e-5 / 1e-5 / 3e-5, global batch 512, response cap 1024 / 2048 / 2048 | `configs/opd/{healthbench,researchqa,rubrichub_science}.yaml` |
| App. A.2 "AdamW" | Optimizer | `rrh.opd.trainer.build_optimizer` |
| Table 3 rollout sampling | T = 1.0, top-p = 1.0, top-k = −1 (we also use it for OPD rollouts, D6) | `rollout.*` in `configs/opd/base.yaml` |
| §4 datasets / App. A split sizes | HealthBench 4,500 / 500. ResearchQA 16,961 / 500. RubricHub Science 28,918 / 500. | `rrh.data.opd_data` (with `healthbench.py`, `researchqa.py`, `rubrichub.py`) |
| §4 / Fig. 3 vs Fig. 4 | HealthBench evaluation on 500 train-overlap prompts and 500 held-out prompts | `rrh.data.opd_data` writes `eval_heldout.jsonl` and `eval_overlap.jsonl` |
| App. A.1 rubric cards | ResearchQA items carry +10 and the "Does the response …?" wording becomes "The response …" | `rrh.data.researchqa.rubric_item_to_criterion` |
| App. A.3 eval decoding | 1 response per prompt, T = 0.7, top-p 0.8, max 1,024 / 1,024 / 2,048 tokens | `rrh.opd.evaluate` (`eval.*` config) |
| App. A.1 rubric judge prompt (Qwen3-32B) | Verbatim prompt. Items rendered as "index. [weight points] criterion". 0-indexed boolean JSON. | `rrh.grading.opd_prompts.RUBRIC_JUDGE_PROMPT`, `rrh.grading.opd_grader.BooleanRubricJudge` |
| App. A.3 judge decoding | Deterministic, 512 output tokens, no explanations | `configs/models.yaml` → `llama-3.3-70b-judge` (temperature 0, max_tokens 512) |
| §5.3 / App. C.1 shortcut-aware judge (GPT-4o-mini) | Verbatim system prompt. Returns standard_met and hack_aware_met. Citation check. | `rrh.grading.opd_prompts.SHORTCUT_AWARE_SYSTEM_PROMPT`, `rrh.grading.opd_grader.ShortcutAwareJudge`, `rrh.grading.citations` |
| App. C "evidence checks" | Credit is removed when a quote is missing, is only a heading, or only asks for an answer | `rrh.grading.citations.check_item_evidence` |
| Fig. 2 (stage-1 curves) | Rubric score on held-out sets vs OPD step. Gold line = rubric-conditioned teacher. | `rrh.opd.evaluate` (per checkpoint) + `rrh.analysis.opd_report.figure2` |
| Fig. 2 teacher reference | Teacher generates with the rubric in context, scored by the same judge | `rrh.opd.evaluate --policy teacher` (context `teacher`) |
| Table 1 "RP-OPD (ours, stage 1 only)" row | Shortcut-aware independent judge on held-out sets | `rrh.analysis.opd_report.table1` |
| Fig. 5 / Fig. 12 (b) entropy | Token entropy over the 20 most probable tokens, renormalized | eval: `rrh.opd.metrics.top20_entropy` (vLLM logprobs = 20). Train: `rrh.opd.loss` stats |
| Fig. 13 judge comparison | Same responses scored by the curve judge and by the standard and shortcut-aware independent judges | `rrh.analysis.judge_agreement` (+ agreement / FP / FN) |
| App. A.2 rollout correction | Geometric rejection sampling (RL only) | Not stage 1. Not implemented (D17). |
| §5.4, C.2 reward-hacking regrading | Applies to RL rollouts | Out of scope (stage 2). Shortcut-aware grading is implemented for Table 1. |

## 2. Model substitutions (strict no-Chinese-model rule)

| Paper role | Paper model | Ours | Why |
|---|---|---|---|
| Student | Llama-3.1-8B-Instruct | **meta-llama/Llama-3.1-8B-Instruct** | Same model as the paper's Llama row |
| Teacher | Llama-3.1-70B-Instruct | **meta-llama/Llama-3.1-70B-Instruct** | Same model as the paper; identical tokenizer |
| Curve / reward judge | Qwen3-32B (deterministic, 512 tokens, JSON) | **meta-llama/Llama-3.3-70B-Instruct** (vLLM) | Closest permitted dense instruction model that returns JSON without reasoning tokens (D10) |
| Independent judge | GPT-4o-mini (shortcut-aware) | **gpt-4o-mini** via API (same model as the paper; Gemma-3-27B-it as fallback) | User-approved, about $2–3 per run (D11) |
| Reference judge (validation only) | — | **openai/gpt-oss-120b** (vLLM) | A second strong judge to check the strong-judge substitution (D12) |
| Smoke test | — | Llama-3.2-1B-Instruct student, Llama-3.2-3B-Instruct teacher | Same tokenizer family (D18) |

Every substitution is validated by `rrh.analysis.judge_agreement`, which computes criterion-level
agreement, FP and FN against the strong judge, plus mean scores per judge (paper Fig. 13 style).

## 3. Code layout (new files)

```
src/rrh/opd/
  config.py       YAML + dotlist overrides -> resolved config (saved to run/config.yaml)
  contexts.py     student/teacher message + token construction (Eq. 4 contexts, App. A.2)
  loss.py         teacher top-K, top-K forward KL (Eq. 4), reverse-KL k3 (B.4), entropy stats
  batching.py     aligned left/right padding so all response positions share columns
  dist.py         torch.distributed / FSDP2 helpers, sharded safetensors loading (meta init)
  rollout.py      HFRollout (smoke/fallback) and VLLMColocatedRollout (sleep mode + weight sync)
  checkpoint.py   HF-format export (stage-2 hand-off) + DCP resume state
  trainer.py      on-policy loop: sample -> teacher top-K -> loss -> AdamW step; metrics; resume
  train_cli.py    entry point (torchrun / srun)
  evaluate.py     generate (A.3 decoding) at checkpoints + teacher reference + entropy
  metrics.py      top-20 renormalized entropy, aggregation
src/rrh/grading/
  opd_prompts.py  verbatim A.1 judge, A.2 teacher, C.1 shortcut-aware prompts (+ our user templates)
  opd_grader.py   BooleanRubricJudge (A.1), ShortcutAwareJudge (C.1), resumable grading CLI
  citations.py    C.1 citation checks
  reward.py       + normalized_rubric_score (Eq. 1)
src/rrh/data/
  researchqa.py   ResearchQA (realliyifei/ResearchQA): train = full − test − valid = 16,961
  opd_data.py     builds the three OPD datasets + eval sets; prompt-length filtering
src/rrh/results/
  layout.py       results/<experiment>/<model>__<judge>__<dataset>__seed<k>__<stamp>/ + INDEX.csv
src/rrh/analysis/
  judge_agreement.py  substitution validation (agreement / FP / FN, score gaps)
  opd_report.py       rebuilds Fig. 2 (stage-1 curves), Fig. 12b-style entropy, Table 1 row, tables
configs/opd/      base.yaml, healthbench.yaml, researchqa.yaml, rubrichub_science.yaml, smoke*.yaml
scripts/cluster/inspect_cluster.sh  writes docs/cluster/<name>.txt (sinfo, partitions, GPUs, net)
scripts/slurm/opd/  prepare_data, smoke_test, train_opd, generate_eval, teacher_reference,
                    serve_judge, grade, judge_validation, report (+ lib/common.sh, lib/vllm.sh)
scripts/submit_opd_pipeline.sh  one-command launch with --dependency chains
tests/test_opd_*.py              loss, contexts, rubric formatting, results writing, grader, data
```

## 4. Training algorithm (one step)

1. Draw `global_batch = 512` prompts (D7). Each rank takes the slice `rank::world`.
2. **Rollout.** Sample y ~ π_θ(·|x) with the student's *current* weights (vLLM colocated, synced
   every step; T = 1, top-p = 1, max response = dataset cap). The stop token (`<|eot_id|>`) is part
   of y.
3. **Teacher.** Teacher input = chat(teacher_messages(x, r)) + y. The frozen 70B forward pass gives
   log π_T at every response position. Keep the top-K = 256 (indices, log-probs). The teacher's
   log-prob of the sampled token is also kept, for the k3 ablation and for metrics.
4. **Student.** Student input = chat(student_messages(x)) + y. Compute log π_θ over the full
   vocabulary and gather at the teacher's top-K indices.
5. **Loss** = Σ_tokens Σ_{k∈topK} p_T(k)·(log p_T(k) − log p_θ(k)), divided by the number of
   response tokens in the global batch ("token-mean", D3). p_T is not renormalized over the top-K
   (D2).
6. Backward with gradient accumulation over micro-batches (every rank runs the same number of
   micro-batches; pads with zero-weight dummies). Clip the grad-norm at 1.0, then take an AdamW step.
7. Log `metrics/train.jsonl`: loss, teacher top-K mass, top-1 agreement, student entropy (full and
   top-20), response length, truncation rate, grad norm, lr, timings.
8. Every `save_hf_every` steps, write an HF checkpoint. Every `save_resume_every` steps (and on
   SIGUSR1 before walltime), write a DCP resume state.

## 5. Evaluation protocol (App. A.3)

For each checkpoint (step 0 = base model) and each evaluation set:
1. vLLM generation, 1 response per prompt, T = 0.7, top-p = 0.8, max tokens 1,024 / 1,024 / 2,048,
   student context (no rubric). Writes `generations/<evalset>/step_XXXXXX.jsonl`, with the per-token
   top-20 entropy.
2. The curve judge (Llama-3.3-70B) uses the A.1 prompt. Writes
   `judgments/<judge>/<evalset>/step_XXXXXX.jsonl`, raw outputs included. Score = Eq. (1), mean over
   prompts.
3. The independent judge (GPT-4o-mini) uses the C.1 prompt and gives standard and shortcut-aware
   scores after citation checks (Table 1), on the final checkpoint only.
4. Teacher reference (Fig. 2 gold lines): the teacher generates with the rubric system prompt at the
   same decoding settings, and the curve judge grades it.
5. Writes `metrics/eval/<evalset>/<judge>/step_XXXXXX.json`. `rrh.analysis.opd_report` rebuilds the
   figures and tables from these.

## 6. Engineering

- **Results layout and INDEX.csv:** `rrh.results.layout`. Each run's `config.yaml` holds the fully
  resolved config and the git commit hash. `checkpoints/` is a symlink to scratch.
- **Resumable training:** DCP sharded model and optimizer state, a sampler cursor and RNG state.
  `--resume auto` picks the latest. SIGUSR1 (sent via `#SBATCH --signal=B:USR1@600`) triggers a
  save and a clean exit. Training is split into chained segments with `--dependency=afterany`.
- **Caching:** all judge calls go through the existing SQLite `ResponseCache`. Raw judge outputs
  also go to `judgments/*.jsonl`, and grading skips items it already has.
- **Smoke test first:** `configs/opd/smoke.yaml` (Llama-3.2-1B student, 3B teacher, 50 prompts,
  4 steps, 1 GPU), `scripts/slurm/opd/smoke_test.slurm`. A local CPU/GPU wiring test with tiny
  random Llama models is in `tests/test_opd_trainer_tiny.py`.
- **Tests:** `tests/test_opd_loss.py`, `test_opd_contexts.py`, `test_opd_grading.py`,
  `test_opd_results.py`, `test_opd_data.py`, `test_opd_trainer_tiny.py`.

## 7. Stage-2 hand-off (not implemented; interface only)

- `checkpoints/hf/step_XXXXXX/` contains `config.json`, `model-*.safetensors` (+ index), tokenizer
  files and `generation_config.json`. verl loads it as `actor_rollout_ref.model.path`.
- `checkpoints/hf/latest` → newest step. `stage2_handoff.json` (in the run dir) lists each
  checkpoint with its eval scores, so OPD@k can be chosen for RL (paper: OPD@100/480/680/980 for
  Llama).
- The rubric reward used in stage 2 must follow Eq. (1) (`normalized_rubric_score`), not the
  Σ|w|-normalized reward of the previous replication. This is flagged in DECISIONS.md (D20).
