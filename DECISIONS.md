# DECISIONS: RP-OPD replication ("OPD Before RL", arXiv:2610.02781v1)

This file records every choice the paper does not pin down and every model substitution. Code and
configs cite these entries by ID: `(Dn)` in `configs/opd/*.yaml` and in docstrings. The
replication of *Reward Hacking in Rubric-Based RL* keeps its own notes in `REPLICATION_NOTES.md`.

Legend: **Paper** = what the paper says. **Ours** = what we do. **Why** = the reason.

---

## Scope

**D1 Stage 1 only.** We implement RP-OPD (§3 Stage 1, Eq. 4) and its evaluation protocol. Rubric-RL
after OPD (Stage 2, Tables 3 / Figs. 3–5, 7–8, 14), rollout correction (App. A.2) and the RL
reward-hacking regrading (§5.4, App. C, C.2) are not implemented. Stage 2 can be added without
changes to stage 1:
- Each run writes HF-format checkpoints to `checkpoints/hf/step_XXXXXX`.
- `stage2_handoff.json` lists the checkpoints with their scores.
- `POLICY=<hf_path> bash scripts/train_grpo.sh` starts verl GRPO from any OPD@k.

## Objective and optimisation

**D2 Top-K forward KL definition.**
- **Paper:** "We compute the forward KL over the teacher's top-K probabilities, with K = 256". The
  implementation builds on veRL's on-policy distillation recipe.
- **Ours:** ℓ_t = Σ_{k∈TopK_T} p_T(k)·(log p_T(k) − log p_θ(k)).
  - p_T is the teacher's full-softmax probability, not renormalised over the top-K.
  - log p_θ is the student's full-vocabulary log-softmax.
- **Why:** This matches veRL's `forward_kl_topk` (truncated to the teacher's top-K, without the
  aggregate tail bucket of `forward_kl_topk_tail`). The tests check that it equals the exact KL
  when K = V, and that its gradient is (Σp_T)·p_θ − p_T on the top-K.

**D3 Loss aggregation.**
- **Paper:** Eq. 4 sums over t, with no normalisation stated.
- **Ours:** The masked sum is divided by the number of response tokens in the global batch of 512
  sequences ("token-mean", the veRL default). Gradient accumulation and FSDP averaging are
  corrected so the result does not depend on the number of GPUs or micro-batches (tested).

**D6 OPD rollout sampling.** The paper does not state OPD's sampling settings. We use temperature
1.0, top-p 1.0, top-k −1, the paper's RL rollout settings (Table 3). Rollouts are capped at the
Table 2 "response length cap": 1,024 for HealthBench, 2,048 for ResearchQA and 2,048 for RubricHub
Science.

**D7 Batch semantics and training length.**
- **Paper (Table 2):** RP-OPD "global batch size 512". The Llama warm starts are OPD@100/480/680/980
  (Fig. 3). Fig. 11 plots up to about 0.5M "stage-one presentations".
- **Ours:**
  - 512 prompts × 1 sampled response per step.
  - One AdamW update per rollout batch, so training is strictly on-policy.
  - 1,000 steps, which is 512k presentations.
- **Consequence:** HealthBench (4,500 prompts) gets about 114 epochs and RubricHub Science about 18.
  The paper presumably does the same; it does not report epochs.

**D8 Optimiser and precision.**
- **Paper:** AdamW. Learning rate from Table 2: 1e-5 (HealthBench), 1e-5 (ResearchQA), 3e-5
  (RubricHub Science).
- **Ours:**
  - We apply the Table 2 learning rates, which the paper gives for Qwen2.5-3B, to Llama-3.1-8B,
    because the paper gives no Llama-specific values.
  - Betas (0.9, 0.999), eps 1e-8, weight decay 0.01, grad-norm clip 1.0, constant schedule, no
    warm-up. These are the veRL actor defaults, since the implementation builds on veRL.
  - fp32 master weights with bf16 compute (FSDP2 MixedPrecisionPolicy). The frozen teacher runs in
    bf16. All log-softmaxes are computed in fp32.

**D25 Truncated rollouts.** Rollouts that hit the response cap without a stop token are still
trained on every token. `truncated_frac` is logged. When the model stops, the stop token
(`<|eot_id|>`) is part of y, so the student learns when to stop.

## Contexts and prompts

**D4 Teacher context for RP-OPD.**
- **Paper:**
  - The paper prints only the SFT teacher's system prompt (App. A.2).
  - It states that "SFT and RP-OPD use the same teacher".
  - It says the teacher "observes the prompt, rubric, and student-generated prefix".
- **Ours:**
  - Teacher = [system: App. A.2 prompt with the rubric] + the original conversation + y<t, with the
    prompt verbatim.
  - Student = the original conversation + y<t.
  - If a conversation has its own system message, it is kept unchanged after the rubric message.
    None of the three datasets currently has one.
  - The rubric-blind control (`configs/opd/rubric_blind.yaml`) drops the rubric message.

**D5 Rubric rendering.**
- **Paper:** App. A.1 gives the judge's item format as "index. [weight points] criterion", with JSON
  keys 0, 1, 2, … . The teacher prompt only says "{rubrics} denotes the criteria and their weights".
- **Ours:** Both use `"{i}. [{w:g} points] {criterion}"`. The judge numbers items from 0 to match its
  JSON keys. The teacher numbers them from 1.

**D9 Judge prompt layout.** The A.1 prompt body is verbatim. Its section titles are rendered as
markdown `#` headings, as in the HealthBench grader template that this prompt extends. The
conversation is rendered as `role: content` blocks, including the evaluated assistant turn.

**D13 Shortcut-aware judge (App. C.1).**
- **Verbatim:** the system prompt, both halves.
- **Ours, not printed in the paper:**
  - The user-message layout: conversation, `L<n>:`-numbered response lines, and rubric items with
    weights.
  - The JSON key names, and the allowed `basis` values (`substantive_content`, `shortcut_only`,
    `not_present`, `pitfall_present`, `pitfall_absent`, `whole_response_property`).
  - The `confidence` field (low / medium / high).
- **Scoring:**
  - A positive item drops from standard_met to hack_aware_met = false only when all three hold:
    classification = `confirmed`, confidence = `high`, and the index is listed.
  - Negative items keep their polarity.
  - Citation checks then remove credit from a positive met item whose quote is missing, is not an
    exact substring of the cited line, is only a heading, or only asks for content. "Only asks"
    means the quote starts with one of: Discuss, Provide, Explain, Describe, Include, Mention,
    Derive, Outline, Present, Summarize, Address, Cite, Elaborate.
  - Like Table 4, we report three scores: standard, LLM regrade before evidence checks, and after
    evidence checks. Table 1 uses the last.

**D28 Chat template.** Llama-3.1's template inserts its default "Cutting Knowledge / Today Date"
header. Student, teacher, rollouts and evaluation all use the same template.

## Data

**D19 Splits.** Every size matches the paper exactly. They were verified locally on 2026-10-31; see
`data/opd/manifest.json`.

| Dataset | Train | Eval | How |
|---|---|---|---|
| HealthBench | 4,500 | 500 held-out (+500 train-overlap) | 5,000 public examples, seeded shuffle (seed 0). RuscaRL's split files are not public, so the split is ours. The overlap set (Fig. 3) is 500 train prompts (seed 1). |
| ResearchQA | 16,961 | 500 | train = full.json − test.json − valid.json (exactly 16,961). Eval = 500 sampled from test (seed 0). |
| RubricHub Science | 28,918 | 500 | `RuRL/rurbichub_v1_Science.parquet`, deduplicated and shuffled (seed 0). The first 500 are eval, the next 28,918 train. Unlike the previous replication, there is no source filter. |

**D23 ResearchQA rubric conversion.**
- **Source data:** the items are unweighted yes/no questions.
- **Paper:** App. A.1 shows them as "+10 The response include …".
- **Ours:** weight 10, and a leading "Does the response" becomes "The response", with the "?"
  becoming ".". This covers 160,696 of 160,705 items; the other 9 are kept verbatim.

**D14 Length caps.** The student context is capped at 3,072 tokens and the teacher context
(rubric + prompt) at 6,144; longer prompts are dropped and logged in `logs/prompt_filter.json`.
Measured with the Llama-3 tokenizer on 3,000 training prompts per dataset, maximum student and
teacher lengths were:

| Dataset | Max student | Max teacher |
|---|---|---|
| HealthBench | 1,655 | 2,899 |
| ResearchQA | 93 | 555 |
| RubricHub Science | 1,537 | 3,324 |

So no prompt is dropped in practice.

## Models and judges (strict no-Chinese-model rule)

| Role | Paper | Ours | ID |
|---|---|---|---|
| Student | Llama-3.1-8B-Instruct | meta-llama/Llama-3.1-8B-Instruct (same) | — |
| Teacher | Llama-3.1-70B-Instruct | meta-llama/Llama-3.1-70B-Instruct (same) | — |
| Curve / reward judge | Qwen3-32B (deterministic, 512 output tokens, JSON only) | meta-llama/Llama-3.3-70B-Instruct (vLLM, temperature 0, max_tokens 512) | D10 |
| Independent judge (Table 1) | GPT-4o-mini, shortcut-aware | google/gemma-3-27b-it (vLLM, temperature 0, max_tokens 4,096). The original gpt-4o-mini is available as `gpt-4o-mini-judge` | D11 |
| Reference judge for validation | — | openai/gpt-oss-120b (vLLM, reasoning_effort low) | D12 |
| Smoke test | — | Llama-3.2-1B-Instruct student, Llama-3.2-3B-Instruct teacher, Llama-3.1-8B-Instruct judge | D18 |

**D10 Qwen3-32B → Llama-3.3-70B-Instruct.**
- **Why:** Qwen is not allowed. Llama-3.3-70B is the strongest permitted dense instruction model
  that returns bare JSON within the paper's 512-token deterministic budget.
- **Alternative rejected:** gpt-oss-120b is a reasoning model whose hidden reasoning would use up a
  512-token budget, so it was not used as the curve judge.
- **Caveat:** Llama-3.3 judging Llama-3.1 outputs is a same-family setting. The paper has the same
  situation (Qwen3 judging Qwen2.5 students and teachers).

**D11 GPT-4o-mini → Gemma-3-27B-it.**
- **Why:** This follows the user's mapping for the "GPT-4o-mini class". Temperature 0 is our choice;
  the paper does not give one. GPT-4o-mini itself is permitted (it is not a Chinese model), so
  `gpt-4o-mini-judge` exists for an exact-model check. See "Cost" for what that costs.
- **Caveat:** Gemma-3's chat template has no system role, so vLLM folds the C.1 system prompt into
  the first user turn.

**D12 Validating the substitutions.** `rrh.analysis.judge_agreement` runs on a fixed 200-prompt
hashed subset of each eval set, across all checkpoints plus the teacher.
- Gemma-3-27B (A.1 boolean) vs Llama-3.3-70B: criterion-level agreement / FP / FN. This is the
  weak–strong gap.
- gpt-oss-120b vs Llama-3.3-70B: whether the strong substitute agrees with an independent strong
  judge.
- Mean Eq. (1) score per judge on identical responses. The paper's only cross-judge evidence is
  Fig. 13, where the GPT-4o-mini standard score is within about ±0.03 of Qwen3-32B on the same
  responses. Those values are printed next to ours in `tables/paper_fig13_reference.csv`.

The paper gives no criterion-level agreement numbers, so the target is the score gap.

## Infrastructure

**D15 Rollout engine.**
- **Ours:** one vLLM engine per training rank (`external_launcher`, TP = 1), colocated with FSDP2
  through sleep mode level 1. Before every generation the current student weights are pushed in
  with `load_weights` (TRL's colocate approach).
- **Fallback:** `rollout.backend=hf` uses `model.generate`, with `synced_gpus` under FSDP.
- **Risk:** reaching the in-process vLLM model touches version-specific internals
  (`VLLM_ENABLE_V1_MULTIPROCESSING=0`). Run the 1-GPU smoke test and then a 2-GPU smoke test with
  `torchrun` before the full run.

**D16 Checkpoint cadence and disk.**
- **HF exports:** every 50 steps plus steps {20, 100, 480, 680, 980} (the paper's Llama warm starts
  and Fig. 12's first point). That is about 25 × 16 GB ≈ 400 GB per run.
- **Resume states:** fp32 weights plus AdamW moments, about 96 GB each for 8B. One every 100 steps
  and one at each walltime exit, about 1 TB per run.
- **Deletion:** nothing is deleted automatically. `train.keep_resume_last: k` prunes old resume
  states only, and is off until the user approves it. HF exports are never pruned.

**D17 Rollout correction** (App. A.2: geometric rejection sampling and token veto) applies to RL
only. It is not part of stage 1.

**D21 Checkpoint reported in the Table 1 "stage 1 only" row.** The paper does not say which
checkpoint it uses. We report the final checkpoint (OPD@1000). The best checkpoint by the curve
judge is also shown, but it is not used for selection, because picking on the held-out set would
leak.

**D22 Entropy.**
- **Evaluation:** the mean over response tokens of the entropy of the top-20 renormalised
  distribution (the Fig. 12b definition). It comes from vLLM's 20 logprobs, which are the raw
  model distribution before temperature and top-p processing (vLLM's default `raw_logprobs` mode).
- **Training:** both the full-vocabulary entropy and the top-20 entropy on the rollouts.

**D24 Evaluation randomness.** One sample per prompt (A.3). The per-request seed is 0 + the row
index. Generations are cached on disk, so re-grading never re-samples.

**D27 Judge failures.** Up to 3 re-asks, keeping items already parsed. Items still missing count as
not met (g = 0) and are logged as `n_failed_items`.

**D20 Stage-2 reward.** The paper's Eq. (1) divides by the sum of *positive* weights, and met
pitfalls subtract (`rrh.grading.reward.normalized_rubric_score`). The previous replication's verl
reward (`rubric_reward`) divides by Σ|w|. When stage 2 is added, `reward_fn.py` must switch to
Eq. (1) for this paper.

## Cost and compute estimates (logged before any large run)

**API.** The default pipeline makes **no paid API calls**: every judge is self-hosted with vLLM.
The optional `gpt-4o-mini-judge` (paper-exact independent judge, $0.15 / $0.60 per 1M tokens in /
out) costs:
- about 3k input and 1.5k output tokens per response, or about **$1.35 per 1,000 responses**;
- Table 1 only (final checkpoint + teacher, 3 datasets, 500 prompts each, about 3k responses):
  **about $4**;
- every checkpoint (about 25 × 500 × 4 eval sets, about 50k responses): **about $70**.

Check the price and that `gpt-4o-mini` is still served before using it. The API panel of the
previous replication is not used here.

**GPU.** Rough figures for H100-80GB; A100 is about 2.5–3× slower, and Newton's GPU types are
unconfirmed (docs/CLUSTER.md).
- **Per training step** (512 rollouts):
  - about 1.1M teacher-prefill tokens (≈1.5e17 FLOP);
  - about 0.5M student train tokens (≈3e16 FLOP);
  - decoding 512 × ≈800 tokens.
- **One step on 16 GPUs:** about 100–150 s.
- **1,000 steps:** **about 30–40 h on 2 × 8 H100, about 500–650 GPU-h per dataset**, about 1.5–2k
  GPU-h for all three datasets. On A100: about 4–5k GPU-h.
- **Eval generation:** about 25 checkpoints × 500–1,000 prompts, about 2 GPU-h per dataset.
- **Judging:** Llama-3.3-70B on TP4 is about 4–8 GPU-h per dataset. Gemma-3-27B shortcut-aware on
  TP2 is about 6 GPU-h. gpt-oss-120b on the 200-prompt subset is about 2 GPU-h.
- **Smoke test:** under 1 GPU-h.

Update these numbers from `metrics/train.jsonl` (`time_*` fields) after the smoke test and the
first 20 real steps.
