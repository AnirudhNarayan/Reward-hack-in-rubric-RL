# Replication notes: deviations and choices

This file covers every place where this code departs from the paper (arXiv:2605.12474v1) or has to pick a value the paper leaves open. Cite it in the thesis.

## Forced substitutions

| Paper | Here | Why |
|---|---|---|
| Policy Qwen2.5-7B-Instruct | `meta-llama/Llama-3.1-8B-Instruct` (`POLICY` env var) | Qwen models are not allowed. OLMo-2-7B-Instruct is the fully-open alternative. |
| Scale ablation Qwen2.5-14B/32B | Pick from Mistral-Small-24B-Instruct, Llama-3.3-70B, gemma-3-12b/27b-it | Same constraint. Run medical/weak only (App. C). |
| Qwen3-30B-A3B candidate verifier (Table 4) | `llama-3.1-8b-judge` (self-hosted) | Keeps a low-agreement open-weight row. Not a like-for-like replacement. |
| Training framework (not named) | verl v0.9.1, GRPO | |

## Values the paper leaves open

- **Batch geometry.** Table 3 gives per-device batch 8, grad-accum 1 and 16 GPUs, which is 128 sequences per optimizer update. We use 128 prompts × 16 rollouts per iteration, in 16 mini-batches of 8 prompts. Medical then comes to about 98 iterations per epoch, or about 490 over 5 epochs. That is consistent with the paper's last checkpoint at 475.
- **Science length.** 5 epochs on science is about 775 iterations, but the paper's science figures stop at 475. Pass `MAX_STEPS=475` to match the figures.
- **KL estimator.** Coefficient is 0.01 (paper). The estimator type is not given; we use `low_var_kl` (k3), applied as a KL loss rather than as a reward penalty.
- **Prompt length.** Max prompt length is 584, which is 2,584 total minus 2,000 response tokens. Longer prompts are filtered out.
- **Eval sampling.** Checkpoint evaluation uses temperature 1.0, top-p 1.0, seed 0 and one response per prompt. This matches the rollout distribution; the paper does not specify eval decoding.
- **Prompt-only context (A.3).** This is the user message with no system message. Llama-3.1's chat template still inserts its default date header. Training, evaluation and ℓ_prompt all use the same template.
- **R_ref under a split panel.** The paper says "unanimous consensus". The default here is `all_met`: a criterion is credited only if all three judges mark it met. The alternative `not_all_reject` reads it as 1−J. Switch with `--consensus_rule`. Exploitation's J is unaffected: it is always "all three reject".
- **Grader decoding.** Temperature is not reported. We use the provider default and leave it unset for reasoning models. GPT-OSS runs with `reasoning_effort: medium`.
- **Unparseable grader items.** The grader retries up to 3 times, keeping items already parsed. Items that still fail count as not met; the count is logged as `n_failed`.
- **Rubric list format** in the A.1 `{rubric_list_string}` slot: items are numbered `1. criterion`. A `[±p points]` tag is added only when a rubric has negative items, as HealthBench does. RubricHub has none, and its weights range over 0–30.
- **Prompts written for this replication.** Prompts marked OURS in `src/rrh/grading/prompts.py` are not printed in the paper:
  - the H.1 user message
  - the taxonomy classifier
  - the I.1 user/JSON template
  - the science variant of I.1 (domain words swapped)
  - the rubric-category classifier
  - claim extraction

  The A.1, A.2, H.1 (system) and I.1 (system) prompts are verbatim from the paper. Line wrapping was reflowed from the PDF.

## Data

- **Source.** Data comes from `sojuL/RubricHub_v1`, using `RuRL/rurbichub_v1_{Medical,Science}.parquet` (about 29.7k and 29.4k rows).
- **Source filter.** The paper keeps only II-medical-reasoning prompts for medical and RaR-science / ResearchQA / MegaScience for science. The RL files have no source column. `--source_filter` recovers sources by joining prompt text against the SFT files' `source` column. Only part of the data can be traced this way, so report the kept counts the script prints.
- **Splits.** Data is subsampled with seed 0 to the paper's sizes (12,519 / 1,391 and 19,806 / 2,201). If fewer prompts survive the filter, the same 90/10 ratio is kept.
- **Evaluation subsets.**
  - eval300 (seed 0) comes from test.
  - The Sec. 4 set has 500 prompts from test (seed 2).
  - Verifier selection uses 1,000 train prompts (seed 1).
- **HealthBench.** We use the 5,000-example public jsonl and draw 1,000 with `random.Random(0).sample` (as simple-evals does). Grading uses simple-evals' own `GRADER_TEMPLATE` with `gpt-4.1-2025-04-14`, one call per item. Scores are reported both as the original value (Σmet / Σpositive, with the mean clipped to [0,1]) and the flipped value (App. I.5).

## Things that may differ by the time you run

The panel model IDs (`gpt-5.4`, `gemini-3-pro-preview`, `claude-opus-4-6`) and every price marked VERIFY in `configs/models.yaml` must be checked against the providers before budgeting. `gpt-4o-mini` is marked deprecated by OpenAI; if it is retired, the weak verifier has to be re-chosen from the low end of Table 4, and that must be reported.
