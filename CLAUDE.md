# CLAUDE.md: rubric-rh

This repo replicates "Reward Hacking in Rubric-Based Reinforcement Learning" (Scale AI, arXiv:2605.12474v1) as a PhD-thesis baseline (UCF, Mental Health Project). It runs on the UCF Stokes / Newton SLURM clusters. A full paper summary is in `docs/rubric_conversations.txt`, PART 1. Every deviation and choice is in `REPLICATION_NOTES.md`; keep that file updated.

## RP-OPD ("OPD Before RL", stage 1 only)
- Plan: `IMPLEMENTATION_PLAN.md`. Every unstated choice and model substitution: `DECISIONS.md` (cite D-IDs in code/configs; keep it updated).
- Code: `src/rrh/opd/` (contexts, loss, batching, dist/FSDP2, rollout, checkpoint, trainer, evaluate), `src/rrh/grading/{opd_prompts,opd_grader,citations}.py`, `src/rrh/data/{opd_data,researchqa}.py`, `src/rrh/results/layout.py`, `src/rrh/analysis/{opd_report,judge_agreement}.py`.
- Jobs: `scripts/slurm/opd/*.slurm`; one-command launch `scripts/submit_opd_pipeline.sh`. Cluster values only in `scripts/cluster.env` (see `docs/CLUSTER.md`).
- Also forbidden: DeepSeek, Kimi/Moonshot, GLM/Zhipu, Yi, InternLM, Baichuan, MiniMax, or any other Chinese-developed model, in any role.
- Never prune checkpoints without asking (`train.keep_resume_last` stays null unless the user approves).

## Hard rules
- **Never use Qwen models**, either as the policy or as a verifier.
- **Never run training, vLLM or evaluation on the login node.** Always use `sbatch` or `srun`.
- API keys live in `~/.env` (loaded by `rrh.utils.io.load_env`). Never print or commit them.
- Cluster paths go in `scripts/env.sh` and `configs/local.yaml`. Both are git-ignored.
- Checkpoints and data go on scratch, not `$HOME`.
- Commit fixes to git. Ask before deleting checkpoints or data.

## Layout
- `src/rrh/grading/`: grading.
  - `prompts.py`: A.1, A.2, H.1 and I.1 verbatim, plus our own prompts, which are labelled.
  - `grader.py`: one call grades every rubric item.
  - `reward.py`: R, consensus.
  - `clients.py`: OpenAI, vLLM, Gemini and Anthropic clients, with caching and cost tracking.
  - `grade_cli.py`: resumable batch grading.
- `src/rrh/training/reward_fn.py`: the verl `compute_score` (async). It reads the rubric from `extra_info`.
- `src/rrh/data/`: RubricHub splits (JSONL + verl parquet) and the HealthBench-1000 subset.
- `src/rrh/eval/`:
  - vLLM generation in two contexts: prompt-only and rubric-conditioned.
  - Self-gap log-prob scoring.
  - HealthBench grading.
- `src/rrh/analysis/`:
  - Exploitation rate, reward curves, self-gap, taxonomy.
  - Pairwise rubric-free judging, rubric categories, claims and fixed effects.
  - Verifier selection.
  - `report.py` (Sec. 3) and `rubric_hacking.py` (Sec. 4).
- `scripts/slurm/`: `prepare_data`, `serve_verifier`, `smoke_test`, `train_grpo` (2-node Ray), `eval_checkpoints` (GPU), `grade_run` (API).
- `configs/`: `models.yaml` (every LLM alias), `grpo_base.yaml` (Table 3 mapping), `cost_plan.yaml`.

## Pipeline per run
`train_grpo` → `eval_checkpoints` → `grade_run` → `python -m rrh.analysis.report [--taxonomy]` → Sec. 4: `eval_checkpoints` with `SEC4_STEPS` → `grade_cli` on `gen_sec4` → `rrh.analysis.rubric_hacking`.

## Tests
`pytest -q` runs pure-Python tests with no GPU or network. Run them before every commit.

## Open items (ask the user)
- GPU type and count per node on Stokes and Newton, max walltime, partition and account names.
- Whether compute nodes have internet. The weak verifier and the panel are API calls made from inside training and grading jobs.
- API budget. Run `python -m rrh.utils.cost_cli`; prices marked VERIFY are placeholders.
