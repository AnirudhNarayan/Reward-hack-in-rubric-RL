# rubric-rh

This is a replication of *Reward Hacking in Rubric-Based Reinforcement Learning* (Mahmoud et al., Scale AI, arXiv:2605.12474). It uses a non-Qwen policy (Llama-3.1-8B-Instruct by default), verl GRPO and SLURM. Every deviation and choice is listed in `REPLICATION_NOTES.md`.

## Paper → code

| Paper | Code |
|---|---|
| §2.1 reward R | `rrh.grading.reward.rubric_reward` |
| App. A.1 grader (verifier + panel) | `rrh.grading.prompts`, `rrh.grading.grader` |
| §2.2 R_proxy / R_ref (unanimous panel) | `rrh.analysis.exploitation.reward_curves` |
| §2.3 Tables 1 and 4 (verifier selection) | `rrh.analysis.verifier_selection` |
| App. B Table 3 (GRPO) | `configs/grpo_base.yaml`, `scripts/train_grpo.sh` |
| §3.1 exploitation rate, Fig. 1 | `rrh.analysis.exploitation`, `rrh.analysis.report` |
| §3.2 / App. H taxonomy, Fig. 3 | `rrh.analysis.taxonomy` (`report --taxonomy`) |
| §3.3 self-internalization gap, Fig. 4, App. C | `rrh.eval.self_gap_cli`, `rrh.analysis.self_gap` |
| App. G.1 rubric-conditioned check, G.3 length | `report.py` (`rubric_conditioned`, `length`) |
| App. F HealthBench, Fig. 2, Table 7 | `rrh.data.healthbench`, `rrh.eval.healthbench_grade` |
| §4.1 / App. I.1–I.2 rubric-free pairwise judging | `rrh.analysis.pairwise` |
| §4.2 rubric categories (Tables 2, 13, 14) | `rrh.analysis.rubric_types` |
| §4.3 / App. I.3–I.4 claims, fixed effects | `rrh.analysis.claims` |
| App. E panel vs human macro-F1 | `rrh.analysis.verifier_agreement.macro_f1`. Needs expert labels, which are not public. |

## Quick start (cluster)

```bash
git clone https://github.com/AnirudhNarayan/Reward-hack-in-rubric-RL.git rubric-rh && cd rubric-rh
bash scripts/setup_env.sh                       # conda env, verl 0.9.1, simple-evals
$EDITOR scripts/env.sh configs/local.yaml       # scratch paths, module loads
# ~/.env: OPENAI_API_KEY=..., GEMINI_API_KEY=..., ANTHROPIC_API_KEY=..., HF_TOKEN=... (chmod 600)
pytest -q
python -m rrh.utils.cost_cli                    # budget before anything expensive

sbatch -p <cpu> scripts/slurm/prepare_data.slurm                       # Phase 0
DOMAIN=medical sbatch -p <gpu> --gres=gpu:4 scripts/slurm/smoke_test.slurm   # 1-node smoke run

# Main runs: {medical, science} x {gpt-4o-mini, gpt-oss-120b}
sbatch -p <gpu> --gres=gpu:2 scripts/slurm/serve_verifier.slurm        # strong verifier only
DOMAIN=medical VERIFIER=gpt-4o-mini sbatch -p <gpu> --gres=gpu:8 scripts/slurm/train_grpo.slurm
RUN_NAME=medical_gpt-4o-mini_Llama-3.1-8B-Instruct DOMAIN=medical \
  SEC4_STEPS="0 75 150 225 300 375 450 475" sbatch -p <gpu> --gres=gpu:1 scripts/slurm/eval_checkpoints.slurm
RUN_NAME=... DOMAIN=medical VERIFIER=gpt-4o-mini sbatch -p <cpu> scripts/slurm/grade_run.slurm
python -m rrh.analysis.report --run_dir $RUNS_DIR/<run> --data $DATA_DIR/medical/eval300.jsonl \
  --train_judge gpt-4o-mini --taxonomy
```

All LLM calls are cached in SQLite, so re-running an analysis does not pay for the same call twice.

---

# RP-OPD: stage 1 of "OPD Before RL" (arXiv:2610.02781)

This is rubric-privileged on-policy distillation. The student (Llama-3.1-8B-Instruct) never sees
the rubric. It matches the top-256 next-token distribution of a rubric-aware teacher
(Llama-3.1-70B-Instruct + App. A.2 rubric system prompt), computed on the student's own sampled
responses. The loss is forward KL (Eq. 4).

- `IMPLEMENTATION_PLAN.md`: maps each paper section, equation and table to code.
- `DECISIONS.md`: every unstated choice, the judge substitutions, and cost and GPU estimates.
- `docs/CLUSTER.md`: cluster inspection and sizing.

```bash
# once per cluster (login node): inspect, then fill scripts/cluster.env
bash scripts/cluster/inspect_cluster.sh newton
cp scripts/cluster.env.example scripts/cluster.env && $EDITOR scripts/cluster.env
pytest -q                                                   # 59 CPU tests, no network

bash scripts/submit_opd_pipeline.sh --smoke                 # 1 GPU, Llama-3.2-1B, 4 steps, end to end
bash scripts/submit_opd_pipeline.sh --dataset healthbench   # full DAG: prep -> train segments ->
                                                            # gen + teacher ref -> judges -> validation -> report
bash scripts/submit_opd_pipeline.sh --dataset all           # HealthBench, ResearchQA, RubricHub Science
bash scripts/submit_opd_pipeline.sh --dataset healthbench --variant rubric_blind      # Fig. 2 control
bash scripts/submit_opd_pipeline.sh --dataset rubrichub_science --variant reverse_kl  # App. B.4 ablation
```

Results go to
`results/<experiment>/<model>__<judge>__<dataset>__seed<k>__<stamp>/`, which contains:
- `config.yaml`, the resolved config plus the git commit;
- `logs/`, `metrics/`, `generations/`, `judgments/`, `figures/`, `tables/`;
- `checkpoints/`, a symlink to scratch.

`results/INDEX.csv` has one row per run. `python -m rrh.analysis.opd_report` rebuilds every figure
and table, including `results/paper/`.

**Stage 2 (not implemented yet):** pick a checkpoint from `stage2_handoff.json` and run
`POLICY=<run>/checkpoints/hf/step_000980 bash scripts/train_grpo.sh`. Switch the reward to Eq. (1)
first (DECISIONS.md D20).
