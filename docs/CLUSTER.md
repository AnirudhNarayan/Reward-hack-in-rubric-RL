# UCF clusters (Newton = GPU, Stokes = CPU/GPU)

## Status: Newton inspected 2026-10-31 (metadata); Stokes not yet

`inspect_cluster.sh newton` was run on the login node with `SKIP_SRUN=1`. Confirmed Newton
values are in the table below and in `scripts/cluster.newton.env`; install them with
`cp scripts/cluster.newton.env scripts/cluster.env`. The `srun` probe on `highgpu` (evc101) confirmed
the H100 80GB HBM3, InfiniBand, and compute-node internet (huggingface.co HTTP 200). Still to
confirm: the exact Lustre directory (`SCRATCH_ROOT`). Stokes is not needed for stage 1: every job runs on Newton.

Newton partitions (`sinfo`, `scontrol show partition`):

| Partition | Nodes | Per node | MaxTime | Access for `an548089` |
|---|---|---|---|---|
| `highgpu` | evc[101-104] | 8x H100 SXM 80GB (`nvidia_h100_80gb_hbm3`), 128 CPUs, ~1 TB | unlimited (default 10 min) | yes, account `abedi` |
| `normal` (default) | ~40 | 2x H100 PCIe 80GB (`nvidia_h100_pcie`) or 2x V100 16/32GB; 32-64 CPUs | unlimited | yes |
| `preemptable` | same as `normal` | same | 2 days | yes (QOS preemptable) |
| `short` | evc105 | 4x H200 NVL (`nvidia_h200_nvl`), 192 CPUs | 4 h | no (AllowAccounts=arcc) |
| `ucfit` | evc105 | same | unlimited | no (arcc, gmartin, ucfit) |

Accounts: `abedi` (QOS normal, ood, preemptable; max 50 jobs) and `course_cap6614`.
`highgpu` nodes are shared (`OverSubscribe=FORCE:4`) and the partition is open to many groups,
so expect queue waits for 2-node jobs.
`preemptable` jobs (PriorityTier 1, PreemptMode CANCEL) are cancelled by `normal` jobs
(PriorityTier 100) on the same nodes. `highgpu` shares no nodes with another partition, so
training there is not preempted; never put training on `preemptable`.

## 1. Inspect each cluster (about 10 minutes, once)

On each login node:

```bash
git clone https://github.com/AnirudhNarayan/Reward-hack-in-rubric-RL.git rubric-rh && cd rubric-rh
bash scripts/cluster/inspect_cluster.sh newton
bash scripts/cluster/inspect_cluster.sh stokes
```

Each run writes `docs/cluster/<name>.raw.txt`, which collects:
- `sinfo` and `scontrol show partition`: partitions, MaxTime, nodes, GRES;
- `sacctmgr`: accounts and QOS;
- `module avail`, filtered to CUDA, conda and Python;
- `df` and quota for `$HOME` and scratch candidates;
- one 5-minute `srun` per GPU partition, which runs `nvidia-smi` and `nvidia-smi topo -m`, checks
  for InfiniBand, and probes internet access from the compute node to huggingface.co and
  api.openai.com.

These are only probes (`srun` with a 5-minute limit); no training or vLLM runs on the login node.

## 2. Fill in this table and `scripts/cluster.env`

| Item | Newton | Stokes | Used by |
|---|---|---|---|
| GPU partition(s) | `highgpu` (all GPU jobs) | ? | `GPU_PARTITION` |
| CPU partition | `normal`, no GPU requested (Newton has no CPU-only partition) | ? | `CPU_PARTITION` (prep, report, judge validation) |
| Account / QOS | `abedi` / default (normal) | ? | `ACCOUNT`, `QOS` |
| GPU type, memory, GPUs/node | `nvidia_h100_80gb_hbm3`, 80 GB, 8 | ? | `GPU_TYPE`, `GPUS_PER_NODE`, sizing below |
| Interconnect (IB / NVLink) | HDR InfiniBand (node feature `hdr`); NVLink within highgpu nodes (SXM) | ? | multi-node training speed |
| Max walltime | unlimited on highgpu (default 10 min, so `--time` is always set); 48 h segments | ? | `MAX_WALLTIME` (training segment length) |
| CUDA / conda module names | `anaconda/anaconda-2024.10`, `cuda/cuda-12.6.0` | ? | `MODULES` |
| Scratch path and quota | Lustre `/lustre/fs1`: 1000 GB soft / 1010 GB hard, 1M files (`lfs quota`) | ? | `SCRATCH_ROOT` (peak ≈ 660 GB per run, DECISIONS.md D16) |
| Shared filesystem between Stokes and Newton? | ? | ? | whether prep can run on Stokes |
| Compute-node internet | yes (huggingface.co 200 from evc101) | ? | `COMPUTE_HAS_INTERNET` (HF downloads, optional API judges) |

## 3. Sizing guide (pick `TRAIN_NODES` from the GPU type)

Per-GPU memory during RP-OPD on 8B / 70B with FSDP2 over N GPUs:

| Component | Memory per GPU |
|---|---|
| Student fp32 weights + AdamW | 128 GB / N |
| Teacher bf16 | 140 GB / N |
| Activations (gradient checkpointing, `micro_batch_tokens=16384`) | about 8–12 GB |
| vLLM rollout engine while awake (`gpu_memory_utilization`) | 16 GB weights + KV |

| GPUs | Suggested setup |
|---|---|
| 80 GB (H100 / A100-80G) | 2 nodes × 8 (N = 16): about 17 GB of sharded state + vLLM 0.30 × 80. **Default.** |
| 40 GB (A100-40G) | 4 nodes × 8 (N = 32); set `rollout.gpu_memory_utilization=0.45`, `train.micro_batch_tokens=8192` |
| 32 GB V100 | Not recommended: no bf16 or FlashAttention-2, and the 70B teacher needs ≥ 5 nodes |

The judges are served by vLLM inside the grade jobs:

| Judge | Tensor parallel on 80 GB GPUs |
|---|---|
| Llama-3.3-70B | TP = 4 |
| gpt-oss-120b | TP = 2 |

On 40 GB GPUs, double TP.

## 4. Environment

The existing replication env is reused (`bash scripts/setup_env.sh`: conda env `rrh` with
verl 0.9.1, vLLM, torch and flash-attn), so the stage-2 GRPO pipeline and RP-OPD share one
environment. RP-OPD needs:
- torch ≥ 2.4 (FSDP2; ≥ 2.6 preferred);
- vLLM with sleep mode;
- transformers ≥ 4.45.

Check versions after setup with `python -c "import torch, vllm, transformers; print(torch.__version__, vllm.__version__, transformers.__version__)"`.

Put `HF_TOKEN=...` in `~/.env` (chmod 600). It is needed for the gated Llama downloads. Put
`OPENAI_API_KEY=...` there too for the GPT-4o-mini Table 1 judge.
