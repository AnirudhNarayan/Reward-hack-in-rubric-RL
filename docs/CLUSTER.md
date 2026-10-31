# UCF clusters (Newton = GPU, Stokes = CPU/GPU)

## Status: NOT YET INSPECTED

The code was written on 2026-10-31 on a workstation that **cannot reach either cluster**:
`newton.ist.ucf.edu` does not resolve without the UCF VPN, and the machine has no SSH keys. So
nothing below is confirmed yet, and no script hard-codes a partition, account, GPU type, module
name or path. Every cluster-specific value lives in `scripts/cluster.env`, which is git-ignored and
copied from `scripts/cluster.env.example`.

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
| GPU partition(s) | ? | ? | `GPU_PARTITION` |
| CPU partition | ? | ? | `CPU_PARTITION` (prep, report, judge validation) |
| Account / QOS | ? | ? | `ACCOUNT`, `QOS` |
| GPU type, memory, GPUs/node | ? | ? | `GPU_TYPE`, `GPUS_PER_NODE`, sizing below |
| Interconnect (IB / NVLink) | ? | ? | multi-node training speed |
| Max walltime | ? | ? | `MAX_WALLTIME` (training segment length) |
| CUDA / conda module names | ? | ? | `MODULES` |
| Scratch path and quota | ? | ? | `SCRATCH_ROOT` (checkpoints: about 1.4 TB per run, DECISIONS.md D16) |
| Shared filesystem between Stokes and Newton? | ? | ? | whether prep can run on Stokes |
| Compute-node internet | ? | ? | `COMPUTE_HAS_INTERNET` (HF downloads, optional API judges) |

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
| Gemma-3-27B | TP = 2 |
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

Put `HF_TOKEN=...` in `~/.env` (chmod 600). It is needed for the gated Llama and Gemma downloads.
