"""RP-OPD training entry point (launch with torchrun / srun; never on a login node).

    torchrun --nproc_per_node 8 -m rrh.opd.train_cli \
        --config configs/opd/base.yaml configs/opd/healthbench.yaml \
        --run_id "$RUN_ID" --set train.total_steps=1000

Re-running with the same --run_id resumes from the latest complete resume checkpoint.
--deadline (unix seconds; the SLURM scripts pass the job end time) makes the trainer save and
exit cleanly before the walltime.
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path

from rrh.opd.config import load_opd_config
from rrh.opd.dist import barrier, broadcast_object, init_distributed
from rrh.results.layout import (RunDir, create_run, index_row_from_config, make_run_id,
                                update_index)
from rrh.utils.io import load_env


def setup_run(cfg: dict, run_id: str | None, results_root: str) -> RunDir:
    run_id = run_id or make_run_id(cfg["student"]["name"], cfg["eval"]["curve_judge"],
                                   cfg["data"]["name"], cfg.get("seed", 0))
    return create_run(results_root, cfg.get("experiment", "rp_opd"), cfg["student"]["name"],
                      cfg["eval"]["curve_judge"], cfg["data"]["name"], cfg.get("seed", 0), run_id)


def scratch_ckpt_dir(cfg: dict, run: RunDir) -> Path:
    root = cfg.get("paths", {}).get("ckpt_root")
    if not root or "${" in str(root):
        root = os.environ.get("CKPT_ROOT") or str(run.path / "_checkpoints_local")
    return Path(root) / run.experiment / run.run_id


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", nargs="+", required=True)
    ap.add_argument("--set", nargs="*", default=[], dest="overrides")
    ap.add_argument("--run_id", default=os.environ.get("RRH_RUN_ID"))
    ap.add_argument("--results_root", default=os.environ.get("RESULTS_ROOT", "results"))
    ap.add_argument("--deadline", type=float, default=float(os.environ.get("RRH_DEADLINE", 0) or 0))
    a = ap.parse_args(argv)

    load_env()
    di = init_distributed()
    cfg = load_opd_config(a.config, a.overrides)

    run = None
    if di.is_main:
        run = setup_run(cfg, a.run_id, a.results_root)
        ck = run.link_checkpoints(scratch_ckpt_dir(cfg, run))
        cfg.setdefault("paths", {})["ckpt_dir_resolved"] = str(ck)
        run.write_config(cfg)
        update_index(run.results_root, index_row_from_config(run, cfg, "running"))
        print(f"[opd] run dir: {run.path}\n[opd] checkpoints: {ck}", flush=True)
    run_path, ck = broadcast_object((str(run.path), str(run.checkpoint_dir())) if run else None, di)
    run = RunDir(run_path)

    from rrh.opd.trainer import OPDTrainer
    status = "failed"
    try:
        trainer = OPDTrainer(cfg, run, di, Path(ck))
        status = trainer.train(deadline=a.deadline or None)
        return 0
    except Exception:
        traceback.print_exc()
        raise
    finally:
        if di.is_main:
            last = _last_step(run)
            update_index(run.results_root, {"run_id": run.run_id, "status": status, "last_step": last})
        try:
            barrier(di)
        except Exception:  # noqa: BLE001
            pass


def _last_step(run: RunDir):
    p = run.train_metrics()
    if not p.exists():
        return ""
    last = ""
    with open(p, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                last = line
    import json
    return json.loads(last)["step"] if last else ""


if __name__ == "__main__":
    sys.exit(main())
