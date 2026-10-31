"""Results directory layout shared by every job (training, eval, grading, analysis).

results/<experiment>/<model>__<judge>__<dataset>__seed<k>__<YYYYMMDD-HHMM>/
    config.yaml        fully resolved config + git commit hash
    logs/              stdout/stderr, SLURM logs
    metrics/           train.jsonl (per step); eval/<evalset>/<judge>/step_XXXXXX.json
    generations/       <evalset>/step_XXXXXX.jsonl
    judgments/         <judge>/<evalset>/step_XXXXXX.jsonl (raw judge outputs; reusable)
    checkpoints/       symlink to scratch (never in $HOME or git)
    figures/ tables/

results/INDEX.csv has one row per run, upserted by run_id.
"""

from __future__ import annotations

import csv
import datetime as _dt
import os
import re
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path

import yaml

SUBDIRS = ("logs", "metrics", "generations", "judgments", "figures", "tables")
INDEX_COLUMNS = ["run_id", "experiment", "model", "judge", "dataset", "seed", "created", "status",
                 "git_commit", "path", "lr", "global_batch", "topk", "loss_mode", "total_steps",
                 "last_step", "best_step", "best_score", "final_score", "teacher_ref_score",
                 "shortcut_aware_score", "updated"]
REPO_ROOT = Path(__file__).resolve().parents[3]


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", str(s).split("/")[-1]).strip("-").lower()


def make_run_id(model: str, judge: str, dataset: str, seed: int, stamp: str | None = None) -> str:
    stamp = stamp or _dt.datetime.now().strftime("%Y%m%d-%H%M")
    return f"{_slug(model)}__{_slug(judge)}__{_slug(dataset)}__seed{int(seed)}__{stamp}"


def git_commit(repo: Path = REPO_ROOT) -> str:
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True,
                             text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=repo,
                               capture_output=True, text=True).stdout.strip()
        return sha + ("-dirty" if dirty else "")
    except Exception:  # noqa: BLE001 - no git on some compute nodes
        return os.environ.get("RRH_GIT_COMMIT", "unknown")


class RunDir:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.run_id = self.path.name
        self.experiment = self.path.parent.name
        self.results_root = self.path.parent.parent

    # ---- creation ----
    def ensure(self) -> "RunDir":
        for d in SUBDIRS:
            (self.path / d).mkdir(parents=True, exist_ok=True)
        return self

    def link_checkpoints(self, scratch_dir: str | Path) -> Path:
        """checkpoints/ -> scratch. Falls back to a pointer file where symlinks are not allowed."""
        scratch_dir = Path(scratch_dir)
        scratch_dir.mkdir(parents=True, exist_ok=True)
        link = self.path / "checkpoints"
        if link.is_symlink() or link.exists():
            return self.checkpoint_dir()
        try:
            link.symlink_to(scratch_dir.resolve(), target_is_directory=True)
        except OSError:
            (self.path / "checkpoints.path").write_text(str(scratch_dir.resolve()))
        return self.checkpoint_dir()

    def checkpoint_dir(self) -> Path:
        ptr = self.path / "checkpoints.path"
        if ptr.exists():
            return Path(ptr.read_text().strip())
        return self.path / "checkpoints"

    def write_config(self, cfg: dict) -> None:
        cfg = dict(cfg)
        cfg["_meta"] = {"run_id": self.run_id, "experiment": self.experiment,
                        "git_commit": git_commit(),
                        "written": _dt.datetime.now().isoformat(timespec="seconds"),
                        "slurm_job_id": os.environ.get("SLURM_JOB_ID")}
        path = self.path / "config.yaml"
        if path.exists():  # resumed segment: keep the original, log the new one alongside
            path = self.path / "logs" / f"config.{os.environ.get('SLURM_JOB_ID', 'local')}.yaml"
        path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")

    def read_config(self) -> dict:
        return yaml.safe_load((self.path / "config.yaml").read_text(encoding="utf-8"))

    # ---- standard file locations ----
    def train_metrics(self) -> Path:
        return self.path / "metrics" / "train.jsonl"

    def eval_metrics(self, evalset: str, judge: str, step: int | str) -> Path:
        return self.path / "metrics" / "eval" / evalset / judge / f"{step_name(step)}.json"

    def generations(self, evalset: str, step: int | str) -> Path:
        return self.path / "generations" / evalset / f"{step_name(step)}.jsonl"

    def judgments(self, judge: str, evalset: str, step: int | str) -> Path:
        return self.path / "judgments" / judge / evalset / f"{step_name(step)}.jsonl"


def step_name(step: int | str) -> str:
    return f"step_{int(step):06d}" if str(step).isdigit() else str(step)


def create_run(results_root: str | Path, experiment: str, model: str, judge: str, dataset: str,
               seed: int, run_id: str | None = None) -> RunDir:
    run_id = run_id or make_run_id(model, judge, dataset, seed)
    rd = RunDir(Path(results_root) / experiment / run_id).ensure()
    return rd


@contextmanager
def _file_lock(path: Path, timeout: float = 120.0):
    lock = path.with_suffix(path.suffix + ".lock")
    t0 = time.time()
    while True:
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError:
            if time.time() - t0 > timeout:   # stale lock from a killed job
                try:
                    os.remove(lock)
                except FileNotFoundError:
                    pass
            time.sleep(0.2)
    try:
        yield
    finally:
        os.close(fd)
        try:
            os.remove(lock)
        except FileNotFoundError:
            pass


def read_index(results_root: str | Path) -> list[dict]:
    p = Path(results_root) / "INDEX.csv"
    if not p.exists():
        return []
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def update_index(results_root: str | Path, row: dict) -> None:
    """Upsert one run row (keyed by run_id) in results/INDEX.csv. Unknown keys are ignored."""
    p = Path(results_root) / "INDEX.csv"
    p.parent.mkdir(parents=True, exist_ok=True)
    with _file_lock(p):
        rows = read_index(results_root)
        row = {k: v for k, v in row.items() if k in INDEX_COLUMNS}
        row["updated"] = _dt.datetime.now().isoformat(timespec="seconds")
        for r in rows:
            if r["run_id"] == row["run_id"]:
                r.update({k: ("" if v is None else v) for k, v in row.items()})
                break
        else:
            rows.append({c: ("" if row.get(c) is None else row.get(c, "")) for c in INDEX_COLUMNS})
        tmp = p.with_suffix(".csv.tmp")
        with open(tmp, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=INDEX_COLUMNS)
            w.writeheader()
            for r in rows:
                w.writerow({c: r.get(c, "") for c in INDEX_COLUMNS})
        os.replace(tmp, p)


def index_row_from_config(rd: RunDir, cfg: dict, status: str) -> dict:
    g = lambda *ks, d=None: _get(cfg, ks, d)  # noqa: E731
    return {"run_id": rd.run_id, "experiment": rd.experiment, "model": g("student", "name"),
            "judge": g("eval", "curve_judge"), "dataset": g("data", "name"), "seed": g("seed"),
            "created": rd.run_id.rsplit("__", 1)[-1], "status": status, "git_commit": git_commit(),
            "path": str(rd.path), "lr": g("optim", "lr"), "global_batch": g("train", "global_batch"),
            "topk": g("loss", "topk"), "loss_mode": g("loss", "mode"),
            "total_steps": g("train", "total_steps")}


def _get(d, keys, default=None):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d
