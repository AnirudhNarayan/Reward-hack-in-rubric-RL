"""Rebuild RP-OPD figures and tables from results/ (CPU only, idempotent).

    python -m rrh.analysis.opd_report --results_root results

Per run (results/<experiment>/<run>/):
  figures/fig2_<evalset>.png       rubric score vs OPD step (curve judge, 95% CI), teacher gold line,
                                   independent shortcut-aware score if graded       (paper Fig. 2)
  figures/entropy_<evalset>.png    top-20 renormalized token entropy vs step         (Fig. 5c / 12b)
  figures/train_curves.png         KL loss, top-1 agreement, entropy, response length
  tables/eval_<evalset>.{csv,tex}  per-checkpoint scores
  stage2_handoff.json              HF checkpoint paths + scores, for choosing OPD@k warm starts
  INDEX.csv row                    best/final/teacher/shortcut-aware scores
Across runs (results/paper/):
  figures/fig2_stage1.png          one panel per dataset; rubric-aware vs rubric-blind OPD + teacher
  figures/fig12_fwd_vs_rev.png     forward top-K vs reverse-KL k3 (if both runs exist)
  tables/table1_stage1.{csv,tex}   "RP-OPD (ours, stage 1 only)" row + the paper's values
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from rrh.results.layout import RunDir, update_index
from rrh.utils.io import read_jsonl

PAPER_TABLE1_STAGE1 = {  # Table 1, "RP-OPD (ours, stage 1 only)", shortcut-aware GPT-4o-mini
    ("llama-3.1-8b", "healthbench"): 0.501,
    ("qwen2.5-3b", "healthbench"): 0.495, ("qwen2.5-3b", "researchqa"): 0.594,
    ("qwen2.5-3b", "rubrichub-science"): 0.642,
    ("qwen2.5-7b", "healthbench"): 0.582, ("qwen2.5-7b", "researchqa"): 0.614,
    ("qwen2.5-7b", "rubrichub-science"): 0.705}
PAPER_FIG2_TEACHER = {"healthbench": 0.831, "researchqa": 0.980, "rubrichub-science": 0.917}


def _step_int(s: str):
    return int(s[5:]) if s.startswith("step_") and s[5:].isdigit() else None


def load_eval(run: RunDir) -> dict:
    """-> {evalset: {judge_dir: {step_name: metrics}}}"""
    out: dict = defaultdict(lambda: defaultdict(dict))
    base = run.path / "metrics" / "eval"
    if base.exists():
        for f in base.glob("*/*/*.json"):
            out[f.parent.parent.name][f.parent.name][f.stem] = json.loads(f.read_text())
    return out


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def run_report(run: RunDir) -> dict:
    import pandas as pd
    cfg = run.read_config()
    curve = cfg["eval"]["curve_judge"]
    indep = cfg["eval"].get("independent_judge")
    indep_dir = f"{indep}-shortcut" if indep else None
    ev = load_eval(run)
    plt = _plt()
    summary: dict = {"run_id": run.run_id}
    for evalset, judges in ev.items():
        rows = []
        steps = sorted({s for j in judges.values() for s in j if _step_int(s) is not None}, key=_step_int)
        for s in steps:
            r = {"step": _step_int(s)}
            c = judges.get(curve, {}).get(s)
            if c:
                r.update({"score": c["score_mean"], "ci_lo": c["score_ci95"][0], "ci_hi": c["score_ci95"][1],
                          "entropy_top20": c.get("entropy_top20_mean"), "mean_tokens": c.get("mean_tokens"),
                          "truncated_frac": c.get("truncated_frac")})
            i = judges.get(indep_dir, {}).get(s) if indep_dir else None
            if i:
                r.update({"shortcut_aware": i["score_mean"], "indep_standard": i.get("score_standard_mean"),
                          "indep_pre_citation": i.get("score_pre_citation_mean")})
            rows.append(r)
        df = pd.DataFrame(rows)
        teacher = judges.get(curve, {}).get("teacher")
        teacher_ind = judges.get(indep_dir, {}).get("teacher") if indep_dir else None
        if not df.empty:
            df.to_csv(run.path / "tables" / f"eval_{evalset}.csv", index=False)
            (run.path / "tables" / f"eval_{evalset}.tex").write_text(
                df.to_latex(index=False, float_format="%.3f", na_rep="--"))
        if "score" in df:
            fig, ax = plt.subplots(figsize=(5, 3.4))
            ax.plot(df["step"], df["score"], marker="o", ms=3, label=f"RP-OPD ({curve})")
            ax.fill_between(df["step"], df["ci_lo"], df["ci_hi"], alpha=0.2)
            if "shortcut_aware" in df:
                d2 = df.dropna(subset=["shortcut_aware"])
                ax.plot(d2["step"], d2["shortcut_aware"], marker="s", ms=3, ls=":",
                        label=f"shortcut-aware ({indep})")
            if teacher:
                ax.axhline(teacher["score_mean"], color="goldenrod", ls="--",
                           label=f"rubric-informed teacher {teacher['score_mean']:.3f}")
            ax.set_xlabel("OPD step")
            ax.set_ylabel("Rubric score")
            ax.set_title(f"{cfg['data']['name']} / {evalset}")
            ax.legend(fontsize=7)
            fig.tight_layout()
            fig.savefig(run.path / "figures" / f"fig2_{evalset}.png", dpi=150)
            plt.close(fig)
        if "entropy_top20" in df and df["entropy_top20"].notna().any():
            fig, ax = plt.subplots(figsize=(5, 3.4))
            ax.plot(df["step"], df["entropy_top20"], marker="o", ms=3, label="eval generations")
            tm = run.train_metrics()
            if tm.exists():
                t = pd.DataFrame(read_jsonl(tm))
                ax.plot(t["step"], t["entropy_top20"].rolling(10, min_periods=1).mean(), alpha=0.6,
                        label="train rollouts (10-step avg)")
            ax.set_xlabel("OPD step")
            ax.set_ylabel("Token entropy (nats, top-20)")
            ax.legend(fontsize=7)
            fig.tight_layout()
            fig.savefig(run.path / "figures" / f"entropy_{evalset}.png", dpi=150)
            plt.close(fig)
        if "score" in df and df["score"].notna().any():
            d = df.dropna(subset=["score"])
            best = d.loc[d["score"].idxmax()]
            final = d.iloc[-1]
            summary[evalset] = {
                "best_step": int(best["step"]), "best_score": float(best["score"]),
                "final_step": int(final["step"]), "final_score": float(final["score"]),
                "teacher_score": teacher["score_mean"] if teacher else None,
                "teacher_shortcut_aware": teacher_ind["score_mean"] if teacher_ind else None,
                "final_shortcut_aware": float(final["shortcut_aware"]) if "shortcut_aware" in final
                and final["shortcut_aware"] == final["shortcut_aware"] else None,
                "step0_score": float(d.iloc[0]["score"]) if int(d.iloc[0]["step"]) == 0 else None}
    _train_plot(run, plt)
    _handoff(run, summary)
    primary = summary.get("heldout") or next((v for k, v in summary.items() if k != "run_id"), None)
    if primary:
        update_index(run.results_root, {
            "run_id": run.run_id, "best_step": primary["best_step"], "best_score": round(primary["best_score"], 4),
            "final_score": round(primary["final_score"], 4),
            "teacher_ref_score": round(primary["teacher_score"], 4) if primary["teacher_score"] is not None else "",
            "shortcut_aware_score": round(primary["final_shortcut_aware"], 4)
            if primary["final_shortcut_aware"] is not None else ""})
    (run.path / "metrics" / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def _train_plot(run: RunDir, plt) -> None:
    import pandas as pd
    tm = run.train_metrics()
    if not tm.exists():
        return
    t = pd.DataFrame(read_jsonl(tm))
    if t.empty:
        return
    fig, axes = plt.subplots(1, 4, figsize=(14, 3))
    for ax, (col, lab) in zip(axes, [("loss", "top-K forward KL"), ("top1_agree", "top-1 agreement"),
                                      ("entropy_top20", "entropy (top-20)"),
                                      ("response_len_mean", "response tokens")]):
        if col in t:
            ax.plot(t["step"], t[col], alpha=0.35)
            ax.plot(t["step"], t[col].rolling(20, min_periods=1).mean())
        ax.set_title(lab, fontsize=9)
        ax.set_xlabel("OPD step")
    fig.tight_layout()
    fig.savefig(run.path / "figures" / "train_curves.png", dpi=130)
    plt.close(fig)


def _handoff(run: RunDir, summary: dict) -> None:
    from rrh.opd.checkpoint import list_hf_checkpoints
    ck = list_hf_checkpoints(run.checkpoint_dir()) if run.checkpoint_dir().exists() else []
    ev = load_eval(run)
    cfg = run.read_config()
    curve = cfg["eval"]["curve_judge"]
    items = []
    for step, path in ck:
        sc = {es: j.get(curve, {}).get(f"step_{step:06d}", {}).get("score_mean") for es, j in ev.items()}
        items.append({"step": step, "hf_path": str(path), "scores": sc})
    (run.path / "stage2_handoff.json").write_text(json.dumps({
        "run_id": run.run_id, "student": cfg["student"]["path"], "dataset": cfg["data"]["name"],
        "how_to_use": "POLICY=<hf_path> bash scripts/train_grpo.sh  (verl actor_rollout_ref.model.path)",
        "checkpoints": items, "summary": summary}, indent=2))


def paper_outputs(results_root: Path, runs: list[RunDir], summaries: dict[str, dict]) -> None:
    import pandas as pd
    out = results_root / "paper"
    (out / "figures").mkdir(parents=True, exist_ok=True)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    plt = _plt()
    by_ds: dict = defaultdict(list)
    for r in runs:
        try:
            cfg = r.read_config()
        except FileNotFoundError:
            continue
        by_ds[cfg["data"]["name"]].append((r, cfg))
    # Fig. 2 (stage-1 part): rubric-informed vs rubric-blind OPD, teacher reference
    if by_ds:
        fig, axes = plt.subplots(1, len(by_ds), figsize=(4.6 * len(by_ds), 3.4), squeeze=False)
        for ax, (ds, lst) in zip(axes[0], sorted(by_ds.items())):
            for r, cfg in lst:
                if r.experiment not in ("rp_opd", "opd_rubric_blind"):
                    continue
                p = r.path / "tables" / "eval_heldout.csv"
                if not p.exists():
                    continue
                d = pd.read_csv(p).dropna(subset=["score"])
                lab = "OPD, rubric-informed teacher" if cfg["teacher"].get("rubric_in_context", True) \
                    else "OPD, rubric-blind teacher"
                ax.plot(d["step"], d["score"], ls="--", label=f"{lab} (seed {cfg.get('seed', 0)})")
                t = summaries.get(r.run_id, {}).get("heldout", {}).get("teacher_score")
                if t is not None and cfg["teacher"].get("rubric_in_context", True):
                    ax.axhline(t, color="goldenrod", ls="--", lw=1)
                    ax.text(0, t, f" Rubric-informed teacher {t:.3f}", fontsize=7, va="bottom")
            ax.set_title(ds)
            ax.set_xlabel("Training step")
            ax.set_ylabel("Rubric score")
            ax.legend(fontsize=6)
        fig.tight_layout()
        fig.savefig(out / "figures" / "fig2_stage1.png", dpi=150)
        plt.close(fig)
    # Fig. 12: forward vs reverse KL
    fwd = [r for r, c in by_ds.get("rubrichub-science", []) if r.experiment == "rp_opd"]
    rev = [r for r, c in by_ds.get("rubrichub-science", []) if r.experiment == "rp_opd_reverse_kl"]
    if fwd and rev:
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
        for r, lab in ((fwd[-1], "Forward KL (top-K)"), (rev[-1], "Reverse KL (k3)")):
            p = r.path / "tables" / "eval_heldout.csv"
            if p.exists():
                d = pd.read_csv(p).dropna(subset=["score"])
                axes[0].plot(d["step"], d["score"], marker="o", ms=3, label=lab)
                axes[1].plot(d["step"], d["entropy_top20"], marker="o", ms=3, label=lab)
        axes[0].set_ylabel("Rubric score")
        axes[1].set_ylabel("Token entropy (nats, top-20)")
        for ax in axes:
            ax.set_xlabel("Distillation step")
            ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out / "figures" / "fig12_fwd_vs_rev.png", dpi=150)
        plt.close(fig)
    # Table 1, stage-1 row
    rows = []
    for ds, lst in sorted(by_ds.items()):
        for r, cfg in lst:
            if r.experiment != "rp_opd":
                continue
            s = summaries.get(r.run_id, {}).get("heldout")
            if not s:
                continue
            model = cfg["student"]["name"].replace("-instruct", "")
            rows.append({"model": cfg["student"]["name"], "dataset": ds, "seed": cfg.get("seed", 0),
                         "run_id": r.run_id, "final_step": s["final_step"],
                         "shortcut_aware_final": s["final_shortcut_aware"],
                         "curve_judge_final": s["final_score"], "curve_judge_best": s["best_score"],
                         "best_step": s["best_step"], "teacher_curve_judge": s["teacher_score"],
                         "paper_stage1_shortcut_aware": PAPER_TABLE1_STAGE1.get((model, ds)),
                         "paper_teacher_fig2": PAPER_FIG2_TEACHER.get(ds)})
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(out / "tables" / "table1_stage1.csv", index=False)
        (out / "tables" / "table1_stage1.tex").write_text(
            df.drop(columns=["run_id"]).to_latex(index=False, float_format="%.3f", na_rep="--"))
        print(df.to_string(index=False))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_root", default="results")
    ap.add_argument("--experiments", nargs="*", default=["rp_opd", "opd_rubric_blind", "rp_opd_reverse_kl", "smoke"])
    a = ap.parse_args(argv)
    root = Path(a.results_root)
    runs, summaries = [], {}
    for exp in a.experiments:
        d = root / exp
        if not d.exists():
            continue
        for p in sorted(d.iterdir()):
            if (p / "config.yaml").exists():
                rd = RunDir(p)
                runs.append(rd)
                try:
                    summaries[rd.run_id] = run_report(rd)
                except Exception as e:  # noqa: BLE001 - one broken run must not block the rest
                    print(f"[report] {rd.run_id}: {type(e).__name__}: {e}")
    paper_outputs(root, runs, summaries)


if __name__ == "__main__":
    main()
