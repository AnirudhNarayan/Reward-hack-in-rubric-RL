"""Static matplotlib figures mirroring the paper's Figures 1-4 for one run.

Colors: the validated categorical palette, assigned in fixed order by entity
(reference panel = slot 1, training verifier = slot 2, self-gap = slot 3), never by rank.
"""

from __future__ import annotations

from pathlib import Path

INK, INK2, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
REF, TRAIN, GAP = SERIES[0], SERIES[1], SERIES[2]


def _style(ax, title, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    ax.set_xlabel("Training step", color=INK2)
    ax.set_ylabel(ylabel, color=INK2)
    ax.grid(axis="y", color="#e6e5e1", linewidth=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=INK2)


def _band(ax, xs, rows, key, lo, hi, color, label):
    ys = [r[key] for r in rows]
    ax.plot(xs, ys, color=color, linewidth=2, marker="o", markersize=4, label=label)
    if lo in rows[0]:
        ax.fill_between(xs, [r[lo] for r in rows], [r[hi] for r in rows], color=color, alpha=0.15,
                        linewidth=0)


def plot_run(rep: dict, out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    name = rep["run"]

    rw = rep.get("rewards") or []
    if rw:
        fig, ax = plt.subplots(figsize=(6, 4), facecolor=SURFACE)
        xs = [r["step"] for r in rw]
        _band(ax, xs, rw, "proxy_reward", "proxy_ci_lo", "proxy_ci_hi", TRAIN,
              f"Training verifier ({rep['train_judge']})")
        _band(ax, xs, rw, "ref_reward", "ref_ci_lo", "ref_ci_hi", REF, "Reference panel (consensus)")
        _style(ax, f"Eval-set reward - {name}", "Rubric reward")
        ax.legend(frameon=False, labelcolor=INK)
        fig.tight_layout()
        fig.savefig(out_dir / "rewards.png", dpi=160)
        plt.close(fig)

    er = rep.get("exploitation") or []
    if er:
        fig, ax = plt.subplots(figsize=(6, 4), facecolor=SURFACE)
        xs = [r["step"] for r in er]
        anchor = er[0]["exploitation_rate"]
        ax.plot(xs, [100 * r["delta_vs_first"] for r in er], color=TRAIN, linewidth=2, marker="o",
                markersize=4)
        ax.fill_between(xs, [100 * (r["ci_lo"] - anchor) for r in er],
                        [100 * (r["ci_hi"] - anchor) for r in er], color=TRAIN, alpha=0.15,
                        linewidth=0)
        ax.axhline(0, color=INK2, linewidth=0.8)
        _style(ax, f"Change in P(incorrect | newly credited), anchor {100 * anchor:.1f}%",
               "Delta vs first checkpoint (pp)")
        fig.tight_layout()
        fig.savefig(out_dir / "exploitation.png", dpi=160)
        plt.close(fig)

    sg = rep.get("self_gap") or []
    if sg:
        fig, ax = plt.subplots(figsize=(6, 4), facecolor=SURFACE)
        xs = [r["step"] for r in sg]
        _band(ax, xs, sg, "self_gap", "ci_lo", "ci_hi", GAP, "Self-internalization gap")
        cmp = rep.get("self_gap_vs_rewards", {})
        for key, color, ls, lab in (("argmax_consensus", REF, "--", "consensus argmax"),
                                    ("argmax_train_verifier", INK2, ":", "train-verifier argmax"),
                                    ("argmax_self_gap", GAP, "-.", "self-gap argmax")):
            if key in cmp:
                ax.axvline(cmp[key], color=color, linestyle=ls, linewidth=1.2, label=lab)
        r = cmp.get("pearson_gap_vs_consensus")
        _style(ax, "Self-gap" + (f"  (r vs consensus = {r:.2f})" if r is not None else ""),
               "Delta(t) = l_prompt - l_cond")
        ax.legend(frameon=False, labelcolor=INK, fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / "self_gap.png", dpi=160)
        plt.close(fig)

    tx = rep.get("taxonomy")
    if tx and tx.get("per_step_counts"):
        labels = ["A.1", "A.2", "B.1", "B.2", "C.1", "C.2"]
        steps = sorted(tx["per_step_counts"], key=int)
        fig, ax = plt.subplots(figsize=(8, 4), facecolor=SURFACE)
        bottom = [0] * len(steps)
        for lab, color in zip(labels, SERIES):
            v = [tx["per_step_counts"][s].get(lab, 0) for s in steps]
            ax.bar([int(s) for s in steps], v, bottom=bottom, width=18, color=color, label=lab,
                   edgecolor=SURFACE, linewidth=1)
            bottom = [b + x for b, x in zip(bottom, v)]
        _style(ax, "Verifier failure sub-modes per checkpoint", "Exploited rubrics")
        ax.legend(frameon=False, ncol=6, fontsize=8, labelcolor=INK)
        fig.tight_layout()
        fig.savefig(out_dir / "taxonomy.png", dpi=160)
        plt.close(fig)
