"""Paper-style figures, written to figures/.

    python -m ledger.figures
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ledger.report import load  # noqa: E402

OUT = Path("figures")
KS = [1, 2, 5, 10, 20, -1]
GREY, PINK, RED, GREEN = "0.55", "#d98aa6", "#a3133f", "#1f5f4a"

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral"], "mathtext.fontset": "stix",
    "font.size": 9, "axes.labelsize": 9, "legend.fontsize": 7.5, "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
    "ytick.major.width": 0.6, "svg.fonttype": "path", "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03, "axes.spines.top": False, "axes.spines.right": False,
})


def curve(ax, df: pd.DataFrame, value: str, label: str, color: str, ls: str = "-", marker: str = "o",
          band: bool = True) -> None:
    g = df.groupby("k")[value].agg(["mean", "std"]).reindex([k for k in KS if k in set(df["k"])])
    x = np.arange(len(g))
    ax.plot(x, g["mean"], ls=ls, marker=marker, ms=3.5, lw=1.1, color=color, label=label)
    if band:
        ax.fill_between(x, g["mean"] - g["std"], (g["mean"] + g["std"]).clip(upper=1), color=color, alpha=0.12, lw=0)
    ax.set_xticks(x, ["all" if k == -1 else str(k) for k in g.index])


def fig_detection() -> None:
    """Fig. 1: AUPRC with all labels; mining needs no label."""
    v2, v35 = load("phase2_v2.jsonl"), load("phase2_v3.5.jsonl")
    d = pd.concat([v2[v2["method"].isin(["fpgrowth", "lgbm"])], v35.assign(method="tabpfn")])
    d = d[(d["rate"] == 0.02) & (d["k"] == -1) & d["fs"].isin(["presence", "-"])]
    tasks = ["R1", "R2", "R3", "R4", "pooled"]
    methods = [("fpgrowth", "FP-growth (no labels)", GREY), ("lgbm", "LightGBM", PINK), ("tabpfn", "TabPFN-3.5", RED)]
    fig, ax = plt.subplots(figsize=(6.0, 2.1))
    for i, (m, label, color) in enumerate(methods):
        g = d[d["method"] == m].groupby("task")["auprc"].agg(["mean", "std"]).reindex(tasks)
        x = np.arange(len(tasks)) + (i - 1) * 0.26
        ax.bar(x, g["mean"], 0.24, yerr=g["std"], color=color, label=label, error_kw={"lw": 0.6, "capsize": 1.5})
    ax.set_xticks(np.arange(len(tasks)), tasks)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Test AUPRC")
    ax.legend(ncol=3, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    fig.savefig(OUT / "fig1_detection.svg")
    plt.close(fig)


def fig_label_efficiency() -> None:
    """Fig. 2: AUPRC against the number of labelled violations in context (presence features)."""
    v2, v35 = load("phase2_v2.jsonl"), load("phase2_v3.5.jsonl")
    d = pd.concat([v2.assign(method=v2["method"].replace({"tabpfn": "tabpfn-v2"})), v35.assign(method="tabpfn-3.5")])
    d = d[(d["fs"] == "presence") & (d["rate"] == 0.02)]
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.2), sharey=True)
    for ax, task in zip(axes, ["R4", "pooled"]):
        t = d[d["task"] == task]
        curve(ax, t[t["method"] == "lgbm"], "auprc", "LightGBM", PINK, marker="s")
        curve(ax, t[t["method"] == "tabpfn-v2"], "auprc", "TabPFN v2", GREEN, ls="--")
        curve(ax, t[t["method"] == "tabpfn-3.5"], "auprc", "TabPFN-3.5", RED)
        ax.set_title(task if task == "R4" else "Pooled (all purchases and sales)", fontsize=9)
        ax.set_xlabel("Labelled violations in context")
        ax.set_ylim(0, 1.05)
    axes[0].set_ylabel("Test AUPRC")
    axes[1].legend(frameon=False, loc="lower right")
    fig.savefig(OUT / "fig2_label_efficiency.svg")
    plt.close(fig)


def fig_exception() -> None:
    """Fig. 3: violations vs the non-deductible exception, seen or unseen in context (R4, rare rate 2%)."""
    b, c = load("phase2b_rare.jsonl").assign(ctx="seen"), load("phase2c_unseen_rare.jsonl").assign(ctx="unseen")
    d = pd.concat([b, c])
    d = d[(d["task"] == "R4") & (d["rare_rate"] == 0.02) & (d["fs"] == "presence")]
    col = "auroc_viol_vs_eu_non_deductible"
    fig, ax = plt.subplots(figsize=(4.2, 2.4))
    curve(ax, d[(d["method"] == "tabpfn") & (d["ctx"] == "seen")], col, "TabPFN-3.5, exception seen", RED, ls="--", band=False)
    curve(ax, d[(d["method"] == "tabpfn") & (d["ctx"] == "unseen")], col, "TabPFN-3.5, exception unseen", RED)
    curve(ax, d[(d["method"] == "lgbm") & (d["ctx"] == "seen")], col, "LightGBM, exception seen", PINK, ls="--", marker="s", band=False)
    curve(ax, d[(d["method"] == "lgbm") & (d["ctx"] == "unseen")], col, "LightGBM, exception unseen", PINK, marker="s", band=False)
    ax.axhline(0.5, color="0.3", lw=0.6, ls=":")
    ax.text(0.02, 0.51, "chance", fontsize=7, color="0.3", transform=ax.get_yaxis_transform())
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Labelled violations in context")
    ax.set_ylabel("AUROC, violations vs exception")
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    fig.savefig(OUT / "fig3_exception.svg")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    fig_detection()
    fig_label_efficiency()
    fig_exception()
    print("figures/ written")


if __name__ == "__main__":
    main()
