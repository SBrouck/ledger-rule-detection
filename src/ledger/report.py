"""Markdown tables behind every number in the README.

    python -m ledger.report
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

RES = Path("results")


def load(name: str) -> pd.DataFrame:
    df = pd.DataFrame([json.loads(line) for line in (RES / name).open()])
    df["rate"] = df["run"].str.extract(r"r([\d.]+)_s")[0].astype(float)
    if df["run"].str.startswith("rare").any():
        df["rare_rate"] = df["run"].str.extract(r"rare([\d.]+)_")[0].astype(float)
    if "k_pos" in df:
        df["k"] = df["k_pos"].fillna(-1).astype(int)
    return df


def ms(s: pd.Series) -> str:
    s = s.dropna()
    if s.empty:
        return "n/a"
    return f"{s.mean():.3f}" if s.std() < 0.0005 or len(s) < 2 else f"{s.mean():.3f} ± {s.std():.3f}"


def k_label(k: int) -> str:
    return "all" if k == -1 else str(k)


def table(df: pd.DataFrame, index: list[str], columns: str, value: str) -> str:
    t = df.pivot_table(index=index, columns=columns, values=value, aggfunc=ms, sort=False)
    if columns == "k":
        t = t[[c for c in [1, 2, 5, 10, 20, -1] if c in t.columns]]
        t.columns = [k_label(c) for c in t.columns]
    return t.to_markdown(disable_numparse=True)


def phase1() -> str:
    rows = []
    for line in (RES / "phase1.jsonl").open():
        r = json.loads(line)
        rec = {k: int(v["found"]) for k, v in r["recovery"].items()}
        rows.append({"rate": r["run"].split("_")[0][1:], "min_support": r["min_support"], **rec,
                     **{f"AUPRC {k}": v["auprc"] for k, v in r["detection"].items()}})
    df = pd.DataFrame(rows)
    rec = df.groupby("min_support")[["R1 purchase", "R1 sale", "R2", "R4 both", "R4 neither", "R3"]].sum()
    rec = rec.astype(str) + f"/{df.groupby('min_support').size().iloc[0]}"
    det = df[df["min_support"] == 0.001].groupby("rate")[[c for c in df if c.startswith("AUPRC")]].agg(ms)
    return "Rule recovery (runs where the exact rule is mined, confidence >= 0.9)\n\n" + rec.to_markdown(disable_numparse=True) + \
        "\n\nDetection, min support 0.001\n\n" + det.to_markdown(disable_numparse=True)


def main() -> None:
    print("## Phase 1\n\n" + phase1())
    v2, v35 = load("phase2_v2.jsonl"), load("phase2_v3.5.jsonl")
    full = pd.concat([v2[v2["method"] != "tabpfn"], v2[v2["method"] == "tabpfn"].assign(method="tabpfn-v2"),
                      v35.assign(method="tabpfn-3.5")])
    full = full[full["k"] == -1]
    full["model"] = full["method"] + " " + full["fs"].replace("-", "")
    print("\n## Phase 2, all labels, AUPRC\n\n" + table(full, ["task", "rate"], "model", "auprc"))
    le = pd.concat([v2.assign(method=v2["method"].replace({"tabpfn": "tabpfn-v2"})),
                    v35.assign(method="tabpfn-3.5")])
    le = le[(le["fs"] == "presence") & (le["method"] != "fpgrowth") & le["task"].isin(["R4", "pooled"])]
    print("\n## Phase 2, label efficiency, presence features, AUPRC\n\n" +
          table(le, ["task", "rate", "method"], "k", "auprc"))
    b, c = load("phase2b_rare.jsonl").assign(ctx="seen"), load("phase2c_unseen_rare.jsonl").assign(ctx="unseen")
    rare = pd.concat([b, c])
    rare = rare[rare["fs"].isin(["presence", "-"])]
    for col, task in [("auroc_viol_vs_eu_non_deductible", "R4"), ("auroc_viol_vs_eu_non_deductible", "pooled"),
                      ("auroc_viol_vs_discount_609", "pooled"), ("auroc_viol_vs_multi_expense", "pooled")]:
        sub = rare[rare["task"] == task]
        print(f"\n## Phase 2b/2c, {task}, {col}\n\n" + table(sub, ["rare_rate", "method", "ctx"], "k", col))


if __name__ == "__main__":
    main()
