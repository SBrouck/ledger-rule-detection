"""Phase 0 checks: FEC format, balanced entries, oracle labels, violation rates, R4 marginals.

The oracle re-derives each rule's eligibility and violation from the FEC lines and the
supplier attribute only. It shares no code path with the generator, so a disagreement
means the generator does not plant what it claims.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd

from ledger.generate import DURABLE_ITEMS, FEC_COLUMNS, R3_THRESHOLD

DURABLE_LABELS = {name for name, _ in DURABLE_ITEMS}


def _cents(s: pd.Series) -> pd.Series:
    return s.str.replace(",", ".", regex=False).astype(float).mul(100).round().astype(int)


def load_fec(run: Path) -> pd.DataFrame:
    frames = []
    for f in sorted((run / "fec").glob("*FEC*.txt")):
        df = pd.read_csv(f, sep="\t", dtype=str, keep_default_na=False)
        if list(df.columns) != FEC_COLUMNS:
            raise ValueError(f"{f.name}: columns differ from the 18 FEC columns")
        df["company"] = f.name[:9]
        frames.append(df)
    fec = pd.concat(frames, ignore_index=True)
    fec["debit"], fec["credit"] = _cents(fec["Debit"]), _cents(fec["Credit"])
    return fec


def oracle(fec: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """One row per entry: the rule it falls under (or None) and whether it breaks it."""
    sup = meta.set_index(["company", "piece"])["supplier_type"]
    rows = []
    for (company, piece), g in fec.groupby(["company", "EcritureNum"], sort=False):
        accs = set(g["CompteNum"])
        journal, label = g["JournalCode"].iat[0], g["EcritureLib"].iat[0]
        stype = sup.get((company, piece))
        rule, violated = None, False
        if journal == "VE":
            rule, violated = "R1", "44566" in accs
        elif journal == "HA" and label in DURABLE_LABELS:
            base = int(g.loc[g["CompteNum"].isin(["2183", "2184", "6063"]), "debit"].sum())
            if base > R3_THRESHOLD:
                rule, violated = "R3", "6063" in accs
        elif journal == "HA" and stype == "fr_normal":
            rule, violated = "R1", "44571" in accs
        elif journal == "HA" and stype == "fr_franchise":
            rule, violated = "R2", "44566" in accs
        elif journal == "HA" and stype == "eu":
            # Exception: reverse charge on a non-deductible expense (6257) books 4452 alone.
            rule, violated = "R4", ("4452" in accs) != ("44566" in accs) and "6257" not in accs
        rows.append({"company": company, "piece": piece, "o_rule": rule, "o_violated": violated,
                     "has_4452": "4452" in accs, "has_44566": "44566" in accs})
    return pd.DataFrame(rows)


def check(run: Path) -> dict:
    cfg = json.loads((run / "config.json").read_text())
    meta = pd.read_csv(run / "entries.csv", dtype={"company": str}, keep_default_na=False)
    meta["rule"] = meta["rule"].replace("", None)
    meta["violated"] = meta["violated"].astype(str) == "True"
    fec = load_fec(run)
    report: dict = {"entries": len(meta), "lines": len(fec), "companies": fec["company"].nunique()}

    one_side = ((fec["debit"] > 0) ^ (fec["credit"] > 0)).all()
    sums = fec.groupby(["company", "EcritureNum"])[["debit", "credit"]].sum()
    dates = pd.to_datetime(fec["EcritureDate"], format="%Y%m%d")
    valid = pd.to_datetime(fec["ValidDate"], format="%Y%m%d")
    report["format"] = {
        "one_side_per_line": bool(one_side),
        "unbalanced_entries": int((sums["debit"] != sums["credit"]).sum()),
        "dates_in_year": bool((dates.dt.year == cfg["year"]).all()),
        "valid_after_entry": bool((valid >= dates).all()),
        "empty_mandatory": int((fec[["JournalCode", "EcritureNum", "CompteNum", "PieceRef"]] == "").sum().sum()),
    }

    o = oracle(fec, meta).merge(meta, on=["company", "piece"], how="outer", indicator=True)
    report["oracle"] = {
        "unmatched_entries": int((o["_merge"] != "both").sum()),
        "rule_mismatch": int((o["o_rule"].fillna("-") != o["rule"].fillna("-")).sum()),
        "violation_mismatch": int((o["o_violated"] != o["violated"]).sum()),
    }

    rates = {}
    for rule, target in cfg["violation_rate"].items():
        sub = o[o["rule"] == rule]
        n, k = len(sub), int(sub["violated"].sum())
        sd = math.sqrt(target * (1 - target) / n) if n else float("nan")
        rates[rule] = {"eligible": n, "violations": k, "rate": k / n if n else None, "target": target,
                       "z": (k / n - target) / sd if n and sd else None}
    report["rates"] = rates

    r4 = o[o["rule"] == "R4"]
    marg = {}
    for acc in ("4452", "44566"):
        col = f"has_{acc}"
        p_valid, p_inv = r4.loc[~r4["violated"], col].mean(), r4.loc[r4["violated"], col].mean()
        n_inv = int(r4["violated"].sum())
        se = math.sqrt(0.25 / max(n_inv, 1) + 0.25 / max(len(r4) - n_inv, 1))
        marg[acc] = {"p_valid": round(p_valid, 3), "p_invalid": round(p_inv, 3),
                     "z": round((p_inv - p_valid) / se, 2)}
    report["r4_marginals"] = marg
    return report


def failures(report: dict) -> list[str]:
    f = report["format"]
    out = []
    if not f["one_side_per_line"]:
        out.append("a line has both or neither debit and credit")
    if f["unbalanced_entries"]:
        out.append(f"{f['unbalanced_entries']} unbalanced entries")
    if not (f["dates_in_year"] and f["valid_after_entry"]):
        out.append("dates out of range")
    if f["empty_mandatory"]:
        out.append("empty mandatory fields")
    out += [f"oracle {k}: {v}" for k, v in report["oracle"].items() if v]
    out += [f"{r} rate z={v['z']:.1f}" for r, v in report["rates"].items() if v["z"] is not None and abs(v["z"]) > 3]
    out += [f"R4 marginal {a} z={v['z']}" for a, v in report["r4_marginals"].items() if abs(v["z"]) > 3]
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("run", type=Path)
    report = check(p.parse_args().run)
    print(json.dumps(report, indent=2, default=str))
    bad = failures(report)
    print("FAIL: " + "; ".join(bad) if bad else "OK: all phase 0 checks pass")
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
