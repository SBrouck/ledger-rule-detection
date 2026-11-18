"""Phase 1: frequent-itemset mining (FP-growth) on ledger entries.

Each entry becomes a transaction of items:
  acc:<account>     an account present in the entry
  no:<account>      a VAT account absent from the entry (lets rules express "never")
  journal:<code>    the journal
  supplier:<type>   the supplier attribute
  durable           the entry label names a durable item
  amt:<bin>         total debit, in fixed bins that do not align with the R3 threshold

Rules are mined on the training companies only, without labels. Two outputs:
  1. recovery: is each planted rule among the mined rules, with what support and confidence?
  2. detection: score each test entry by the confidence of the strongest rule it breaks.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import association_rules, fpgrowth
from sklearn.metrics import average_precision_score, roc_auc_score

from ledger.generate import DURABLE_ITEMS
from ledger.validate import load_fec

VAT_ACCOUNTS = ["4452", "44562", "44566", "44571"]
AMOUNT_BINS = [0, 100_00, 300_00, 1_000_00, 3_000_00, np.inf]  # cents, on total debit (incl. VAT)
AMOUNT_LABELS = ["<100", "100-300", "300-1k", "1k-3k", ">3k"]
DURABLE_LABELS = {name for name, _ in DURABLE_ITEMS}

# Planted rules written as association rules (antecedent -> consequent). R3 has no exact
# itemset form, so it is checked separately (see recovery()).
TARGETS = {
    "R1 purchase": ({"journal:HA", "supplier:fr_normal", "acc:401"}, {"acc:44566"}),
    "R1 sale": ({"journal:VE"}, {"acc:44571"}),
    "R2": ({"journal:HA", "supplier:fr_franchise"}, {"no:44566"}),
    "R4 both": ({"journal:HA", "supplier:eu", "acc:4452"}, {"acc:44566"}),
    "R4 neither": ({"journal:HA", "supplier:eu", "no:4452"}, {"no:44566"}),
}


def transactions(run: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One-hot item matrix (one row per entry) and the matching entries table."""
    fec = load_fec(run)
    meta = pd.read_csv(run / "entries.csv", dtype={"company": str}, keep_default_na=False)
    meta["rule"] = meta["rule"].replace("", None)
    meta["violated"] = meta["violated"].astype(str) == "True"
    g = fec.groupby(["company", "EcritureNum"], sort=False)
    accs = g["CompteNum"].agg(set)
    first = g[["JournalCode", "EcritureLib"]].first()
    total = g["debit"].sum()
    df = pd.DataFrame({"accs": accs, "journal": first["JournalCode"], "label": first["EcritureLib"],
                       "total": total}).reset_index().rename(columns={"EcritureNum": "piece"})
    df = df.merge(meta[["company", "piece", "supplier_type", "rule", "violated"]], on=["company", "piece"])
    bins = pd.cut(df["total"], AMOUNT_BINS, labels=AMOUNT_LABELS, right=False)

    def items(row, b):
        out = {f"acc:{a}" for a in row.accs} | {f"journal:{row.journal}", f"amt:{b}"}
        out |= {f"no:{a}" for a in VAT_ACCOUNTS if a not in row.accs}
        if row.supplier_type:
            out.add(f"supplier:{row.supplier_type}")
        if row.label in DURABLE_LABELS:
            out.add("durable")
        return out

    item_sets = [items(r, b) for r, b in zip(df.itertuples(), bins)]
    vocab = sorted(set().union(*item_sets))
    onehot = pd.DataFrame([[i in s for i in vocab] for s in item_sets], columns=vocab)
    return onehot, df


def mine(onehot: pd.DataFrame, min_support: float, min_conf: float = 0.5) -> pd.DataFrame:
    freq = fpgrowth(onehot, min_support=min_support, use_colnames=True, max_len=4)
    rules = association_rules(freq, num_itemsets=len(onehot), metric="confidence", min_threshold=min_conf)
    return rules[["antecedents", "consequents", "support", "confidence", "lift"]]


def recovery(rules: pd.DataFrame, conf: float = 0.9) -> dict:
    out = {}
    for name, (ante, cons) in TARGETS.items():
        hit = rules[(rules["antecedents"] == frozenset(ante)) & (rules["consequents"] == frozenset(cons))]
        row = hit.iloc[0] if len(hit) else None
        out[name] = {"found": bool(row is not None and row["confidence"] >= conf),
                     "support": None if row is None else round(float(row["support"]), 4),
                     "confidence": None if row is None else round(float(row["confidence"]), 4)}
    # R3: the best rule that sends a durable purchase to the fixed-asset VAT account, using
    # causes only (durable flag, amount bin, journal, supplier). An antecedent holding an
    # account such as 2183 or 404 already encodes the outcome, so it does not count.
    def causes_only(a):
        return "durable" in a and all(i == "durable" or i.split(":")[0] in ("amt", "journal", "supplier")
                                      for i in a)

    r3 = rules[rules["antecedents"].apply(causes_only)
               & rules["consequents"].apply(lambda c: c == frozenset({"acc:44562"}))]
    best = r3.sort_values("confidence", ascending=False).head(1)
    out["R3"] = {"found": bool(len(best) and best["confidence"].iat[0] >= conf),
                 "best_rule": None if best.empty else
                 f"{sorted(best['antecedents'].iat[0])} -> acc:44562",
                 "support": None if best.empty else round(float(best["support"].iat[0]), 4),
                 "confidence": None if best.empty else round(float(best["confidence"].iat[0]), 4)}
    return out


def score(onehot: pd.DataFrame, rules: pd.DataFrame, conf: float = 0.9, min_lift: float = 1.1) -> np.ndarray:
    """Confidence of the strongest high-confidence rule each entry breaks (0 if none)."""
    strong = rules[(rules["confidence"] >= conf) & (rules["lift"] >= min_lift)]
    x = onehot.to_numpy()
    col = {c: i for i, c in enumerate(onehot.columns)}
    s = np.zeros(len(onehot))
    for ante, cons, c in zip(strong["antecedents"], strong["consequents"], strong["confidence"]):
        if not all(i in col for i in ante | cons):
            continue
        a = x[:, [col[i] for i in ante]].all(axis=1)
        k = x[:, [col[i] for i in cons]].all(axis=1)
        s = np.where(a & ~k, np.maximum(s, c), s)
    return s


def evaluate(run: Path, min_support: float, seed: int = 0) -> dict:
    onehot, df = transactions(run)
    companies = np.array(sorted(df["company"].unique()))
    rng = np.random.default_rng(seed)
    train_c = set(rng.choice(companies, size=len(companies) // 2, replace=False))
    is_train = df["company"].isin(train_c).to_numpy()
    rules = mine(onehot[is_train].reset_index(drop=True), min_support)
    test_x, test_df = onehot[~is_train].reset_index(drop=True), df[~is_train].reset_index(drop=True)
    s = score(test_x, rules)
    det = {}
    for rule in ["R1", "R2", "R3", "R4"]:
        m = (test_df["rule"] == rule).to_numpy()
        y = test_df.loc[m, "violated"].to_numpy()
        if y.sum() == 0 or y.all():
            det[rule] = {"n": int(m.sum()), "pos": int(y.sum()), "auprc": None, "auroc": None}
            continue
        det[rule] = {"n": int(m.sum()), "pos": int(y.sum()), "prevalence": round(float(y.mean()), 4),
                     "auprc": round(float(average_precision_score(y, s[m])), 3),
                     "auroc": round(float(roc_auc_score(y, s[m])), 3)}
    return {"run": run.name, "min_support": min_support, "n_rules": len(rules),
            "n_strong": int(((rules["confidence"] >= 0.9) & (rules["lift"] >= 1.1)).sum()),
            "recovery": recovery(rules), "detection": det}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("runs", type=Path, nargs="+")
    p.add_argument("--min-support", type=float, nargs="+", default=[0.001, 0.005, 0.01, 0.05])
    p.add_argument("--out", type=Path, default=Path("results/phase1.jsonl"))
    a = p.parse_args()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w") as f:
        for run in a.runs:
            for ms in a.min_support:
                r = evaluate(run, ms)
                f.write(json.dumps(r) + "\n")
                rec = " ".join(f"{k}={'Y' if v['found'] else 'n'}" for k, v in r["recovery"].items())
                det = " ".join(f"{k}={v['auprc']}" for k, v in r["detection"].items())
                print(f"{run.name} ms={ms} rules={r['n_rules']} | {rec} | AUPRC {det}", flush=True)


if __name__ == "__main__":
    main()
