"""Phase 2: supervised violation detection, TabPFN vs LightGBM vs FP-growth.

For each task, rows are ledger entries and the label is "breaks a planted rule".
  R1..R4   entries eligible for that rule
  pooled   every purchase and sale entry; label = breaks any rule

Two feature sets:
  presence  account presence (one column per account), journal, supplier type, durable
            flag, log of the total debit. The total is the same for valid and invalid R4
            entries, so R4 is a pure co-occurrence test here.
  amounts   presence plus each account's signed share of the total (debit - credit) / total.
            Phase 0 showed that the 401 / expense ratio alone gives R4 away.

Companies are split in two halves (same split as phase 1). The supervised context is drawn
from the training half; with --k-pos, at most k labelled violations are kept in it.
FP-growth is mined on the same training half without labels and scored on the same test rows.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from ledger.generate import ACCOUNTS, DURABLE_ITEMS, SUPPLIER_TYPES
from ledger.mining import mine, score, transactions
from ledger.validate import load_fec

DURABLE_LABELS = {name for name, _ in DURABLE_ITEMS}
ACCS = sorted(ACCOUNTS)


def features(run: Path) -> pd.DataFrame:
    fec = load_fec(run)
    meta = pd.read_csv(run / "entries.csv", dtype={"company": str}, keep_default_na=False)
    meta["rule"] = meta["rule"].replace("", None)
    meta["violated"] = meta["violated"].astype(str) == "True"
    fec["net"] = fec["debit"] - fec["credit"]
    key = ["company", "EcritureNum"]
    net = fec.pivot_table(index=key, columns="CompteNum", values="net", aggfunc="sum").reindex(columns=ACCS)
    first = fec.groupby(key)[["JournalCode", "EcritureLib"]].first()
    total = fec.groupby(key)["debit"].sum()
    x = pd.DataFrame(index=net.index)
    for a in ACCS:
        x[f"acc_{a}"] = net[a].notna().astype(int)
    for j in ("HA", "VE", "BQ", "OD"):
        x[f"journal_{j}"] = (first["JournalCode"] == j).astype(int)
    x["durable"] = first["EcritureLib"].isin(DURABLE_LABELS).astype(int)
    x["log_total"] = np.log1p(total / 100)
    for a in ACCS:
        x[f"share_{a}"] = (net[a].fillna(0) / total).round(4)
    x = x.reset_index().rename(columns={"EcritureNum": "piece"})
    if "rare" not in meta:
        meta["rare"] = ""
    x = x.merge(meta[["company", "piece", "supplier_type", "rule", "violated", "rare"]], on=["company", "piece"])
    for s in SUPPLIER_TYPES:
        x[f"supplier_{s}"] = (x["supplier_type"] == s).astype(int)
    return x


def columns(x: pd.DataFrame, fs: str) -> list[str]:
    base = [c for c in x.columns if c.startswith(("acc_", "journal_", "supplier_")) and c != "supplier_type"] + ["durable", "log_total"]
    return base + ([c for c in x.columns if c.startswith("share_")] if fs == "amounts" else [])


def task_mask(x: pd.DataFrame, task: str) -> np.ndarray:
    if task == "pooled":
        return (x["journal_HA"].eq(1) | x["journal_VE"].eq(1)).to_numpy()
    return (x["rule"] == task).to_numpy()


def split(x: pd.DataFrame, seed: int = 0) -> np.ndarray:
    companies = np.array(sorted(x["company"].unique()))
    rng = np.random.default_rng(seed)
    return x["company"].isin(set(rng.choice(companies, size=len(companies) // 2, replace=False))).to_numpy()


def context(y: np.ndarray, rng, max_ctx: int, k_pos: int | None) -> np.ndarray:
    pos, neg = np.flatnonzero(y), np.flatnonzero(~y)
    if k_pos is not None:
        pos = rng.choice(pos, size=min(k_pos, len(pos)), replace=False)
    neg = rng.choice(neg, size=min(len(neg), max_ctx - len(pos)), replace=False)
    return np.sort(np.concatenate([pos, neg]))


def fit_predict(method: str, xtr, ytr, xte, seed: int, model: str) -> np.ndarray:
    if ytr.sum() == 0:
        return np.zeros(len(xte))
    if method == "lgbm":
        import lightgbm as lgb
        clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, min_child_samples=1,
                                 random_state=seed, verbose=-1)
        clf.fit(xtr, ytr)
        return clf.predict_proba(xte)[:, 1]
    from tabpfn import TabPFNClassifier
    from tabpfn.constants import ModelVersion
    if model == "v2":
        clf = TabPFNClassifier.create_default_for_version(ModelVersion.V2, device="mps", random_state=seed)
    else:
        clf = TabPFNClassifier(device="mps", random_state=seed)
    clf.fit(xtr, ytr)
    # Predict in chunks: the pooled task (about 12k test rows) runs out of MPS memory in one call.
    return np.concatenate([clf.predict_proba(xte[i:i + 2000])[:, 1] for i in range(0, len(xte), 2000)])


def rare_metrics(x_te: pd.DataFrame, s: np.ndarray) -> dict:
    """Violations vs rare valid entries only: does the score separate errors from rarity?"""
    rare = (x_te["rare"].fillna("") != "").to_numpy()
    viol = x_te["violated"].to_numpy()
    if not rare.any() or not viol.any():
        return {}
    m = rare | viol
    out = {"n_rare_test": int(rare.sum()),
           "auroc_viol_vs_rare": round(float(roc_auc_score(viol[m], s[m])), 4),
           "rare_above_min_viol": round(float((s[rare] >= s[viol].min()).mean()), 4)}
    for kind in sorted(set(x_te.loc[rare, "rare"])):
        k = (x_te["rare"] == kind).to_numpy()
        mk = k | viol
        out[f"auroc_viol_vs_{kind}"] = round(float(roc_auc_score(viol[mk], s[mk])), 4)
    return out


def metrics(y, s) -> dict:
    if y.sum() == 0 or y.all():
        return {"auprc": None, "auroc": None}
    return {"auprc": round(float(average_precision_score(y, s)), 4), "auroc": round(float(roc_auc_score(y, s)), 4)}


def run(path: Path, tasks, feature_sets, methods, k_pos_list, model, max_ctx, out, unseen_rare=False) -> None:
    x = features(path)
    is_train = split(x)
    if unseen_rare:
        # Rare valid patterns never reach the context or the mined rules: they appear only at test time.
        is_rare = (x["rare"].fillna("") != "").to_numpy()
    else:
        is_rare = np.zeros(len(x), dtype=bool)
    fp = None
    if "fpgrowth" in methods:  # unsupervised, independent of k_pos and the feature set
        onehot, df = transactions(path)
        df = df[["company", "piece"]].reset_index(names="t_idx")
        df = df.merge(x[["company", "piece"]].reset_index(names="x_idx"), on=["company", "piece"])
        assert len(df) == len(x)
        onehot = onehot.iloc[df.sort_values("x_idx")["t_idx"].to_numpy()].reset_index(drop=True)
        rules = mine(onehot[is_train & ~is_rare].reset_index(drop=True), 0.001)
        fp = np.zeros(len(x))
        fp[~is_train] = score(onehot[~is_train].reset_index(drop=True), rules)
    for task in tasks:
        m = task_mask(x, task)
        tr, te = np.flatnonzero(m & is_train & ~is_rare), np.flatnonzero(m & ~is_train)
        ytr, yte = x["violated"].to_numpy()[tr], x["violated"].to_numpy()[te]
        base = {"run": path.name, "task": task, "n_test": len(te), "pos_test": int(yte.sum()), "model": model,
                "unseen_rare": unseen_rare}
        if fp is not None:
            out({**base, "method": "fpgrowth", "fs": "-", "k_pos": None, **metrics(yte, fp[te]),
                 **rare_metrics(x.iloc[te], fp[te])})
        for k in k_pos_list:
            ctx = context(ytr, np.random.default_rng(1000 + (k or 0)), max_ctx, k)
            for fs in feature_sets:
                cols = columns(x, fs)
                xtr, xte = x.iloc[tr[ctx]][cols].to_numpy(float), x.iloc[te][cols].to_numpy(float)
                for method in [mm for mm in methods if mm != "fpgrowth"]:
                    t = time.time()
                    s = fit_predict(method, xtr, ytr[ctx], xte, 0, model)
                    out({**base, "method": method, "fs": fs, "k_pos": k, "n_ctx": len(ctx),
                         "pos_ctx": int(ytr[ctx].sum()), "seconds": round(time.time() - t, 1), **metrics(yte, s),
                         **rare_metrics(x.iloc[te], s)})


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("runs", type=Path, nargs="+")
    p.add_argument("--tasks", nargs="+", default=["R1", "R2", "R3", "R4", "pooled"])
    p.add_argument("--fs", nargs="+", default=["presence", "amounts"])
    p.add_argument("--methods", nargs="+", default=["fpgrowth", "lgbm", "tabpfn"])
    p.add_argument("--k-pos", nargs="+", default=["all"], help="max labelled violations in context, or 'all'")
    p.add_argument("--model", default="v2", choices=["v2", "v3.5"])
    p.add_argument("--max-ctx", type=int, default=3000)
    p.add_argument("--out", type=Path, default=Path("results/phase2.jsonl"))
    p.add_argument("--unseen-rare", action="store_true", help="keep rare valid patterns out of context and mining")
    a = p.parse_args()
    os.environ.setdefault("TABPFN_NO_BROWSER", "1")
    from dotenv import load_dotenv
    load_dotenv(Path(".env"))
    k_list = [None if k == "all" else int(k) for k in a.k_pos]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("a") as f:
        def out(r):
            f.write(json.dumps(r) + "\n")
            f.flush()
            print(f"{r['run']} {r['task']:6} {r['method']:8} {r['fs']:8} k={r['k_pos']} "
                  f"AUPRC={r['auprc']} AUROC={r['auroc']} pos_test={r['pos_test']}", flush=True)
        for path in a.runs:
            run(path, a.tasks, a.fs, a.methods, k_list, a.model, a.max_ctx, out, a.unseen_rare)


if __name__ == "__main__":
    main()
