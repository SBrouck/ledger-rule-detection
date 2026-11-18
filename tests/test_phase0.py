import pandas as pd

from ledger.generate import Config, generate
from ledger.validate import check, failures


def _run(tmp_path, rate=0.05, seed=1):
    cfg = Config(n_companies=6, seed=seed, violation_rate={r: rate for r in ("R1", "R2", "R3", "R4")})
    generate(cfg, tmp_path)
    return check(tmp_path)


def test_clean_run_passes(tmp_path):
    report = _run(tmp_path)
    assert failures(report) == [], failures(report)
    assert all(v["eligible"] > 0 for v in report["rates"].values())


def test_zero_rate_has_no_violations(tmp_path):
    report = _run(tmp_path, rate=0.0)
    assert all(v["violations"] == 0 for v in report["rates"].values())


def test_oracle_catches_a_wrong_label(tmp_path):
    _run(tmp_path)
    meta = pd.read_csv(tmp_path / "entries.csv", dtype={"company": str})
    i = meta.index[meta["rule"] == "R4"][0]
    meta.loc[i, "violated"] = not meta.loc[i, "violated"]
    meta.to_csv(tmp_path / "entries.csv", index=False)
    assert check(tmp_path)["oracle"]["violation_mismatch"] == 1


def test_oracle_catches_an_unbalanced_entry(tmp_path):
    _run(tmp_path)
    fec_file = next((tmp_path / "fec").glob("*.txt"))
    fec = pd.read_csv(fec_file, sep="\t", dtype=str, keep_default_na=False)
    fec.loc[0, "Debit"] = "1,00" if fec.loc[0, "Debit"] != "1,00" else "2,00"
    fec.loc[0, "Credit"] = "0,00"
    fec.to_csv(fec_file, sep="\t", index=False)
    assert check(tmp_path)["format"]["unbalanced_entries"] >= 1


def test_rare_valid_patterns_stay_valid(tmp_path):
    cfg = Config(n_companies=8, seed=2, violation_rate={r: 0.05 for r in ("R1", "R2", "R3", "R4")},
                 rare_valid_rate=0.05)
    meta = generate(cfg, tmp_path)
    report = check(tmp_path)
    assert failures(report) == [], failures(report)
    rare = meta[meta["rare"].fillna("") != ""]
    assert set(rare["rare"]) == {"multi_expense", "discount_609", "eu_non_deductible"}
    assert not rare["violated"].any()


def test_zero_rare_rate_reproduces_old_data(tmp_path):
    a = generate(Config(n_companies=3, seed=4), tmp_path / "a")
    b = generate(Config(n_companies=3, seed=4, rare_valid_rate=0.0), tmp_path / "b")
    assert (tmp_path / "a" / "entries.csv").read_text() == (tmp_path / "b" / "entries.csv").read_text()
