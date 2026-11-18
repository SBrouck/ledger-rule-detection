from ledger.generate import Config, generate
from ledger.mining import evaluate, recovery


def test_mining_recovers_cooccurrence_rules(tmp_path):
    generate(Config(n_companies=10, seed=3, violation_rate={r: 0.05 for r in ("R1", "R2", "R3", "R4")}), tmp_path)
    r = evaluate(tmp_path, min_support=0.005)
    for name in ("R1 purchase", "R1 sale", "R2", "R4 both", "R4 neither"):
        assert r["recovery"][name]["found"], name


def test_r3_ignores_rules_that_start_from_the_outcome():
    import pandas as pd
    rules = pd.DataFrame({"antecedents": [frozenset({"acc:2183", "durable"})],
                          "consequents": [frozenset({"acc:44562"})],
                          "support": [0.01], "confidence": [1.0], "lift": [50.0]})
    assert recovery(rules)["R3"]["found"] is False
