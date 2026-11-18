"""Synthetic French general ledgers (FEC) with planted rules and labelled violations.

Amounts are integer cents. One entry (pièce) is a balanced list of lines. Each entry is
eligible for at most one planted rule, and carries at most one violation.

Planted rules:
  R1  purchases book deductible VAT on 44566, sales book collected VAT on 44571
  R2  a supplier under franchise en base never comes with a 44566 line
  R3  a durable purchase over 500 EUR excl. VAT goes to 2183/2184 + 44562 / 404
  R4  an intra-EU purchase carries both 4452 and 44566, or neither; each account has
      the same frequency in valid and invalid entries, only the pair separates them
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

FEC_COLUMNS = [
    "JournalCode", "JournalLib", "EcritureNum", "EcritureDate", "CompteNum", "CompteLib",
    "CompAuxNum", "CompAuxLib", "PieceRef", "PieceDate", "EcritureLib", "Debit", "Credit",
    "EcritureLet", "DateLet", "ValidDate", "Montantdevise", "Idevise",
]
JOURNALS = {"HA": "Achats", "VE": "Ventes", "BQ": "Banque", "OD": "Operations diverses"}
ACCOUNTS = {
    "2183": "Materiel de bureau et informatique", "2184": "Mobilier",
    "401": "Fournisseurs", "404": "Fournisseurs d'immobilisations", "411": "Clients",
    "421": "Personnel remunerations dues", "431": "Securite sociale", "437": "Autres organismes sociaux",
    "4452": "TVA due intracommunautaire", "44562": "TVA deductible sur immobilisations",
    "44566": "TVA deductible sur ABS", "44571": "TVA collectee", "512": "Banque",
    "6063": "Fournitures d'entretien et petit equipement", "6064": "Fournitures administratives",
    "607": "Achats de marchandises", "613": "Locations", "622": "Honoraires",
    "625": "Deplacements missions receptions", "626": "Frais postaux et telecommunications",
    "641": "Remunerations du personnel", "645": "Charges de securite sociale",
    "706": "Prestations de services",
    "609": "Rabais remises ristournes obtenus", "6257": "Receptions",
}
EXPENSE_ACCOUNTS = ["607", "6064", "613", "622", "626"]
DURABLE_ITEMS = [("Ordinateur portable", "2183"), ("Ecran", "2183"), ("Bureau", "2184"), ("Fauteuil", "2184")]
SUPPLIER_TYPES = ["fr_normal", "fr_franchise", "eu", "non_eu"]
R3_THRESHOLD = 50_000  # 500 EUR excl. VAT, in cents


@dataclass
class Config:
    n_companies: int = 40
    year: int = 2025
    seed: int = 0
    violation_rate: dict[str, float] = field(
        default_factory=lambda: {"R1": 0.02, "R2": 0.02, "R3": 0.02, "R4": 0.02}
    )
    supplier_mix: tuple[float, ...] = (0.65, 0.15, 0.15, 0.05)  # order of SUPPLIER_TYPES
    purchases_per_month: float = 30.0
    sales_per_month: float = 20.0
    durable_share: float = 0.10  # share of fr_normal purchases that are durable items
    # Share of valid fr_normal and EU purchases turned into a rare but valid pattern. At 0 no
    # extra random draw is made, so datasets from phases 0 to 2 are reproduced exactly.
    rare_valid_rate: float = 0.0


@dataclass
class Entry:
    company: str
    journal: str
    day: date
    label: str
    piece: str
    lines: list  # (account, aux_code, aux_name, debit_cents, credit_cents)
    meta: dict


def _vat(ht: int, rate: float = 0.20) -> int:
    return int(round(ht * rate))


def _amount(rng: np.random.Generator, median_eur: float, sigma: float = 0.9) -> int:
    return max(500, int(round(rng.lognormal(np.log(median_eur * 100), sigma))))


class CompanyGenerator:
    def __init__(self, idx: int, cfg: Config, rng: np.random.Generator):
        self.cfg, self.rng = cfg, rng
        self.siren = f"9{idx:08d}"
        self.employees = int(rng.choice([0, 0, 1, 2, 3, 5, 8]))
        n_sup = int(rng.integers(12, 25))
        self.suppliers = []
        for j in range(n_sup):
            kind = SUPPLIER_TYPES[rng.choice(4, p=cfg.supplier_mix)]
            self.suppliers.append({
                "code": f"F{idx:03d}{j:03d}", "name": f"Fournisseur {idx}-{j}", "type": kind,
                "expense": str(rng.choice(EXPENSE_ACCOUNTS)),
            })
        self.customers = [{"code": f"C{idx:03d}{j:03d}", "name": f"Client {idx}-{j}"}
                          for j in range(int(rng.integers(8, 30)))]
        self.counters = {j: 0 for j in JOURNALS}
        self.entries: list[Entry] = []

    def _violate(self, rule: str) -> bool:
        return bool(self.rng.random() < self.cfg.violation_rate.get(rule, 0.0))

    def _rare(self, meta) -> bool:
        if not self.cfg.rare_valid_rate or meta["violated"]:
            return False
        return bool(self.rng.random() < self.cfg.rare_valid_rate)

    def _add(self, journal, day, label, lines, meta):
        self.counters[journal] += 1
        piece = f"{journal}{day:%y}{self.counters[journal]:05d}"
        assert sum(l[3] for l in lines) == sum(l[4] for l in lines), (label, lines)
        self.entries.append(Entry(self.siren, journal, day, label, piece, lines, meta))

    def purchase(self, day: date):
        rng, s = self.rng, self.suppliers[int(self.rng.integers(len(self.suppliers)))]
        aux = (s["code"], s["name"])
        meta = {"template": "purchase", "supplier_type": s["type"], "category": "expense",
                "rule": None, "violated": False, "rare": ""}
        if s["type"] == "fr_normal" and rng.random() < self.cfg.durable_share:
            item, acc = DURABLE_ITEMS[int(rng.integers(len(DURABLE_ITEMS)))]
            ht = _amount(rng, 500, 0.6)
            meta.update(category="durable", amount_ht=ht)
            if ht > R3_THRESHOLD:
                meta["rule"] = "R3"
                if self._violate("R3"):  # expensed instead of capitalised
                    meta["violated"] = True
                    lines = [("6063", "", "", ht, 0), ("44566", "", "", _vat(ht), 0),
                             ("401", *aux, 0, ht + _vat(ht))]
                else:
                    lines = [(acc, "", "", ht, 0), ("44562", "", "", _vat(ht), 0),
                             ("404", *aux, 0, ht + _vat(ht))]
            else:
                lines = [("6063", "", "", ht, 0), ("44566", "", "", _vat(ht), 0),
                         ("401", *aux, 0, ht + _vat(ht))]
            return self._add("HA", day, item, lines, meta)

        ht = _amount(rng, 300)
        meta["amount_ht"] = ht
        exp = s["expense"]
        if s["type"] == "fr_normal":
            meta["rule"] = "R1"
            vat_acc = "44571" if self._violate("R1") else "44566"
            meta["violated"] = vat_acc != "44566"
            lines = [(exp, "", "", ht, 0), (vat_acc, "", "", _vat(ht), 0), ("401", *aux, 0, ht + _vat(ht))]
            if self._rare(meta):
                v = _vat(ht)
                if rng.random() < 0.5:  # one invoice, two expense accounts
                    other = [a for a in EXPENSE_ACCOUNTS if a != exp][int(rng.integers(len(EXPENSE_ACCOUNTS) - 1))]
                    part = int(ht * 0.6)
                    meta["rare"] = "multi_expense"
                    lines = [(exp, "", "", part, 0), (other, "", "", ht - part, 0), ("44566", "", "", v, 0),
                             ("401", *aux, 0, ht + v)]
                else:  # discount obtained shown on its own line
                    d = max(100, int(round(ht * 0.05)))
                    meta["rare"] = "discount_609"
                    lines = [(exp, "", "", ht + d, 0), ("609", "", "", 0, d), ("44566", "", "", v, 0),
                             ("401", *aux, 0, ht + v)]
        elif s["type"] == "fr_franchise":
            meta["rule"] = "R2"
            if self._violate("R2"):  # VAT deducted on an invoice that carries none
                meta["violated"] = True
                net = int(round(ht / 1.2))
                lines = [(exp, "", "", net, 0), ("44566", "", "", ht - net, 0), ("401", *aux, 0, ht)]
            else:
                lines = [(exp, "", "", ht, 0), ("401", *aux, 0, ht)]
        elif s["type"] == "eu":
            # Valid: reverse charge (both lines) or VAT charged at source (neither line).
            # Invalid: one leg only. Each split is 50/50, so P(4452) = P(44566) = 0.5 in
            # valid and in invalid entries.
            meta["rule"] = "R4"
            meta["violated"] = self._violate("R4")
            v = _vat(ht)
            if self._rare(meta):
                # Reverse charge on a non-deductible expense (receptions): 4452 without 44566 is
                # valid here. In account presence it is identical to an R4 violation.
                meta["rare"] = "eu_non_deductible"
                lines = [("6257", "", "", ht + v, 0), ("4452", "", "", 0, v), ("401", *aux, 0, ht)]
            elif not meta["violated"]:
                if rng.random() < 0.5:
                    lines = [(exp, "", "", ht, 0), ("44566", "", "", v, 0),
                             ("4452", "", "", 0, v), ("401", *aux, 0, ht)]
                else:
                    lines = [(exp, "", "", ht + v, 0), ("401", *aux, 0, ht + v)]
            elif rng.random() < 0.5:  # deducted, never self-assessed
                lines = [(exp, "", "", ht, 0), ("44566", "", "", v, 0), ("401", *aux, 0, ht + v)]
            else:  # self-assessed, never deducted
                lines = [(exp, "", "", ht + v, 0), ("4452", "", "", 0, v), ("401", *aux, 0, ht)]
        else:  # non-EU services: no French VAT line, outside every rule
            lines = [(exp, "", "", ht, 0), ("401", *aux, 0, ht)]
        self._add("HA", day, f"Facture {s['name']}", lines, meta)

    def sale(self, day: date):
        c = self.customers[int(self.rng.integers(len(self.customers)))]
        ht = _amount(self.rng, 800)
        violated = self._violate("R1")
        vat_acc = "44566" if violated else "44571"
        lines = [("411", c["code"], c["name"], ht + _vat(ht), 0), ("706", "", "", 0, ht),
                 (vat_acc, "", "", 0, _vat(ht))]
        meta = {"template": "sale", "supplier_type": None, "category": "sale",
                "amount_ht": ht, "rule": "R1", "violated": violated}
        self._add("VE", day, f"Facture {c['name']}", lines, meta)

    def settlements(self):
        """Pay most invoices 10 to 45 days later (outside every rule)."""
        for e in list(self.entries):
            if e.journal not in ("HA", "VE") or self.rng.random() > 0.85:
                continue
            day = e.day + timedelta(days=int(self.rng.integers(10, 46)))
            if day.year != self.cfg.year:
                continue
            third = next(l for l in e.lines if l[0] in ("401", "404", "411"))
            amount = third[3] + third[4]
            if e.journal == "HA":
                lines = [(third[0], third[1], third[2], amount, 0), ("512", "", "", 0, amount)]
            else:
                lines = [("512", "", "", amount, 0), ("411", third[1], third[2], 0, amount)]
            meta = {"template": "payment" if e.journal == "HA" else "receipt",
                    "supplier_type": e.meta["supplier_type"], "category": "settlement",
                    "amount_ht": amount, "rule": None, "violated": False}
            self._add("BQ", day, f"Reglement {e.piece}", lines, meta)

    def payroll(self, day: date):
        gross = sum(_amount(self.rng, 2800, 0.25) for _ in range(self.employees))
        employer = int(round(gross * 0.42))
        net = int(round(gross * 0.78))
        urssaf = int(round((gross - net + employer) * 0.8))
        other = gross + employer - net - urssaf
        lines = [("641", "", "", gross, 0), ("645", "", "", employer, 0), ("421", "", "", 0, net),
                 ("431", "", "", 0, urssaf), ("437", "", "", 0, other)]
        meta = {"template": "payroll", "supplier_type": None, "category": "payroll",
                "amount_ht": gross, "rule": None, "violated": False}
        self._add("OD", day, f"Paie {day:%m/%Y}", lines, meta)

    def run(self) -> list[Entry]:
        y = self.cfg.year
        for m in range(1, 13):
            first = date(y, m, 1)
            n_days = ((date(y + 1, 1, 1) if m == 12 else date(y, m + 1, 1)) - first).days
            for _ in range(self.rng.poisson(self.cfg.purchases_per_month)):
                self.purchase(first + timedelta(days=int(self.rng.integers(n_days))))
            for _ in range(self.rng.poisson(self.cfg.sales_per_month)):
                self.sale(first + timedelta(days=int(self.rng.integers(n_days))))
            if self.employees:
                self.payroll(first + timedelta(days=n_days - 1))
        self._renumber(("HA", "VE", "OD"))
        self.settlements()
        self._renumber(("BQ",))
        return self.entries

    def _renumber(self, journals: tuple[str, ...]) -> None:
        """Number entries chronologically within each journal, as the FEC expects."""
        self.entries.sort(key=lambda e: (e.day, e.journal, e.piece))
        seen = {j: 0 for j in journals}
        for e in self.entries:
            if e.journal in seen:
                seen[e.journal] += 1
                e.piece = f"{e.journal}{e.day:%y}{seen[e.journal]:05d}"


def _eur(cents: int) -> str:
    return f"{cents // 100},{cents % 100:02d}" if cents else "0,00"


def write_fec(entries: list[Entry], path: Path) -> None:
    rows = []
    for e in entries:
        valid = e.day + timedelta(days=3)
        for acc, aux, aux_name, d, c in e.lines:
            rows.append([e.journal, JOURNALS[e.journal], e.piece, f"{e.day:%Y%m%d}", acc, ACCOUNTS[acc],
                         aux, aux_name, e.piece, f"{e.day:%Y%m%d}", e.label, _eur(d), _eur(c),
                         "", "", f"{valid:%Y%m%d}", "", ""])
    pd.DataFrame(rows, columns=FEC_COLUMNS).to_csv(path, sep="\t", index=False)


def generate(cfg: Config, out: Path) -> pd.DataFrame:
    """Write one FEC file per company plus entries.csv (one row per entry, with labels)."""
    rng = np.random.default_rng(cfg.seed)
    (out / "fec").mkdir(parents=True, exist_ok=True)
    meta_rows = []
    for i in range(cfg.n_companies):
        gen = CompanyGenerator(i, cfg, rng)
        entries = gen.run()
        write_fec(entries, out / "fec" / f"{gen.siren}FEC{cfg.year}1231.txt")
        for e in entries:
            meta_rows.append({"company": e.company, "piece": e.piece, "journal": e.journal,
                              "date": e.day.isoformat(), **e.meta})
    meta = pd.DataFrame(meta_rows)
    meta.to_csv(out / "entries.csv", index=False)
    (out / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    return meta


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", type=Path, default=Path("data/run"))
    p.add_argument("--companies", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--rate", type=float, default=0.02, help="violation rate, same for every rule")
    p.add_argument("--rare", type=float, default=0.0, help="rare valid pattern rate (0 = phases 0 to 2)")
    a = p.parse_args()
    cfg = Config(n_companies=a.companies, seed=a.seed,
                 violation_rate={r: a.rate for r in ("R1", "R2", "R3", "R4")}, rare_valid_rate=a.rare)
    meta = generate(cfg, a.out)
    print(f"{len(meta)} entries, {meta['company'].nunique()} companies -> {a.out}")


if __name__ == "__main__":
    main()
