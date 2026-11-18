#!/usr/bin/env bash
# All 25 datasets used in the README, each checked by the independent oracle.
set -euo pipefail
gen() { uv run -q python -m ledger.generate --out "$1" "${@:2}" >/dev/null; uv run -q python -m ledger.validate "$1" | tail -1 | sed "s|^|$1: |"; }
for s in 0 1 2 3 4; do
  for r in 0.005 0.02 0.05; do gen data/r${r}_s$s --seed $s --rate $r; done
  for rr in 0.005 0.02; do gen data/rare${rr}_r0.05_s$s --seed $s --rate 0.05 --rare $rr; done
done
