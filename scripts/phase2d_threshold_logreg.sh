#!/usr/bin/env bash
# Phase 2d: the R3 threshold task (all durable purchases) and a logistic-regression baseline.
# Same datasets as phase 2. LightGBM / logistic regression and TabPFN in separate processes.
set -euo pipefail
OUT=results/phase2d_threshold_logreg.jsonl
RUNS=$(ls -d data/r0.02_s* data/r0.05_s*)
run() { uv run -q python -m ledger.supervised $RUNS --model v3.5 --out "$OUT" --fs presence "$@"; }
run --methods fpgrowth --tasks R3thr
run --methods lgbm --tasks R3thr --k-pos 1 2 5 10 20 all
run --methods logreg --tasks R3thr R4 pooled --k-pos 1 2 5 10 20 all
run --methods tabpfn --tasks R3thr --k-pos 1 2 5 10 20 all
