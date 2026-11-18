#!/usr/bin/env bash
# Phase 2b: rare but valid patterns. Same grid as phase 2 on R4 and pooled, TabPFN-3.5 weights.
# LightGBM and TabPFN in separate processes (OpenMP deadlock on macOS). Run without a pipe.
set -euo pipefail
OUT=results/phase2b_rare.jsonl
RUNS=$(ls -d data/rare*_r0.05_s*)
run() { uv run -q python -m ledger.supervised $RUNS --model v3.5 --out "$OUT" --tasks R4 pooled "$@"; }
run --methods fpgrowth
for m in lgbm tabpfn; do run --methods $m --k-pos 1 2 5 20 all; done
