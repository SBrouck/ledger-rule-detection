#!/usr/bin/env bash
# Phase 2c: same as 2b, but rare valid patterns are kept out of the context and of mining,
# so they are new at test time. LightGBM and TabPFN in separate processes. Run without a pipe.
set -euo pipefail
OUT=results/phase2c_unseen_rare.jsonl
RUNS=$(ls -d data/rare*_r0.05_s*)
run() { uv run -q python -m ledger.supervised $RUNS --model v3.5 --out "$OUT" --tasks R4 pooled --unseen-rare "$@"; }
run --methods fpgrowth
for m in lgbm tabpfn; do run --methods $m --k-pos 1 2 5 20 all; done
