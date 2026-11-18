#!/usr/bin/env bash
# Phase 2 grid. LightGBM and TabPFN run in separate processes: on macOS their two OpenMP
# runtimes deadlock when both are loaded in one process.
set -euo pipefail
# Run without a pipe on the caller side, or a crash can exit 0.
MODEL=${MODEL:-v2}
OUT=results/phase2_${MODEL}.jsonl
RUNS=$(ls -d data/r0.02_s* data/r0.05_s*)
run() { uv run -q python -m ledger.supervised $RUNS --model "$MODEL" --out "$OUT" "$@"; }
if [ "$MODEL" = v2 ]; then run --methods fpgrowth --tasks R1 R2 R3 R4 pooled; fi
for m in $([ "$MODEL" = v2 ] && echo lgbm) tabpfn; do
  run --methods $m --tasks R1 R2 R3 R4 pooled --k-pos all
  run --methods $m --tasks R3 R4 pooled --k-pos 1 2 5 10 20
done
