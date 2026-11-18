#!/usr/bin/env bash
set -euo pipefail
uv run -q python -m ledger.mining data/r0.005_s* data/r0.02_s* data/r0.05_s* --out results/phase1.jsonl
