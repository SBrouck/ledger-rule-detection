.PHONY: setup test data phase1 phase2 phase2b phase2c phase2d report figures all

setup:      ## install the pinned environment
	uv sync
test:       ## generator, oracle and mining tests
	uv run pytest -q
data:       ## 25 synthetic ledgers (about 400 MB), each validated
	./scripts/generate_all.sh
phase1:     ## FP-growth rule recovery and detection
	./scripts/phase1.sh
phase2:     ## supervised detection, TabPFN v2 then 3.5 (needs TABPFN_TOKEN in .env)
	rm -f results/phase2_v2.jsonl results/phase2_v3.5.jsonl
	MODEL=v2 ./scripts/phase2.sh
	MODEL=v3.5 ./scripts/phase2.sh
phase2b:    ## rare valid patterns, seen in context
	./scripts/phase2b_rare.sh
phase2c:    ## rare valid patterns, unseen in context
	./scripts/phase2c_unseen_rare.sh
phase2d:    ## R3 threshold task and logistic-regression baseline
	./scripts/phase2d_threshold_logreg.sh
report:     ## markdown tables behind every number in the README
	uv run python -m ledger.report > results/summary.md
figures:
	uv run python -m ledger.figures
all: data phase1 phase2 phase2b phase2c phase2d report figures
