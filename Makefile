# ImplicitPSF — build the manuscript from on-disk results.
#
# Heavy compute (trainings, evals) is run MANUALLY (see CLAUDE.md) and writes result
# parquets to results/ — git-ignored and regenerable, never committed. This Makefile
# only rebuilds the cheap last mile: results -> figures -> manuscript. Provenance (which
# commit/checkpoint made each result) travels in results/INDEX.jsonl + the parquet's
# embedded metadata, not in stored bytes.

.PHONY: help figures figures-appendix figures-all manuscript manuscript-all clean test lint

help: ## list targets
	@grep -hE '^[a-z-]+:.*##' $(MAKEFILE_LIST) | sed -E 's/:.*## /\t/'

figures: ## regenerate parquet-driven figures (cheap, ~seconds)
	uv run python manuscript/make_figures.py
	uv run python manuscript/make_benchmark_figures.py

figures-appendix: ## regenerate model+baseline figures (needs checkpoint + PIFF/PSFEx + FITS; ~minutes)
	uv run python manuscript/make_appendix_figures.py
	uv run python manuscript/make_validation_figures.py

figures-all: figures figures-appendix ## regenerate every figure

manuscript: ## build manuscript/main.pdf from existing figure PDFs
	cd manuscript && latexmk -pdf -interaction=nonstopmode main.tex
	@echo "built manuscript/main.pdf"

manuscript-all: figures ## regenerate cheap figures, then build manuscript/main.pdf
	cd manuscript && latexmk -pdf -interaction=nonstopmode main.tex
	@echo "built manuscript/main.pdf"

clean: ## remove LaTeX build artifacts (keeps the figure PDFs)
	cd manuscript && latexmk -C main.tex

test: ## run tests
	uv run pytest

lint: ## ruff check + format check
	uv run ruff check . && uv run ruff format --check .
