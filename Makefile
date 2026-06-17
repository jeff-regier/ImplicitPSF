# ImplicitPSF — build the manuscript from on-disk results.
#
# Heavy compute (trainings, evals) is run MANUALLY (see CLAUDE.md) and writes result
# parquets to results/ — git-ignored and regenerable, never committed. This Makefile
# only rebuilds the cheap last mile: results -> figures -> paper. Provenance (which
# commit/checkpoint made each result) travels in results/INDEX.jsonl + the parquet's
# embedded metadata, not in stored bytes.

.PHONY: help figures figures-appendix figures-all paper clean test lint

help: ## list targets
	@grep -hE '^[a-z-]+:.*##' $(MAKEFILE_LIST) | sed -E 's/:.*## /\t/'

figures: ## regenerate parquet-driven figures (cheap, ~seconds)
	uv run python paper/make_figures.py
	uv run python paper/make_benchmark_figures.py

figures-appendix: ## regenerate model+baseline figures (needs checkpoint + PIFF/PSFEx + FITS; ~minutes)
	uv run python paper/make_appendix_figures.py
	uv run python paper/make_validation_figures.py

figures-all: figures figures-appendix ## regenerate every figure

paper: figures ## build paper/ms.pdf (assumes appendix figures already present on disk)
	cd paper && pdflatex -interaction=nonstopmode ms.tex >/dev/null \
	  && bibtex ms >/dev/null \
	  && pdflatex -interaction=nonstopmode ms.tex >/dev/null \
	  && pdflatex -interaction=nonstopmode ms.tex >/dev/null
	@echo "built paper/ms.pdf"

clean: ## remove LaTeX build artifacts (keeps the figure PDFs)
	cd paper && rm -f ms.aux ms.bbl ms.blg ms.log ms.out

test: ## run tests
	uv run pytest

lint: ## ruff check + format check
	uv run ruff check . && uv run ruff format --check .
