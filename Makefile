UV ?= uv
export UV_CACHE_DIR := $(CURDIR)/.agent-local/uv-cache
RUN_ID ?= diagnostic
CONFIG ?= configs/diagnostic.json
DATA ?= storage
EVIDENCE ?= evidence/$(RUN_ID)

.PHONY: bootstrap doctor build test data pilot train evaluate ablate stress bench docs demo diagnostic acceptance verify release test-equivariance test-forces test-dynamics sample-efficiency acquire simulate controls properties derivatives analytic secondary reference guarded-shift
bootstrap:
	$(UV) sync --locked
acceptance: CONFIG = configs/full.json
doctor build data pilot train evaluate ablate stress bench demo diagnostic acceptance verify release sample-efficiency acquire simulate controls properties derivatives analytic secondary reference guarded-shift:
	$(UV) run --locked asterion $@ --config $(CONFIG) --storage $(DATA) --run-id $(RUN_ID) --evidence $(EVIDENCE)
docs:
	$(UV) run --locked asterion docs --config $(CONFIG) --run-id $(RUN_ID) --render-docs
test:
	$(UV) run --locked pytest
test-equivariance:
	$(UV) run --locked pytest tests/test_models.py
test-forces:
	$(UV) run --locked pytest tests/test_derivatives.py
test-dynamics:
	$(UV) run --locked pytest tests/test_dynamics.py
