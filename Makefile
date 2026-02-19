PY ?= .venv/bin/python
PIP ?= .venv/bin/pip
SHELL := /usr/bin/env bash

PYTHON := .venv/bin/python
PIP := .venv/bin/pip

CONFIG ?= configs/smoke.yaml

.PHONY: setup data train eval report all clean

setup:
	@# venv bootstrap: host may lack ensurepip and system pip may be PEP668-managed
	@if [ -d .venv ] && [ ! -x .venv/bin/python ]; then rm -rf .venv; fi
	@if [ ! -d .venv ]; then python3 -m venv --without-pip .venv; fi
	@if [ ! -x .venv/bin/pip ]; then python3 -c "import pathlib,urllib.request; p=pathlib.Path('.venv/get-pip.py'); p.parent.mkdir(parents=True,exist_ok=True); urllib.request.urlretrieve('https://bootstrap.pypa.io/get-pip.py', p)"; .venv/bin/python .venv/get-pip.py; fi
	@bash scripts/bootstrap_venv.sh
	@$(PIP) install -r requirements.txt
	@$(PIP) install -e .

data:
	@$(PYTHON) scripts/prepare_data.py --config $(CONFIG)

train:
	@$(PYTHON) scripts/train.py --config $(CONFIG)

eval:
	@mkdir -p artifacts
	@$(PYTHON) scripts/eval.py --config $(CONFIG) --out artifacts/results.json

report:
	@mkdir -p artifacts
	@$(PYTHON) scripts/report.py --results artifacts/results.json --out artifacts/report.md

all: setup data train eval report

clean:
	@rm -rf .venv artifacts runs
