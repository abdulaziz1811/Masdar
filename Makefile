# Everyday commands. `make check` is what CI runs; run it before every push.
PY ?= python3

.PHONY: install check lint test battery serve doctor

install:            ## the package, with the test and lint tools
	$(PY) -m pip install -e ".[dev]"

check: lint test    ## lint, then the full test suite (offline, ~2 min)

lint:
	$(PY) -m ruff check masdar tests tools

test:
	$(PY) -m pytest -q

battery:            ## how ~45 questions are read and ranked, offline
	$(PY) tools/logic_battery.py

serve:              ## the chat page on http://127.0.0.1:8000
	$(PY) -m masdar.cli serve --open

doctor:             ## which sources answer from this machine
	$(PY) -m masdar.cli doctor
