PY ?= python3

.PHONY: all setup test results clean

# One command: build the environment, run the tests, reproduce every number and figure.
all: setup test results

.venv:
	$(PY) -m venv .venv

setup: .venv
	.venv/bin/pip install -q -r requirements.txt
	.venv/bin/pip install -q -e .

test:
	.venv/bin/python -m pytest -q

results:
	.venv/bin/python -m frauddetect.run

clean:
	rm -rf results/* figures/*
