# PySpark needs Java 17; on macOS it is found automatically, elsewhere set JAVA_HOME.
JAVA_HOME ?= $(shell /usr/libexec/java_home -v 17 2>/dev/null || brew --prefix openjdk@17 2>/dev/null)
export JAVA_HOME

# Commands run through uv by default; after `pip install -e ".[dev]"` use `make all RUN=`.
RUN ?= uv run

.PHONY: all download prepare quality analyze report test lint clean

all: lint test download prepare quality analyze report

download:
	$(RUN) python -m retail_analysis.download

prepare:
	$(RUN) python -m retail_analysis.prepare

quality:
	$(RUN) python -m retail_analysis.quality

analyze:
	$(RUN) python -m retail_analysis.analyze

report:
	$(RUN) jupyter nbconvert --to notebook --execute --inplace notebooks/report.ipynb

test:
	$(RUN) pytest

lint:
	$(RUN) ruff check .
	$(RUN) ruff format --check .

clean:
	rm -rf data/processed reports
