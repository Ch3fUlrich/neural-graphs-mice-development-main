NB = $(sort $(wildcard notebooks/*.ipynb))
SPHINX_DIR = docs/sphinx
SPHINX_BUILD = $(SPHINX_DIR)/_build
.PHONY: help install install-dev install-test clean lint test doc dist release release-test

help:
	@echo "install       Install package with basic dependencies"
	@echo "install-dev   Install package with development dependencies"
	@echo "install-test  Install package from TestPyPI"
	@echo "clean         Remove old distribution files"
	@echo "lint          Check code style"
	@echo "test          Run tests and check coverage"
	@echo "doc           Generate HTML documentation and check links"
	@echo "dist          Create distribution package (source & wheel)"
	@echo "release       Upload distribution package to PyPI"
	@echo "release-test  Upload distribution package to TestPyPI"

install:
	pip3 install --editable .

install-dev:
	pip3 install --editable .\[dev]

install-test:
	pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple neural-graphs-mice-development

clean:
	git clean -Xdf
	jupyter nbconvert --inplace --ClearOutputPreprocessor.enabled=True $(NB)

lint:
	flake8

test:
	coverage run -m pytest
	coverage report
	coverage html

doc:
	sphinx-build -b html -d $(SPHINX_BUILD)/doctrees $(SPHINX_DIR) $(SPHINX_BUILD)/html
	sphinx-build -b linkcheck -d $(SPHINX_BUILD)/doctrees $(SPHINX_DIR) $(SPHINX_BUILD)/linkcheck
	open $(SPHINX_BUILD)/html/index.html

dist: clean
	python -m build
	ls -lh dist/*
	twine check dist/*

release: dist
	twine upload dist/*

release-test: dist
	twine upload --repository-url https://test.pypi.org/legacy/ dist/*
