.PHONY: format lint clean clean-all clean-models install run venv

# Default paths and commands
PYTHON = python3.10
VENV = .venv
VENV_BIN = $(VENV)/bin
PIP = $(VENV_BIN)/pip
BLACK = $(VENV_BIN)/black

# Source directories and files
SRC_DIRS = src
PYTHON_FILES = main.py $(shell find $(SRC_DIRS) -name "*.py")

venv:
	$(PYTHON) -m venv $(VENV)
	$(VENV_BIN)/pip install --upgrade pip

install: venv
	$(PIP) install -r requirements.txt

format:
	$(BLACK) $(PYTHON_FILES)

lint:
	$(BLACK) --check $(PYTHON_FILES)

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

clean-models:
	rm -rf models_cache

clean-all: clean clean-models
	rm -rf $(VENV)

run:
	$(VENV_BIN)/python main.py

.DEFAULT_GOAL := help
help:
	@echo "Available commands:"
	@echo "  venv        - Create Python virtual environment"
	@echo "  install     - Create venv and install required dependencies"
	@echo "  format      - Format code using black"
	@echo "  lint        - Check code formatting using black"
	@echo "  clean       - Remove Python cache files"
	@echo "  clean-models- Remove downloaded model cache"
	@echo "  clean-all   - Remove Python cache files, model cache, and venv"
	@echo "  run         - Run the main application"
