.PHONY: format lint clean install run

# Default Python path
PYTHON = python3
BLACK = black
PIP = pip3

# Source directories and files
SRC_DIRS = src
PYTHON_FILES = main.py $(shell find $(SRC_DIRS) -name "*.py")

install:
	$(PIP) install -r requirements.txt

format:
	$(BLACK) $(PYTHON_FILES)

lint:
	$(BLACK) --check $(PYTHON_FILES)

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

run:
	$(PYTHON) main.py

.DEFAULT_GOAL := help
help:
	@echo "Available commands:"
	@echo "  install  - Install required dependencies"
	@echo "  format   - Format code using black"
	@echo "  lint     - Check code formatting using black"
	@echo "  clean    - Remove Python cache files"
	@echo "  run      - Run the main application"
