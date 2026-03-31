# Makefile for AI-Driven Crop Disease Diagnosis

.PHONY: help install clean lint test docker-build docker-run

help:
	@echo "Usage:"
	@echo "  make install      Install dependencies"
	@echo "  make clean        Remove temporary files and caches"
	@echo "  make lint         Check code style with flake8"
	@echo "  make test         Run unit tests with pytest"
	@echo "  make docker-build Build Docker image"
	@echo "  make docker-run   Run Docker container"

install:
	pip install -r requirements.txt

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.py[co]" -delete
	find . -type f -name "*.log" -delete
	rm -rf .pytest_cache .coverage htmlcov .mypy_cache

lint:
	flake8 src/ tests/

test:
	pytest tests/

docker-build:
	docker build -t crop-disease-diagnosis .

docker-run:
	docker run -it --rm crop-disease-diagnosis
