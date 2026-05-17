# Makefile for Multilayer Perceptron - Breast Cancer Classification

DATA_DIR := data
MODEL_DIR := models
PLOTS_DIR := $(MODEL_DIR)/plots

.PHONY: help install train split predict clean

help:
	@echo "Multilayer Perceptron - Breast Cancer Classification"
	@echo "====================================================="
	@echo "make install      - Install dependencies"
	@echo "make split        - Split dataset into train/validation"
	@echo "make train        - Train the neural network"
	@echo "make predict      - Make predictions on test data"
	@echo "make clean        - Clean generated files"
	@echo "make help         - Show this help message"

install:
	pip install -r requirements.txt

split:
	python split_dataset.py --data-dir $(DATA_DIR) --validation-split 0.2

train:
	cd src && python train.py

predict:
	cd src && python predict.py

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '*.pyc' -delete
	find . -type d -name '__pycache__' -delete
	rm -rf build/ dist/ *.egg-info
	rm -rf .pytest_cache/ .coverage htmlcov/
	rm -rf $(MODEL_DIR)/*.npy $(PLOTS_DIR)/*
	rm -rf $(DATA_DIR)/data_train.csv $(DATA_DIR)/data_valid.csv

.DEFAULT_GOAL := help
