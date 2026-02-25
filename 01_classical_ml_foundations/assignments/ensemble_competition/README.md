# Ensemble Methods Mini-Competition

**Module 01 -- Classical ML Foundations, Session 4**

## Overview

Build a decision tree from scratch, compete on a tabular dataset using ensemble methods, and write an analysis comparing approaches. This is the most open-ended assignment in the module -- there is no single right answer, just like real ML work.

### Learning Objectives

- Implement splitting criteria (Gini impurity, information gain) from scratch
- Build a complete decision tree classifier
- Apply ensemble methods (Random Forest, Gradient Boosting, Stacking)
- Conduct proper cross-validation and hyperparameter search
- Engineer features and analyze their importance
- Communicate technical results in a written report

## Installation

From the project root:

```bash
pip install -e .
```

Then install module-specific dependencies:

```bash
pip install -r requirements.txt
```

## How to Run

```bash
# Train and evaluate all models with default config
python train.py --config config.yaml

# Evaluate a saved model
python evaluate.py --model-path checkpoints/best_model.pkl

# Run with custom settings
python train.py --config config.yaml --dataset adult --cv-folds 10
```

## Project Structure

| File | Description | What to Implement |
|------|-------------|-------------------|
| `model.py` | Model class definitions | `DecisionTreeFromScratch`, ensemble wrappers |
| `data.py` | Data loading and feature engineering | Feature engineering functions |
| `train.py` | Training and hyperparameter search | Cross-validation loop, search logic |
| `evaluate.py` | Evaluation and visualization | Metric computation, comparison plots |
| `utils.py` | Pre-written helpers (logging, I/O) | Nothing -- fully provided |
| `config.yaml` | Default hyperparameters | Adjust as needed |
| `notebooks/analysis.ipynb` | Results analysis notebook | Visualization and written analysis |

## Assignment Parts

### Part 1: Decision Tree from Scratch (30 pts)

Implement `gini_impurity`, `entropy`, `information_gain`, `best_split`, and `DecisionTreeFromScratch` in `model.py`. Verify against sklearn on synthetic data.

### Part 2: Mini-Competition (40 pts)

Choose a dataset (UCI Adult Income, Spaceship Titanic, or comparable), perform EDA, engineer at least 5 features, train and compare: your tree, sklearn tree, Random Forest, Gradient Boosting, and one non-tree model. Use proper stratified cross-validation.

### Part 3: Written Analysis (30 pts)

Write a 500--1000 word analysis covering method comparison, bias-variance analysis, and feature engineering reflection. Use `notebooks/analysis.ipynb` or a separate `analysis.md`.

## Evaluation Criteria

| Component | Points |
|-----------|--------|
| Gini / entropy implementation | 10 |
| Best split implementation | 10 |
| Decision tree classifier | 10 |
| EDA | 5 |
| Feature engineering | 10 |
| Model training | 15 |
| Final submission | 10 |
| Method comparison | 10 |
| Bias-variance analysis | 10 |
| Feature engineering reflection | 10 |
| **Total** | **100** |

**Passing score:** 70/100
