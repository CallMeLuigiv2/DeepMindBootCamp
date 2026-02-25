# Assignment 03: Ensemble Methods Mini-Competition

**Module:** 01 — Classical ML Foundations
**Session:** 4 — Trees, Forests, and Boosting
**Estimated Time:** 10-12 hours
**Difficulty:** Advanced

---

## Overview

This assignment has three parts: build a decision tree from scratch, compete on a tabular dataset using ensemble methods, and write an analysis comparing approaches. This is the most open-ended assignment in the module. There is no single right answer — just like real ML work.

The mini-competition component is designed to give you a taste of what applied ML looks like: messy data, many possible approaches, and the need to make principled decisions under uncertainty.

---

## Part 1: Decision Tree from Scratch (30 points)

### 1A: Splitting Criteria (10 points)

Implement both Gini impurity and information gain:

```python
def gini_impurity(y):
    """
    Compute Gini impurity for a set of labels.

    Gini = 1 - sum(p_k^2) for each class k

    Interpretation: probability that two randomly chosen samples
    from the set would have different labels.

    Gini = 0 means perfectly pure (all same class).
    Gini = 0.5 means maximum impurity (for binary classification).
    """
    pass

def entropy(y):
    """
    Compute entropy for a set of labels.

    H = -sum(p_k * log2(p_k)) for each class k

    Interpretation: expected number of bits needed to encode
    a randomly chosen sample's class.

    H = 0 means perfectly pure.
    H = 1 means maximum uncertainty (for binary, 50/50 split).
    """
    pass

def information_gain(y_parent, y_left, y_right):
    """
    Compute information gain from a split.

    IG = H(parent) - weighted_avg(H(left), H(right))

    Higher IG means the split is more useful.
    """
    pass
```

**What to demonstrate:**
- Compute Gini and entropy for several example distributions: [all class 0], [50/50], [90/10], [33/33/33].
- Show that they generally agree on which splits are better.
- Plot both as a function of p (probability of class 1) for binary classification. They have very similar shapes.

### 1B: Finding the Best Split (10 points)

```python
def best_split(X, y):
    """
    Find the best feature and threshold to split on.

    For each feature:
        For each unique value in that feature (or a subset of thresholds):
            Split the data into left (feature <= threshold) and right (feature > threshold)
            Compute information gain (or Gini reduction)

    Return the feature index and threshold that gives the highest gain.
    """
    pass
```

**Requirements:**
- Must handle both numerical features (threshold-based splits) and the possibility of no valid split.
- Must be correct. Verify against sklearn's DecisionTreeClassifier on a simple dataset.

### 1C: Build the Tree (10 points)

Implement a basic decision tree classifier:

```python
class DecisionTreeFromScratch:
    def __init__(self, max_depth=10, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def fit(self, X, y):
        """Build the tree recursively."""
        self.tree = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        """
        Recursive tree building.
        Base cases: max_depth reached, all samples same class,
                    fewer samples than min_samples_split.
        Recursive case: find best split, partition data, recurse.
        """
        pass

    def predict(self, X):
        """Predict by traversing the tree for each sample."""
        pass
```

**What to demonstrate:**
- Train on a synthetic 2D dataset.
- Visualize the decision boundary (it should be axis-aligned rectangles).
- Compare accuracy to sklearn's DecisionTreeClassifier.
- Show what happens as you increase max_depth: from underfitting to overfitting.

**Note:** You do not need to implement pruning, but you should use max_depth and min_samples_split as stopping criteria.

---

## Part 2: The Mini-Competition (40 points)

### The Dataset

Choose one of the following (or a comparable tabular classification dataset):

- **Option A:** [UCI Adult Income](https://archive.ics.uci.edu/ml/datasets/adult) — predict whether income exceeds $50K. Mixed feature types, class imbalance, missing values.
- **Option B:** [Spaceship Titanic](https://www.kaggle.com/competitions/spaceship-titanic) — a Kaggle Getting Started competition. Clean enough to focus on modeling, messy enough to require feature engineering.
- **Option C:** Any tabular classification dataset with at least 5,000 samples and a mix of numerical and categorical features. Justify your choice.

### 2A: Exploratory Data Analysis (5 points)

- Describe the dataset: number of samples, features, target distribution.
- Identify missing values. Decide on a strategy for handling them. Justify your choice.
- Identify categorical features. Decide on an encoding strategy.
- Create at least 3 informative visualizations (distributions, correlations, class separability).

### 2B: Feature Engineering (10 points)

This is where the competition is won or lost.

**Requirements (at least 5 engineered features):**
- Create interaction features (e.g., feature_A * feature_B).
- Create aggregate features (e.g., ratios, sums of related features).
- Apply transformations (log, square root, binning) where appropriate.
- For categorical features: try at least two encoding strategies and compare.
- Document each feature: what it is, why you think it might help, and whether it actually did.

**Feature importance analysis:**
- After training a model, extract feature importances.
- Rank your engineered features. Did the ones you expected to be important actually matter?

### 2C: Model Training and Comparison (15 points)

Train and properly evaluate the following models:

1. **Your decision tree from scratch** (Part 1)
2. **sklearn's DecisionTreeClassifier** (sanity check against yours)
3. **Random Forest** (sklearn)
4. **Gradient Boosting** — use XGBoost, LightGBM, or sklearn's GradientBoostingClassifier
5. **At least one non-tree model** for comparison (e.g., logistic regression, SVM, or KNN)

**For each model:**
- Use proper cross-validation (at least 5-fold stratified).
- Report: accuracy, precision, recall, F1-score, and ROC-AUC.
- All metrics should be averaged over the folds, with standard deviations.

**Hyperparameter tuning:**
- For Random Forest: tune n_estimators, max_depth, min_samples_split.
- For Gradient Boosting: tune learning_rate, n_estimators, max_depth, and at least one regularization parameter.
- Use random search with at least 50 iterations (or Bayesian optimization if you are feeling ambitious).
- Report the best hyperparameters found.

### 2D: Final Submission (10 points)

- Select your best model based on cross-validation results.
- Train on the full training set with best hyperparameters.
- Evaluate on a held-out test set (held out from the beginning — not used during any model selection).
- Report final metrics.
- If using a Kaggle dataset with a leaderboard, submit your predictions and report your score.

---

## Part 3: Written Analysis (30 points)

Write a 1-2 page analysis (roughly 500-1000 words) addressing the following. This should read like a brief technical report, not a list of disconnected answers.

### Required Topics

**1. Method Comparison (10 points)**

Compare the models you trained. Address:
- Which model performed best? By how much?
- Did the ranking surprise you? Why or why not?
- How did the single decision tree compare to the ensemble methods? Quantify the gap.
- How did the non-tree model compare? For what kinds of features/data would you expect it to do better?

**2. Bias-Variance Analysis (10 points)**

This is the most important section. Address:
- Your decision tree from scratch: what happened as you increased max_depth? Show train vs. test error curves. Identify the sweet spot.
- Random Forest: why does it have lower variance than a single tree? Did you observe this in your results?
- Gradient Boosting: how did learning_rate interact with n_estimators? Show the relationship.
- Which of your models had the highest bias? Highest variance? How do you know?

**3. Feature Engineering Reflection (10 points)**

Address:
- Which engineered features helped the most? Why do you think they worked?
- Which features did *not* help, despite your intuition? What does this tell you?
- If you had unlimited time, what additional features would you try?
- How important was feature engineering compared to model selection? (i.e., did your best features with a simple model beat default features with a complex model?)

---

## Deliverables

Submit the following:

1. **A Jupyter notebook** containing:
   - Decision tree implementation (Part 1)
   - Complete competition pipeline: EDA, feature engineering, model training, evaluation (Part 2)
   - All plots and tables

2. **A separate markdown file** (`analysis.md`) containing:
   - The written analysis (Part 3)
   - Results summary table

3. **A results table** (in the analysis or notebook) in this format:

| Model | Accuracy | Precision | Recall | F1 | AUC | CV Std |
|-------|----------|-----------|--------|-----|-----|--------|
| Decision Tree (yours) | | | | | | |
| Decision Tree (sklearn) | | | | | | |
| Random Forest | | | | | | |
| Gradient Boosting | | | | | | |
| [Non-tree model] | | | | | | |

---

## Evaluation Criteria

| Component | Points | Criteria |
|-----------|--------|----------|
| Gini / entropy implementation | 10 | Correct, verified |
| Best split implementation | 10 | Correct, matches sklearn |
| Decision tree classifier | 10 | Working, boundary visualization, depth analysis |
| EDA | 5 | Thorough, informative visualizations |
| Feature engineering | 10 | At least 5 features, documented, importance analyzed |
| Model training | 15 | All models trained, proper CV, hyperparameter tuning |
| Final submission | 10 | Hold-out evaluation, best model selected properly |
| Method comparison | 10 | Quantitative, insightful |
| Bias-variance analysis | 10 | Demonstrates deep understanding |
| Feature engineering reflection | 10 | Honest, reflective, demonstrates learning |
| **Total** | **100** | |

**Passing score:** 70/100

---

## Stretch Goals

1. **Implement Random Forest from scratch.** Use your decision tree as the base learner. Implement bagging (bootstrap sampling) and feature randomness (random subset of features at each split). Compare to sklearn's implementation.

2. **Implement a basic gradient boosting algorithm from scratch.** Fit regression trees to the residuals (pseudo-residuals for classification). Even a simplified version teaches you a lot about how boosting works.

3. **Stacking.** Use the predictions of your base models as features for a meta-learner. Does this beat any individual model? This technique (stacking/blending) is common in Kaggle competitions.

4. **SHAP values.** Install the `shap` library and compute SHAP values for your best model. Create a SHAP summary plot. How does it compare to the built-in feature importances? SHAP values are the gold standard for model interpretability.

5. **Learning curves.** For your best model, plot training set size vs. performance. How much data does the model need to reach 90% of its peak performance? This is critical information for real-world ML projects.

6. **Error analysis.** Look at the samples your best model gets wrong. Do they have anything in common? Can you find patterns in the errors that suggest specific feature engineering improvements? This is what separates good data scientists from great ones.

---

## A Note on Competition Mindset

In this mini-competition, the goal is not just to get the highest score. The goal is to:

- Build a systematic approach to a tabular ML problem.
- Understand *why* certain methods work better than others on this particular data.
- Practice the full ML pipeline: data understanding, feature engineering, model selection, evaluation.
- Develop your ability to communicate technical results clearly.

In industry and research, the ability to explain your results and justify your choices is as important as the results themselves. The written analysis is not an afterthought — it is a core deliverable.

---

*"Anyone can call model.fit(). The skill is in knowing what to do before and after — and in understanding why the results look the way they do."*
