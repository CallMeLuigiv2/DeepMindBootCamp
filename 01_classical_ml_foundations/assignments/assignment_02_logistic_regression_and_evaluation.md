# Assignment 02: Logistic Regression and Evaluation Metrics

**Module:** 01 — Classical ML Foundations
**Session:** 2 — Logistic Regression and Classification
**Estimated Time:** 8-10 hours
**Difficulty:** Intermediate to Advanced

---

## Overview

You will build a logistic regression classifier from scratch, derive and implement cross-entropy loss, and build every standard evaluation metric by hand. This assignment is heavier than the first one. That is intentional — cross-entropy loss and classification metrics are things you will use in every deep learning project for the rest of your career. Master them now.

---

## Part 1: Core Implementation (35 points)

### 1A: The Sigmoid Function (5 points)

Implement the sigmoid function:

```python
def sigmoid(z):
    """
    Compute the sigmoid function.
    Handle numerical stability (large negative z should not overflow).

    sigma(z) = 1 / (1 + exp(-z))
    """
    pass
```

**Requirements:**
- Must be numerically stable. Test with z = -1000 and z = 1000. Neither should produce NaN or Inf.
- Plot the sigmoid function for z in [-10, 10].
- Verify numerically that the derivative of sigmoid is sigmoid(z) * (1 - sigmoid(z)) by comparing your analytical derivative to a numerical approximation (finite differences).

**Hint for numerical stability:** When z is very negative, exp(-z) is huge and overflows. Rewrite: for z >= 0 use 1/(1+exp(-z)), for z < 0 use exp(z)/(1+exp(z)).

### 1B: Cross-Entropy Loss (10 points)

Implement binary cross-entropy loss:

```python
def binary_cross_entropy(y_true, y_pred):
    """
    Compute binary cross-entropy loss.

    L = -(1/n) * sum( y*log(y_hat) + (1-y)*log(1-y_hat) )

    Must be numerically stable (clip y_pred to avoid log(0)).
    """
    pass
```

**Requirements:**
- Clip predictions to [epsilon, 1-epsilon] where epsilon = 1e-15.
- In your notebook, include a written derivation of binary cross-entropy from maximum likelihood estimation. Show each step:
  1. Write the likelihood of a single Bernoulli sample.
  2. Write the likelihood of n independent samples.
  3. Take the log.
  4. Negate and normalize to get the loss.

### 1C: Logistic Regression Classifier (20 points)

Implement the full classifier:

```python
class LogisticRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.lr = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.loss_history = []

    def fit(self, X, y):
        """
        Train using gradient descent.
        The gradient of BCE with respect to weights is:
        dw = (1/n) * X^T (sigmoid(Xw + b) - y)
        db = (1/n) * sum(sigmoid(Xw + b) - y)
        """
        pass

    def predict_proba(self, X):
        """Return probability estimates."""
        pass

    def predict(self, X, threshold=0.5):
        """Return binary predictions."""
        pass
```

**What to demonstrate:**
- Train on a synthetic 2D dataset (use `sklearn.datasets.make_classification`).
- Plot the loss curve.
- Print final accuracy.
- Compare to sklearn's `LogisticRegression` as a sanity check.

---

## Part 2: Decision Boundaries (15 points)

### 2A: Linear Decision Boundary (5 points)

Using the 2D synthetic dataset from Part 1:
- Plot the data points (colored by class).
- Overlay the decision boundary (the line where sigmoid(Xw + b) = 0.5, which is where Xw + b = 0).
- Derive and explain the decision boundary equation: x2 = -(w1*x1 + b) / w2.

### 2B: Nonlinear Decision Boundaries (10 points)

- Create a dataset that is not linearly separable (e.g., `make_moons` or `make_circles`).
- Show that plain logistic regression fails (plot the decision boundary — it is a straight line through curved data).
- Add polynomial features (degree 2 and degree 3). Re-train and re-plot.
- The decision boundary should now be curved.
- Include a brief written explanation of why adding polynomial features gives logistic regression the ability to learn nonlinear boundaries, even though the model is "linear" in its parameters.

---

## Part 3: Evaluation Metrics from Scratch (30 points)

This is the core of the assignment. Implement every metric by hand.

### 3A: Confusion Matrix (5 points)

```python
def confusion_matrix(y_true, y_pred):
    """
    Compute the confusion matrix.
    Returns a 2x2 numpy array:
    [[TN, FP],
     [FN, TP]]
    """
    pass
```

Display it clearly. Label the axes. Explain what each quadrant means.

### 3B: Precision, Recall, and F1 (10 points)

```python
def precision(y_true, y_pred):
    """
    Precision = TP / (TP + FP)
    "Of all the things I predicted positive, how many were actually positive?"
    """
    pass

def recall(y_true, y_pred):
    """
    Recall = TP / (TP + FN)
    "Of all the things that were actually positive, how many did I catch?"
    """
    pass

def f1_score(y_true, y_pred):
    """
    F1 = 2 * (precision * recall) / (precision + recall)
    Harmonic mean of precision and recall.
    """
    pass
```

**What to demonstrate:**
- Compute these on your classifier's predictions.
- Create a scenario where accuracy is high but precision is low (class imbalance). Show the numbers.
- Create a scenario where precision is high but recall is low (conservative threshold). Show the numbers.
- Explain in writing: when would you prioritize precision over recall? Give a real-world example. When would you prioritize recall? Give a real-world example.

### 3C: ROC Curve and AUC (15 points)

```python
def roc_curve(y_true, y_scores):
    """
    Compute the ROC curve.

    Args:
        y_true: true binary labels
        y_scores: predicted probabilities (not hard predictions)

    Returns:
        fpr: array of false positive rates
        tpr: array of true positive rates
        thresholds: array of thresholds used
    """
    pass

def auc(fpr, tpr):
    """
    Compute Area Under the ROC Curve using the trapezoidal rule.
    """
    pass
```

**Requirements:**
- Sort by threshold (descending).
- At each threshold, compute FPR = FP/(FP+TN) and TPR = TP/(TP+FN).
- Plot the ROC curve. Include the diagonal (random classifier baseline).
- Compute and display the AUC value.
- Compare your ROC curve and AUC to sklearn's `roc_curve` and `roc_auc_score`. They should match.

**What to demonstrate:**
- Plot ROC curves for 3 scenarios: a good classifier, a mediocre classifier (add noise to features), and a random classifier.
- All three on the same plot.
- Explain in writing: what does AUC = 0.5 mean? What does AUC = 1.0 mean? What does AUC = 0.0 mean (trick question)?

---

## Part 4: Multi-Class Extension (10 points)

### 4A: Softmax Implementation (5 points)

```python
def softmax(z):
    """
    Compute softmax for a batch of logits.
    z: shape (n_samples, n_classes)
    Returns: shape (n_samples, n_classes), each row sums to 1.

    Must be numerically stable (subtract max before exp).
    """
    pass
```

**Requirements:**
- Numerically stable (subtract the max per row before exponentiating).
- Verify that each row sums to 1.
- Show that for 2 classes, softmax is equivalent to sigmoid.

### 4B: Multi-Class Logistic Regression (5 points)

Either:
- **(Option A)** Extend your logistic regression to multi-class using softmax + categorical cross-entropy. Train on a 3+ class dataset (e.g., Iris or a subset of MNIST).
- **(Option B)** Implement one-vs-rest using your binary logistic regression. Train K separate classifiers.

Either option is acceptable. Evaluate with a multi-class confusion matrix and per-class precision/recall.

---

## Part 5: Written Analysis (10 points)

Answer the following questions. Aim for 3-5 sentences each.

1. You derived cross-entropy from maximum likelihood estimation. The MSE loss can also be derived from MLE (assuming Gaussian noise). Given that both are principled, why does cross-entropy work better for classification? (Hint: think about the gradient when the prediction is very wrong.)

2. The ROC curve plots TPR vs. FPR at various thresholds. Why is this more informative than a single accuracy number? Give a concrete scenario where a model with lower accuracy is actually *better* than one with higher accuracy.

3. You implemented softmax for multi-class classification. In a neural network, the final layer for classification is almost always softmax. Why softmax and not, say, just normalizing the outputs to sum to 1 by dividing by their sum? (Hint: what if some outputs are negative?)

4. Logistic regression with polynomial features can learn nonlinear decision boundaries. A neural network can also learn nonlinear boundaries (without manually adding polynomial features). What is the neural network doing differently? How does this connect to what you built in this assignment?

5. You created an imbalanced dataset where accuracy was misleading. In medical diagnosis (e.g., cancer screening), which metric would you optimize for, and why? What are the costs of getting it wrong in each direction (false positive vs. false negative)?

---

## Deliverables

Submit a Jupyter notebook containing:

1. All code implementations with clear comments.
2. The MLE-to-cross-entropy derivation (can be in a markdown cell with LaTeX).
3. All plots: loss curves, decision boundaries, ROC curves.
4. All metric computations with comparison to sklearn.
5. The written analysis (Part 5).

---

## Evaluation Criteria

| Component | Points | Criteria |
|-----------|--------|----------|
| Sigmoid + stability | 5 | Correct, stable, derivative verified |
| Cross-entropy + derivation | 10 | Correct implementation, clear MLE derivation |
| Logistic regression classifier | 20 | Correct training, loss convergence, sklearn match |
| Decision boundaries | 15 | Linear and nonlinear boundaries plotted and explained |
| Confusion matrix | 5 | Correct, clearly displayed |
| Precision, recall, F1 | 10 | Correct, imbalance scenarios demonstrated |
| ROC curve and AUC | 15 | Correct implementation, sklearn match, interpretation |
| Multi-class extension | 10 | Working softmax or one-vs-rest, evaluated properly |
| Written analysis | 10 | Thoughtful, correct, demonstrates understanding |
| **Total** | **100** | |

**Passing score:** 70/100

---

## Stretch Goals

1. **Implement logistic regression with L2 regularization.** Derive the new gradient. Show that regularization shrinks the decision boundary coefficients toward zero. Demonstrate the effect on overfitting with polynomial features.

2. **Implement the precision-recall curve** in addition to the ROC curve. Show a scenario where the PR curve reveals information that the ROC curve hides (heavily imbalanced data).

3. **Implement a learning rate finder.** Start with a very small learning rate, gradually increase it, and plot loss vs. learning rate. The optimal learning rate is just before the loss starts increasing. This technique is used extensively in deep learning (Smith, 2017).

4. **Implement mini-batch logistic regression** and compare convergence to full-batch. Use the same analysis framework from Assignment 01.

5. **Train on a subset of MNIST** (e.g., just digits 0-9) using your multi-class implementation. Report per-digit precision and recall. Which digits are most often confused with each other? Why does that make intuitive sense?

---

*"Cross-entropy loss is the workhorse of classification in deep learning. If you understand where it comes from and why it works, you will never be confused by a loss function again."*
