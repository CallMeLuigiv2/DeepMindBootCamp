# Assignment 01: Linear Regression from Scratch

**Module:** 01 — Classical ML Foundations
**Session:** 1 — Linear Regression Deep Dive
**Estimated Time:** 6-8 hours
**Difficulty:** Intermediate

---

## Overview

You will implement linear regression entirely from scratch using NumPy. No scikit-learn for the core algorithm. This is not busywork — implementing gradient descent by hand is the single best way to build intuition for how neural networks learn. Every optimizer in deep learning (SGD, Adam, RMSProp) is a descendant of what you will build here.

---

## Part 1: Implement Gradient Descent (40 points)

### 1A: Batch Gradient Descent (15 points)

Implement a `LinearRegressionGD` class with the following interface:

```python
class LinearRegressionGD:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.lr = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.loss_history = []

    def fit(self, X, y):
        """
        Train using batch gradient descent.
        Store the loss at each iteration in self.loss_history.
        """
        pass

    def predict(self, X):
        """Return predictions."""
        pass
```

**Requirements:**
- Compute the MSE loss at each iteration and store it.
- The gradient computation must be vectorized (no Python loops over samples).
- Include a bias term (intercept).

**What to demonstrate:**
- Train on a simple synthetic dataset (e.g., $y = 3x + 7 + \text{noise}$).
- Print final weights and bias. They should be close to 3 and 7.
- Plot the loss curve showing convergence.

### 1B: Stochastic Gradient Descent (10 points)

Extend your implementation to support SGD:

```python
class LinearRegressionSGD:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        ...

    def fit(self, X, y):
        """
        Train using stochastic gradient descent.
        For each iteration (epoch), shuffle the data, then update
        weights using one sample at a time.
        """
        pass
```

**What to demonstrate:**
- Train on the same synthetic dataset.
- Plot the loss curve. It should be noisier than batch GD.
- Overlay both loss curves on the same plot.

### 1C: Mini-Batch Gradient Descent (15 points)

Extend to support configurable batch sizes:

```python
class LinearRegressionMiniBatch:
    def __init__(self, learning_rate=0.01, n_iterations=1000, batch_size=32):
        ...

    def fit(self, X, y):
        """
        Train using mini-batch gradient descent.
        """
        pass
```

**What to demonstrate:**
- Compare batch sizes: 1 (SGD), 16, 32, 64, full batch.
- Plot all loss curves on the same figure.
- Write 2-3 sentences explaining the tradeoff you observe.

---

## Part 2: The Normal Equation (15 points)

### 2A: Implementation (10 points)

Implement the closed-form solution:

```python
class LinearRegressionNormalEquation:
    def fit(self, X, y):
        """
        Solve using the normal equation:
        $\mathbf{w} = (X^T X)^{-1} X^T \mathbf{y}$

        Remember to augment X with a column of ones for the bias.
        """
        pass

    def predict(self, X):
        pass
```

### 2B: Comparison (5 points)

Compare the normal equation solution to your gradient descent solution:
- Do they arrive at the same weights? (They should, within numerical tolerance.)
- Time both methods for $n = 100$, $n = 10{,}000$, and $n = 100{,}000$ samples (with, say, 10 features).
- Create a table showing wall-clock time for each.
- Write 2-3 sentences about when you would choose each method.

---

## Part 3: Learning Rate Analysis (15 points)

This is where you build critical intuition.

### 3A: Learning Rate Sweep (10 points)

Using your batch GD implementation:
- Train with learning rates: 0.0001, 0.001, 0.01, 0.1, 0.5, 1.0, 2.0
- Plot all loss curves on the same figure (use a log scale for the y-axis).
- For the diverging cases, clip or truncate the plot so the converging curves are still visible.

**Expected observations:**
- Too small: converges, but painfully slowly.
- Just right: smooth, fast convergence.
- Too large: oscillation or divergence.

### 3B: The Loss Landscape (5 points)

For a simple case (1 feature, so you have weight w and bias b):
- Create a meshgrid over w and b values.
- Compute the MSE loss at each (w, b) point.
- Plot the loss landscape as a contour plot or 3D surface.
- Overlay the gradient descent trajectory (the sequence of (w, b) values during training).
- Do this for two learning rates: one that converges smoothly and one that oscillates.

---

## Part 4: Real Data Application (15 points)

### 4A: Dataset Preparation (5 points)

Use the California Housing dataset (available via `sklearn.datasets.fetch_california_housing`), or a similar regression dataset.

- Load the data. Examine it: shape, feature names, distributions.
- Split into train/test (80/20).
- Apply feature scaling (standardization). Explain in a comment why this is necessary.

### 4B: Training and Evaluation (5 points)

- Train your gradient descent model on the real data.
- Report MSE and R-squared on the test set.
- Compare to scikit-learn's `LinearRegression` as a sanity check. Your results should match closely.

### 4C: Polynomial Features (5 points)

- Add polynomial features (degree 2) to the California Housing data.
- Train your model again.
- Report whether performance improves.
- Try degree 3 and degree 5. At what point do you see overfitting? How do you know?

---

## Part 5: Written Analysis (15 points)

Answer the following questions in a markdown file or at the end of your notebook. Aim for 2-4 sentences per question. Clarity matters more than length.

1. Why do we use the *mean* squared error rather than just the *sum* of squared errors? What would change if we used the sum?

2. Gradient descent updates are of the form $w := w - \eta \cdot \nabla L$. Why do we *subtract* the gradient? What would happen if we added it?

3. You observed that SGD has a noisier loss curve than batch GD. In deep learning, this noise is sometimes considered *beneficial*. Why might noise in the gradient help training? (Hint: think about the loss landscape of a neural network vs. linear regression.)

4. The normal equation computes $(X^T X)^{-1}$. When would this matrix be non-invertible? What does that mean about your features?

5. You applied feature scaling and observed faster convergence. Give an intuitive geometric explanation for why unscaled features slow down gradient descent. (Hint: think about the shape of the loss contours.)

---

## Deliverables

Submit a Jupyter notebook (or Python script + markdown) containing:

1. All code implementations, well-commented.
2. All plots specified above.
3. The written analysis (Part 5).
4. A brief "what I learned" section (3-5 sentences).

**Code quality matters.** Use clear variable names. Add docstrings. Structure your code logically.

---

## Evaluation Criteria

| Component | Points | Criteria |
|-----------|--------|----------|
| Batch GD implementation | 15 | Correct, vectorized, loss tracked |
| SGD implementation | 10 | Correct, properly shuffled |
| Mini-batch implementation | 15 | Correct, configurable batch size, comparison plot |
| Normal equation | 15 | Correct implementation, timing comparison |
| Learning rate analysis | 15 | All learning rates tested, loss landscape visualized |
| Real data application | 15 | Proper pipeline, polynomial features, overfitting analysis |
| Written analysis | 15 | Thoughtful, correct, clear |
| **Total** | **100** | |

**Passing score:** 70/100

---

## Stretch Goals

These are optional but strongly recommended if you want to push yourself:

1. **Implement learning rate scheduling.** Start with a larger learning rate and decay it over time (e.g., $\eta = \eta_0 / (1 + \text{decay} \cdot \text{epoch})$). Show that this combines the speed of a large learning rate with the precision of a small one.

2. **Implement L2 regularization (Ridge Regression).** Add the penalty term $\lambda \|\mathbf{w}\|^2$ to your loss function. Derive the new gradient. Show its effect on polynomial regression overfitting.

3. **Implement momentum.** Instead of $w := w - \eta \cdot \nabla L$, use an exponentially weighted moving average of past gradients. This is the precursor to Adam, the most popular DL optimizer.

4. **Animate the gradient descent trajectory** on the loss landscape contour plot. Use matplotlib's `FuncAnimation`.

5. **Compare your implementation to sklearn's SGDRegressor.** Match the hyperparameters and verify you get the same results.

---

*"If you can implement gradient descent from scratch and explain every line, you understand the engine that powers all of deep learning."*
