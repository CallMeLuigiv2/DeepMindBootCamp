# Module 01: Classical ML Foundations — Comprehensive Notes

**Purpose:** These notes are your reference for the entire module. They are dense by design. Read them before each session, return to them when you get stuck on assignments, and keep them bookmarked as you move into deep learning. Nearly every concept here has a direct analog in neural networks.

---

## Table of Contents

1. [Linear Regression and Gradient Descent](#1-linear-regression-and-gradient-descent)
2. [Logistic Regression and Classification](#2-logistic-regression-and-classification)
3. [Support Vector Machines and Kernel Methods](#3-support-vector-machines-and-kernel-methods)
4. [Trees, Forests, and Boosting](#4-trees-forests-and-boosting)
5. [Unsupervised Learning](#5-unsupervised-learning)
6. [The ML Pipeline](#6-the-ml-pipeline)

---

## 1. Linear Regression and Gradient Descent

### 1.1 The Setup

We have data: n samples, each with d features. Arrange them as:

- **X**: a matrix of shape (n, d). Each row is a sample. Each column is a feature.
- **y**: a vector of shape (n,). The target values we want to predict.

Our model is:

```
y_hat = X @ w + b
```

Where `w` is a weight vector of shape (d,) and `b` is a scalar bias. This is the hypothesis — a linear function of the features.

**Intuition:** We are assuming the target is (approximately) a weighted sum of the features plus some constant offset. The job is to find the weights.

### 1.2 The Loss Function: Mean Squared Error

We need to measure how wrong our predictions are. The standard choice is Mean Squared Error:

```
L(w, b) = (1/n) * sum_{i=1}^{n} (y_hat_i - y_i)^2
         = (1/n) * ||X @ w + b - y||^2
```

**What each term means:**
- `y_hat_i - y_i`: the error (residual) for sample i
- `(...)^2`: squaring penalizes large errors more than small ones
- `(1/n) * sum(...)`: averaging over all samples

**Why squared error?**
1. It is differentiable everywhere (unlike absolute error).
2. It has a unique global minimum (the loss function is convex).
3. It corresponds to maximum likelihood estimation under Gaussian noise assumptions.
4. The gradient is simple and proportional to the residual.

**Why not absolute error?** Absolute error `|y_hat - y|` is not differentiable at zero. Its gradient is either +1 or -1 regardless of how far off the prediction is. Squared error gives a gradient proportional to the error magnitude, which provides a natural "step size" signal.

### 1.3 The Loss Landscape

For linear regression with MSE, the loss is a quadratic function of the parameters. In 2D (one weight + bias), it looks like a bowl:

```
Loss
 ^
 |      .  .  .
 |    .        .
 |   .    *     .       * = minimum
 |    .        .
 |      .  .  .
 +-------------------> w

    Contour Plot (top view):

         b ^
           |    (  (  ( * )  )  )
           |
           +--------------------> w

    Elliptical contours.
    The minimum (*) is where the gradient is zero.
```

**Key property:** For linear regression with MSE, the loss landscape is convex. There is exactly one minimum. Gradient descent is guaranteed to find it (with an appropriate learning rate). This is NOT true for neural networks, which have highly non-convex loss landscapes.

### 1.4 Gradient Descent — The Core Algorithm

This is the single most important algorithm in this entire course. Every neural network you will ever train uses a variant of this.

**The idea:** We want to find the parameters (w, b) that minimize the loss. We cannot see the entire loss landscape at once. But we can compute the gradient (the direction of steepest increase) at our current location. If we take a small step in the *opposite* direction, we reduce the loss.

**The update rule:**

```
w := w - lr * dL/dw
b := b - lr * dL/db
```

Where `lr` is the learning rate (step size) and `dL/dw`, `dL/db` are the partial derivatives of the loss with respect to the parameters.

**Deriving the gradient for MSE:**

```
L = (1/n) * sum (X @ w + b - y)^2

Let e = X @ w + b - y    (the residual vector, shape (n,))

dL/dw = (2/n) * X^T @ e = (2/n) * X^T @ (X @ w + b - y)
dL/db = (2/n) * sum(e)   = (2/n) * 1^T @ (X @ w + b - y)
```

**What this means, term by term:**
- `e = X @ w + b - y`: how wrong we are for each sample
- `X^T @ e`: each feature's "contribution" to the total error, summed over samples
- `(2/n)`: normalization and the derivative of the square

**In practice**, the factor of 2 is often absorbed into the learning rate, so you may see implementations without it.

### 1.5 Gradient Descent Variants

#### Batch Gradient Descent

Uses ALL training samples to compute each gradient update.

```python
for epoch in range(n_iterations):
    y_hat = X @ w + b
    error = y_hat - y
    dw = (2 / n) * X.T @ error
    db = (2 / n) * np.sum(error)
    w = w - lr * dw
    b = b - lr * db
```

**Pros:** Stable convergence. Exact gradient.
**Cons:** Slow for large datasets (must process all n samples per update). Expensive when n is millions.

#### Stochastic Gradient Descent (SGD)

Uses ONE randomly chosen sample per update.

```python
for epoch in range(n_iterations):
    indices = np.random.permutation(n)
    for i in indices:
        xi = X[i:i+1]        # shape (1, d)
        yi = y[i:i+1]        # shape (1,)
        y_hat = xi @ w + b
        error = y_hat - yi
        dw = 2 * xi.T @ error
        db = 2 * np.sum(error)
        w = w - lr * dw
        b = b - lr * db
```

**Pros:** Much faster per update. Can escape shallow local minima (due to noise). Can start improving before seeing all data.
**Cons:** Noisy gradient estimates. May not converge to the exact minimum (oscillates around it). Requires careful learning rate tuning or a learning rate schedule.

**The noise is a feature, not a bug.** In deep learning, the noise from SGD acts as implicit regularization, helping models generalize better. This is one of the most important insights in modern ML.

#### Mini-Batch Gradient Descent

The practical compromise. Uses a batch of B samples (typically 32, 64, 128, or 256).

```python
for epoch in range(n_iterations):
    indices = np.random.permutation(n)
    for start in range(0, n, batch_size):
        batch_idx = indices[start:start + batch_size]
        X_batch = X[batch_idx]
        y_batch = y[batch_idx]
        y_hat = X_batch @ w + b
        error = y_hat - y_batch
        dw = (2 / len(batch_idx)) * X_batch.T @ error
        db = (2 / len(batch_idx)) * np.sum(error)
        w = w - lr * dw
        b = b - lr * db
```

**Pros:** Balances gradient quality with computational efficiency. Leverages vectorized operations (GPU parallelism). Standard practice in deep learning.
**Cons:** Adds batch_size as a hyperparameter.

**Rule of thumb:** In deep learning, mini-batch SGD with batch size 32-256 is the default starting point.

### 1.6 Learning Rate: The Most Important Hyperparameter

```
Learning rate effects:

Too small (lr = 0.0001):              Just right (lr = 0.01):
Loss                                   Loss
 |\.                                    |\
 | \.                                   | \
 |  \.                                  |  \.
 |   \.                                 |   '---___
 |    '\.                               |          '----____
 |      '\.                             +-------------------> Epoch
 |        '\.
 +--------'------> Epoch
 (Still decreasing at epoch 1000)       (Converged by epoch 200)

Too large (lr = 1.0):                  Way too large (lr = 10.0):
Loss                                   Loss
 |    /\                                |          /
 |   /  \   /\                          |         /
 |  /    \ /  \   /                     |        /
 | /      V    \ /                      |       /
 |/             V                       |      /
 +-------------------> Epoch            +---/-----------> Epoch
 (Oscillating)                          (Diverging)
```

**How to choose a learning rate:**
1. Start with 0.01 or 0.001.
2. If the loss is not decreasing, try a larger learning rate.
3. If the loss is oscillating or increasing, try a smaller learning rate.
4. In deep learning, learning rate finders (gradually increase lr, plot loss) automate this.

### 1.7 The Closed-Form Solution: The Normal Equation

For linear regression (and only linear regression), we can solve for the optimal weights directly:

```
w* = (X^T X)^{-1} X^T y
```

**Derivation sketch:**
1. Write L(w) = (1/n) ||Xw - y||^2 (absorbing bias into X by adding a column of ones).
2. Take the gradient: dL/dw = (2/n) X^T (Xw - y).
3. Set it to zero: X^T Xw = X^T y.
4. Solve: w = (X^T X)^{-1} X^T y.

**When to use the normal equation vs. gradient descent:**

| Property | Normal Equation | Gradient Descent |
|----------|----------------|-----------------|
| Compute time | O(d^3) for matrix inversion | O(n * d * k) for k iterations |
| Better when... | d is small (< ~10,000) | d is large or n is very large |
| Needs learning rate? | No | Yes |
| Iterative? | No (one-shot) | Yes |
| Generalizes to other models? | No | Yes (this is the key advantage) |

**The critical insight:** The normal equation works ONLY for linear regression with MSE. Gradient descent works for any differentiable loss function with any differentiable model. This is why gradient descent is the universal optimization algorithm in deep learning.

### 1.8 Feature Scaling

**The problem:** If feature 1 ranges from 0 to 1 and feature 2 ranges from 0 to 1,000,000, the loss landscape becomes elongated:

```
Without scaling:               With scaling:
b ^                            b ^
  |  /  /  /  /  /               |   (  (  * )  )
  | /  /  /  /  /                |
  |/  /  /  /  /                 +--------------------> w
  +--------------------> w

  Elongated ellipses.            Circular contours.
  GD zig-zags slowly.           GD goes straight to minimum.
```

**Why this happens:** The gradient in the direction of the large-scale feature is much larger, causing the optimizer to overshoot in that direction while barely moving in the small-scale direction.

**Standardization (z-score normalization):**
```python
X_scaled = (X - X.mean(axis=0)) / X.std(axis=0)
```
Each feature gets mean 0 and standard deviation 1.

**Min-max normalization:**
```python
X_scaled = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
```
Each feature gets range [0, 1].

**Critical rule:** Fit the scaler on the training data only. Transform both training and test data with the training statistics. If you fit on the test data, you have data leakage.

### 1.9 Polynomial Features

Linear regression assumes `y = w1*x1 + w2*x2 + b`. But what if the true relationship is nonlinear?

**The trick:** Create new features that are powers and interactions of existing features.

For a single feature x, degree 2 polynomial features are: [x, x^2]
For two features (x1, x2), degree 2: [x1, x2, x1^2, x1*x2, x2^2]

The model is still *linear in its parameters* — it is w1*x + w2*x^2 + b, which is linear in (w1, w2, b). But it can model quadratic relationships in the original feature.

```python
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)
```

**The overfitting danger:** High-degree polynomial features (degree 5, 10, etc.) create a very flexible model that can fit the training data perfectly but generalizes poorly. This is your first encounter with the bias-variance tradeoff.

---

## 2. Logistic Regression and Classification

### 2.1 Why Not Linear Regression for Classification?

If we try to predict class labels (0 or 1) with linear regression:
- Predictions can be negative or greater than 1.
- There is no probabilistic interpretation.
- The decision boundary works but the confidence estimates are meaningless.

We need a function that maps any real number to the range (0, 1). Enter the sigmoid.

### 2.2 The Sigmoid Function

```
sigma(z) = 1 / (1 + exp(-z))
```

**Properties:**
- Output is always in (0, 1) — valid probability.
- sigma(0) = 0.5 (50/50 when the linear score is zero).
- As z goes to positive infinity, sigma(z) goes to 1.
- As z goes to negative infinity, sigma(z) goes to 0.
- The derivative is: sigma'(z) = sigma(z) * (1 - sigma(z)).

```
Sigmoid function:

1.0  |                          ___-------
     |                    ___---
     |                 _--
0.5  |- - - - - - - -/- - - - - - - - - -
     |             _-
     |          _--
     |    ___---
0.0  |----
     +----+----+----+----+----+----+-----> z
         -3   -2   -1    0    1    2    3
```

**Logistic regression model:**
```
P(y=1 | x) = sigma(x @ w + b) = 1 / (1 + exp(-(x @ w + b)))
```

The linear combination x @ w + b produces a "score" (called the logit). The sigmoid converts it to a probability.

### 2.3 Cross-Entropy Loss — Derived from Maximum Likelihood

This derivation is important. It shows that cross-entropy is not an arbitrary choice — it follows inevitably from assuming our model outputs Bernoulli probabilities.

**Step 1: Single sample likelihood.**
If y is 0 or 1 and our model predicts probability p = sigma(x @ w + b):

```
P(y | x; w, b) = p^y * (1-p)^(1-y)
```

Check: if y=1, this is p. If y=0, this is (1-p). Correct.

**Step 2: Full dataset likelihood.**
Assuming samples are independent:

```
L(w, b) = product_{i=1}^{n} p_i^{y_i} * (1-p_i)^{1-y_i}
```

**Step 3: Log-likelihood.**
Products are numerically unstable and hard to differentiate. Take the log:

```
log L(w, b) = sum_{i=1}^{n} [ y_i * log(p_i) + (1-y_i) * log(1-p_i) ]
```

**Step 4: Negate and normalize.**
We want to *minimize* a loss (not maximize a likelihood), so negate:

```
BCE(w, b) = -(1/n) * sum_{i=1}^{n} [ y_i * log(p_i) + (1-y_i) * log(1-p_i) ]
```

This is **binary cross-entropy**. Every term is justified by probability theory.

**What each term means:**
- `y_i * log(p_i)`: when the true label is 1, we want p_i to be high (close to 1), so log(p_i) is close to 0 (small loss).
- `(1-y_i) * log(1-p_i)`: when the true label is 0, we want p_i to be low (close to 0), so log(1-p_i) is close to 0 (small loss).
- The negative sign makes this positive.
- `(1/n)` averages over samples.

### 2.4 Why Cross-Entropy Instead of MSE for Classification?

**The gradient argument.** Consider a sample with true label y=1 and predicted probability p close to 0 (very wrong prediction).

MSE gradient: `dL/dp = 2(p - 1)`. This is around -2. A fixed-magnitude signal.

Cross-entropy gradient: `dL/dp = -1/p`. When p = 0.01, this is -100. When p = 0.001, this is -1000. The gradient is *enormous* when the prediction is very wrong.

Cross-entropy provides a strong corrective signal for confident wrong predictions. MSE does not. This means cross-entropy trains faster and avoids the "vanishing gradient" problem for classification.

**This same reasoning applies in deep learning.** It is why the output layer of every classification network uses cross-entropy, not MSE.

### 2.5 The Gradient of Logistic Regression

This is one of the cleanest results in ML:

```
dL/dw = (1/n) * X^T @ (sigma(X @ w + b) - y)
dL/db = (1/n) * sum(sigma(X @ w + b) - y)
```

**What this means:** The gradient is proportional to X transposed times the prediction error. The same structure as linear regression. This is not a coincidence — both are generalized linear models.

```python
def logistic_gradient(X, y, w, b):
    n = len(y)
    y_hat = sigmoid(X @ w + b)
    error = y_hat - y                    # prediction errors
    dw = (1/n) * X.T @ error            # shape (d,)
    db = (1/n) * np.sum(error)           # scalar
    return dw, db
```

### 2.6 Decision Boundaries

The decision boundary is the set of points where P(y=1|x) = 0.5, which is where the logit is zero:

```
x @ w + b = 0
```

For 2D features (x1, x2):
```
w1*x1 + w2*x2 + b = 0
x2 = -(w1*x1 + b) / w2
```

This is a straight line. Logistic regression can only produce linear decision boundaries.

**Nonlinear boundaries:** Adding polynomial features to the input allows logistic regression to produce curved boundaries. With degree-2 features, you get quadratic boundaries (ellipses, parabolas, hyperbolas). A neural network learns these nonlinear transformations automatically through its hidden layers.

### 2.7 Evaluation Metrics for Classification

#### The Confusion Matrix

```
                    Predicted
                  Neg      Pos
Actual  Neg  [   TN   |   FP   ]
        Pos  [   FN   |   TP   ]

TN = True Negative:  correctly predicted negative
FP = False Positive: incorrectly predicted positive (Type I error)
FN = False Negative: incorrectly predicted negative (Type II error)
TP = True Positive:  correctly predicted positive
```

Every classification metric is derived from these four numbers.

#### Accuracy

```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

**The problem:** On a dataset with 95% negative and 5% positive, a model that *always* predicts negative has 95% accuracy. Useless but high accuracy.

**Rule:** Never report accuracy alone on an imbalanced dataset.

#### Precision

```
Precision = TP / (TP + FP)
```

**Plain English:** "Of all the things I said were positive, what fraction actually were?"

**When it matters:** When the cost of a false positive is high. Spam filtering: if you move a legitimate email to spam (FP), the user misses it. You want high precision.

#### Recall (Sensitivity, True Positive Rate)

```
Recall = TP / (TP + FN)
```

**Plain English:** "Of all the things that were actually positive, what fraction did I catch?"

**When it matters:** When the cost of a false negative is high. Cancer screening: if you miss a cancer case (FN), the patient does not get treatment. You want high recall.

#### F1-Score

```
F1 = 2 * (Precision * Recall) / (Precision + Recall)
```

The harmonic mean of precision and recall. The harmonic mean is used (rather than the arithmetic mean) because it penalizes imbalance: if either precision or recall is very low, F1 is pulled down sharply.

**Example:**
- Precision = 0.95, Recall = 0.95: F1 = 0.95
- Precision = 0.99, Recall = 0.01: F1 = 0.0198 (terrible, as it should be)

#### ROC Curve and AUC

The ROC (Receiver Operating Characteristic) curve plots TPR vs. FPR at every possible classification threshold.

```
TPR (Recall)
 ^
 |        ___-------
 |      _/
 |    _/
 |   /           Your model
 |  /
 | /     ____---
 |/  ___/       Random classifier (diagonal)
 +--/-----------------------> FPR
 0                          1
```

**How to read it:**
- The diagonal line (TPR = FPR) is a random classifier. AUC = 0.5.
- A perfect classifier goes straight up to (0, 1) then across. AUC = 1.0.
- The further above the diagonal, the better.
- AUC (Area Under the Curve) summarizes the curve in a single number.

**AUC interpretation:** The probability that the model ranks a randomly chosen positive sample higher than a randomly chosen negative sample.

**AUC = 0.0:** This means the model has perfectly *reversed* the labels. Flip its predictions and you get a perfect classifier. So AUC = 0.0 is not worse than 0.5 — it is actually exploitable.

**Computing ROC:**
1. Get predicted probabilities for all samples.
2. Sort by probability (descending).
3. For each possible threshold, compute TPR and FPR.
4. Plot and compute area under the curve (trapezoidal rule).

```python
def compute_roc(y_true, y_scores):
    # Sort by score descending
    sorted_indices = np.argsort(-y_scores)
    y_sorted = y_true[sorted_indices]
    scores_sorted = y_scores[sorted_indices]

    # Get unique thresholds
    thresholds = np.unique(scores_sorted)[::-1]

    tprs, fprs = [0], [0]
    P = np.sum(y_true == 1)  # total positives
    N = np.sum(y_true == 0)  # total negatives

    for thresh in thresholds:
        y_pred = (y_scores >= thresh).astype(int)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        tprs.append(tp / P)
        fprs.append(fp / N)

    return np.array(fprs), np.array(tprs), thresholds
```

### 2.8 Multi-Class Classification: Softmax

For K classes, we want to output K probabilities that sum to 1.

**The softmax function:**
```
softmax(z_k) = exp(z_k) / sum_{j=1}^{K} exp(z_j)
```

Where z is a vector of K "logits" (raw scores, one per class).

**What it does:** Converts arbitrary real-valued scores into a valid probability distribution. Larger logits get larger probabilities. The exponential amplifies differences.

**Numerical stability:** Before computing, subtract the max logit:
```python
def softmax(z):
    z_shifted = z - np.max(z, axis=-1, keepdims=True)
    exp_z = np.exp(z_shifted)
    return exp_z / np.sum(exp_z, axis=-1, keepdims=True)
```

This prevents overflow from exp(large number) and does not change the result (it cancels out).

**Connection to sigmoid:** For 2 classes, softmax reduces to the sigmoid function. Proof: Let the logits be [z, 0]. Then softmax gives [exp(z)/(exp(z)+1), 1/(exp(z)+1)] = [sigmoid(z), 1-sigmoid(z)].

**Categorical cross-entropy** (the multi-class loss):
```
L = -(1/n) * sum_{i=1}^{n} sum_{k=1}^{K} y_{ik} * log(p_{ik})
```

Where y is one-hot encoded. Only the term for the true class survives (since all other y_{ik} are 0), so this simplifies to:

```
L = -(1/n) * sum_{i=1}^{n} log(p_{i, true_class_i})
```

**Plain English:** For each sample, look at the predicted probability for the correct class. Take its log. Negate and average. We want the model to assign high probability to the correct class.

---

## 3. Support Vector Machines and Kernel Methods

### 3.1 The Maximum Margin Principle

Many hyperplanes can separate linearly separable data. SVMs choose the one with the maximum *margin* — the largest gap between the hyperplane and the nearest data points from each class.

```
       |  <--- margin --->  |
       |                    |
   o   |   o                |   x   x
  o o  |     o   o          |     x
   o   | o     o            |   x   x
       |                    |
       |    decision        |
       |    boundary        |
```

**Why maximum margin?** Statistical learning theory (Vapnik) shows that larger margins lead to better generalization. Intuitively: a decision boundary with a large margin is robust to small perturbations in the data.

### 3.2 Support Vectors

The data points that lie exactly on the margin boundary are called **support vectors**. They are the critical points — if you removed all other data points, the decision boundary would not change.

This makes SVMs robust to outliers (points far from the boundary do not matter) and memory-efficient (only support vectors need to be stored for prediction).

### 3.3 The Optimization Problem

The SVM finds the hyperplane w^T x + b = 0 that maximizes the margin 2/||w||, subject to all points being correctly classified with margin at least 1.

**Primal formulation:**
```
minimize    (1/2) * ||w||^2
subject to  y_i * (w^T x_i + b) >= 1  for all i
```

**What this means:**
- `(1/2) * ||w||^2`: minimizing the weight magnitude maximizes the margin (margin = 2/||w||).
- `y_i * (w^T x_i + b) >= 1`: every sample must be on the correct side of the boundary, with a margin of at least 1.

This is a convex quadratic program with linear constraints. It has a unique global solution.

### 3.4 The Kernel Trick

**The problem:** Real data is rarely linearly separable. We could map the data to a higher-dimensional space where it becomes separable. For example, 2D points on concentric circles become separable in 3D by adding a "radius" feature (x1^2 + x2^2).

**The computational problem:** Mapping to high-dimensional space and computing dot products there is expensive. For some mappings, the feature space is infinite-dimensional.

**The kernel trick:** The SVM optimization and prediction depend on the data ONLY through pairwise dot products x_i^T x_j. If we have a function K(x_i, x_j) that computes the dot product in the high-dimensional feature space WITHOUT explicitly computing the mapping, we get the power of the high-dimensional representation at the cost of a simple function evaluation.

```
K(x_i, x_j) = phi(x_i)^T phi(x_j)
```

We never compute phi(x). We only compute K(x_i, x_j).

**Common kernels:**

1. **Linear:** K(x, y) = x^T y. No transformation.

2. **Polynomial:** K(x, y) = (gamma * x^T y + coef0)^degree. Computes dot products in the space of all polynomial feature combinations up to the given degree.

3. **RBF (Radial Basis Function):** K(x, y) = exp(-gamma * ||x - y||^2). This is the most commonly used kernel. It corresponds to an *infinite-dimensional* feature space. The bandwidth parameter gamma controls the "reach" of each training point: small gamma means wide influence, large gamma means narrow influence.

**Why the kernel trick is genius:** It lets you compute in an infinite-dimensional feature space in finite time. The computational cost depends on the number of training samples, not the dimensionality of the feature space.

### 3.5 Soft Margin: The C Parameter

Real data has noise and overlap between classes. A hard-margin SVM (requiring perfect classification) either:
- Cannot find a solution (data not separable), or
- Finds a very narrow margin that overfits to noise.

The soft-margin SVM introduces slack variables, allowing some misclassifications:

```
minimize    (1/2) * ||w||^2 + C * sum(xi_i)
subject to  y_i * (w^T x_i + b) >= 1 - xi_i
            xi_i >= 0
```

**The C parameter controls the tradeoff:**
- Large C: penalize misclassifications heavily. Small margin, low bias, high variance.
- Small C: tolerate more misclassifications. Large margin, high bias, low variance.

```
Small C (underfitting):          Large C (overfitting):
    o  o                              o  o
  o  |  o  x  x                    o   |   o  x  x
 o   |  o  x                      o   |   o|  x
  o  |  x  x  x                    o /   o/  x  x
                                     curved, tight boundary
  Wide margin, some errors.       Narrow margin, few errors.
```

### 3.6 When to Use SVMs

**SVMs shine when:**
- Dataset is small to medium (up to ~100K samples).
- Features are well-engineered (SVMs do not learn features like neural nets).
- The decision boundary has a clear margin (e.g., text classification).
- You need a strong model without much tuning.

**SVMs struggle when:**
- Dataset is very large (kernel SVM training is O(n^2) to O(n^3)).
- Data is high-dimensional and sparse (though linear SVM is fine here).
- You need to output calibrated probabilities (SVMs naturally output distances, not probabilities).
- Feature learning is needed (images, raw text).

---

## 4. Trees, Forests, and Boosting

### 4.1 Decision Trees

A decision tree recursively partitions the feature space into rectangular regions, assigning a prediction to each region.

```
              [Feature 2 < 3.5?]
              /                \
           Yes                  No
           /                      \
  [Feature 1 < 2.0?]          [Class: B]
   /              \
 Yes               No
  |                 |
[Class: A]     [Class: B]
```

Each internal node tests a feature against a threshold. Each leaf assigns a class (classification) or a value (regression).

### 4.2 Splitting Criteria

At each node, the tree tries every feature and every threshold, choosing the split that best separates the classes.

#### Gini Impurity

```
Gini(S) = 1 - sum_{k=1}^{K} p_k^2
```

Where p_k is the proportion of samples belonging to class k.

**Intuition:** The probability that two randomly chosen samples from the set have different classes.

- Gini = 0: pure node (all one class).
- Gini = 0.5: maximum impurity (binary, 50/50 split).

#### Entropy and Information Gain

```
Entropy(S) = -sum_{k=1}^{K} p_k * log2(p_k)
```

**Intuition:** The expected number of bits needed to encode the class of a random sample.

```
Information Gain = Entropy(parent) - weighted_average(Entropy(children))
```

The split that maximizes information gain removes the most uncertainty about the class.

**In practice,** Gini and entropy usually produce very similar trees. Gini is slightly faster (no logarithm), so it is the default in most implementations.

```python
def gini(y):
    classes, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    return 1 - np.sum(probs ** 2)

def entropy(y):
    classes, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    return -np.sum(probs * np.log2(probs + 1e-10))

def information_gain(y_parent, y_left, y_right):
    n = len(y_parent)
    weight_left = len(y_left) / n
    weight_right = len(y_right) / n
    return entropy(y_parent) - (weight_left * entropy(y_left)
                                + weight_right * entropy(y_right))
```

### 4.3 The Bias-Variance Tradeoff — Deeply

This is the most important conceptual framework in all of machine learning. It governs your choice of model, hyperparameters, and regularization.

**The decomposition:** For any model, the expected prediction error can be decomposed as:

```
Expected Error = Bias^2 + Variance + Irreducible Noise
```

**What each term means:**

- **Bias:** Error from wrong assumptions in the model. A linear model has high bias if the true relationship is quadratic — it *cannot* capture the pattern no matter how much data you give it. This is underfitting.

- **Variance:** Error from sensitivity to the specific training set. A model with high variance gives very different predictions depending on which training samples it saw. A deep decision tree memorizes the training set and has high variance. This is overfitting.

- **Irreducible noise:** Error from inherent randomness in the data. No model can eliminate this.

```
                    Error
                      ^
                      |   \          _____/
                      |    \       /
                      |     \    /
                      |      \_/
    Total Error ----> |     . . . . . . . .
                      |    \    /
    Variance -------> |     \_/
                      |      _________
    Bias^2 ---------> |    /
                      |   /
                      +--/---+---+---+----> Model Complexity
                         |       |
                      Simple   Complex
                     (underfit)(overfit)
                              ^
                         Sweet spot
```

**How this maps to specific models:**

| Model | Bias | Variance |
|-------|------|----------|
| Linear regression (few features) | High | Low |
| Polynomial regression (high degree) | Low | High |
| Shallow decision tree | High | Low |
| Deep decision tree | Low | High |
| Random Forest | Low | Low (this is the magic) |
| Gradient Boosting (many rounds) | Low | Medium-High |

### 4.4 Random Forests: Reducing Variance Through Averaging

**The insight:** A single deep tree has low bias but high variance. If we average many trees, the variance decreases — but only if the trees are different from each other. Two perfectly correlated trees, when averaged, give the same variance as one tree.

Random forests create diversity through two mechanisms:

1. **Bagging (Bootstrap Aggregating):** Each tree is trained on a random bootstrap sample (sample with replacement) from the training data. Roughly 63% of samples are selected for each tree; the rest are "out-of-bag."

2. **Feature randomness:** At each split, the tree considers only a random subset of features (typically sqrt(d) for classification, d/3 for regression). This decorrelates the trees.

**The math of why averaging works:**
If you have B independent estimators, each with variance sigma^2, the variance of their average is sigma^2/B. Of course, the trees are not fully independent (they are trained on overlapping data), but the feature randomness helps decorrelate them.

```python
# Simplified Random Forest
class SimpleRandomForest:
    def __init__(self, n_trees=100, max_features='sqrt'):
        self.n_trees = n_trees
        self.max_features = max_features
        self.trees = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        if self.max_features == 'sqrt':
            max_feat = int(np.sqrt(n_features))

        for _ in range(self.n_trees):
            # Bootstrap sample
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            X_boot, y_boot = X[indices], y[indices]

            # Train a tree (with feature randomness handled inside the tree)
            tree = DecisionTree(max_features=max_feat)
            tree.fit(X_boot, y_boot)
            self.trees.append(tree)

    def predict(self, X):
        # Majority vote
        predictions = np.array([tree.predict(X) for tree in self.trees])
        return stats.mode(predictions, axis=0)[0].flatten()
```

**Out-of-bag (OOB) error:** Each tree was trained on about 63% of the data. The remaining 37% can be used as a validation set for that tree. Averaging these per-tree validation errors gives the OOB error, a free estimate of generalization performance without needing a separate validation set.

### 4.5 Gradient Boosting: Reducing Bias Through Sequential Learning

While random forests reduce variance by averaging, boosting reduces bias by *sequentially* correcting errors.

**The core idea:** Train a weak learner (shallow tree). Look at the errors it makes. Train the next weak learner to predict those errors. Add its predictions (scaled by a learning rate) to the ensemble. Repeat.

**Gradient boosting as gradient descent in function space:**
Instead of optimizing parameters, we are optimizing a *function* (the ensemble prediction). At each step, we compute the "gradient" of the loss with respect to the current predictions, and fit a tree to approximate that gradient.

For regression with MSE:
```
Step 1: Fit initial prediction F_0(x) = mean(y)
Step 2: Compute residuals r_i = y_i - F_0(x_i)
Step 3: Fit tree h_1 to residuals r
Step 4: Update: F_1(x) = F_0(x) + lr * h_1(x)
Step 5: Compute new residuals r_i = y_i - F_1(x_i)
Step 6: Repeat...
```

The learning rate (shrinkage) controls how much each tree contributes. Smaller learning rate requires more trees but typically gives better generalization.

```
       Boosting builds up the prediction gradually:

       F(x) = F_0 + lr*h_1 + lr*h_2 + ... + lr*h_T
                |       |         |              |
              initial  corrects corrects      corrects
              guess    first    second        remaining
                       errors   errors        errors
```

**Key hyperparameters for gradient boosting:**
- `n_estimators`: number of trees. More trees = lower bias, but eventually overfits.
- `learning_rate`: shrinkage factor. Smaller = need more trees, but better generalization.
- `max_depth`: depth of each tree. Usually 3-8. Shallow trees = high bias each, but boosting fixes this.
- `subsample`: fraction of data used per tree (adds randomness, like bagging).
- `colsample_bytree` (XGBoost): fraction of features per tree (like RF's feature randomness).

**XGBoost innovations:**
1. Regularization in the objective function (L1 and L2 on leaf weights).
2. Second-order gradient approximation (Newton's method instead of just gradient).
3. Histogram-based splitting for speed.
4. Built-in handling of missing values.
5. Column subsampling (borrowing from random forests).

### 4.6 Bagging vs. Boosting: Summary

```
Bagging (Random Forest):           Boosting (Gradient Boosting):

  Tree1  Tree2  Tree3  ...           Tree1 -> Tree2 -> Tree3 -> ...
    |      |      |                    |        |        |
    v      v      v                    v        v        v
  [   Average / Vote   ]            [  Sequential Addition  ]

  Reduces VARIANCE                   Reduces BIAS
  Trees are independent              Trees depend on each other
  Parallelizable                     Sequential (harder to parallelize)
  Robust to overfitting              Can overfit if not regularized
  Less tuning needed                 More hyperparameters to tune
```

---

## 5. Unsupervised Learning

### 5.1 K-Means Clustering

**The algorithm:**
1. Initialize K cluster centers (randomly or with K-Means++).
2. Assign each point to the nearest center.
3. Recompute each center as the mean of its assigned points.
4. Repeat steps 2-3 until convergence.

```python
def kmeans(X, K, max_iters=100):
    n, d = X.shape
    # Initialize: random selection from data points
    centers = X[np.random.choice(n, K, replace=False)]

    for _ in range(max_iters):
        # Assign points to nearest center
        distances = np.linalg.norm(X[:, None] - centers[None, :], axis=2)
        labels = np.argmin(distances, axis=1)

        # Update centers
        new_centers = np.array([X[labels == k].mean(axis=0) for k in range(K)])

        if np.allclose(centers, new_centers):
            break
        centers = new_centers

    return labels, centers
```

**Objective function:**
```
J = sum_{i=1}^{n} ||x_i - mu_{c_i}||^2
```
Where c_i is the cluster assignment of point i and mu_{c_i} is the center of that cluster.

K-Means minimizes the within-cluster sum of squares. This is a non-convex optimization — the algorithm is not guaranteed to find the global minimum. Run it multiple times with different initializations and keep the best.

**K-Means++ initialization:** Instead of random initialization, K-Means++ selects initial centers that are spread out:
1. Choose first center randomly from data points.
2. For each remaining center: choose a point with probability proportional to its squared distance from the nearest existing center.

This gives provably better results on average.

**Choosing K:**
- **Elbow method:** Plot within-cluster sum of squares vs. K. Look for the "elbow" where adding more clusters stops helping much.
- **Silhouette score:** Measures how similar each point is to its own cluster vs. neighboring clusters. Ranges from -1 to 1. Higher is better.

**Failure modes:**
- Non-convex clusters (K-Means assumes roughly spherical clusters).
- Clusters of very different sizes or densities.
- High-dimensional data (distances become less meaningful — the "curse of dimensionality").

### 5.2 Principal Component Analysis (PCA)

PCA finds the directions of maximum variance in the data and projects onto them. It is the most important unsupervised method and the foundation of dimensionality reduction.

**The variance maximization formulation:**

We want to find a direction (unit vector u) that maximizes the variance of the projected data:

```
Var(X @ u) = u^T Sigma u
```

Where Sigma = (1/n) X^T X is the covariance matrix (assuming zero-mean data).

Subject to ||u|| = 1, this is maximized when u is the eigenvector of Sigma corresponding to the largest eigenvalue.

**The eigenvalue decomposition:**
```
Sigma = V Lambda V^T
```

Where:
- V is the matrix of eigenvectors (each column is a principal component direction).
- Lambda is a diagonal matrix of eigenvalues (each one is the variance explained by that component).
- The eigenvectors are orthogonal. They form a new coordinate system aligned with the data's variance.

```
Original data:              After PCA:
   ^                          ^  PC1 (most variance)
   |  . .  . . .              | /
   | . . . . . .              |/. . . . . . .
   |. . . . . .               +----------------> PC2
   +----------->              Rotated so PC1 captures max variance
```

**Implementation:**

```python
def pca(X, n_components):
    # Center the data
    X_centered = X - X.mean(axis=0)

    # Compute covariance matrix
    cov = (1 / len(X)) * X_centered.T @ X_centered

    # Eigenvalue decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Sort by eigenvalue (descending)
    idx = np.argsort(-eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Select top components
    components = eigenvectors[:, :n_components]

    # Project data
    X_projected = X_centered @ components

    # Explained variance ratio
    explained_var = eigenvalues[:n_components] / np.sum(eigenvalues)

    return X_projected, components, explained_var
```

**Choosing the number of components:**
Plot cumulative explained variance ratio vs. number of components. Choose enough components to explain 90-95% of the variance.

```
Cumulative Explained Variance
1.0 |                    ___-------
    |                ___-
0.9 |------ - - - _-  - - - - - - -  <-- 90% threshold
    |           _-
0.8 |        _-
    |      _-
    |    _-
    |  _-
    +-/---+---+---+---+---+---> # Components
      1   2   3   4   5   6
          ^
     Choose 2-3 components
```

**PCA applications:**
1. **Visualization:** Project to 2D or 3D for plotting.
2. **Noise reduction:** Discard low-variance components (which are often noise).
3. **Preprocessing:** Reduce dimensionality before training a model (speeds up training, can reduce overfitting).
4. **Feature understanding:** Examine which original features load onto each principal component.

### 5.3 t-SNE: Nonlinear Visualization

PCA preserves global structure (large pairwise distances). t-SNE preserves *local* structure (small pairwise distances, i.e., neighborhoods).

**Intuition:**
1. In the high-dimensional space, compute a probability distribution over pairs of points: nearby points have high probability, distant points have low probability (using a Gaussian kernel).
2. In the low-dimensional embedding, compute a similar distribution (using a Student's t-distribution — hence the name).
3. Minimize the KL divergence between the two distributions. This forces the embedding to preserve the neighborhood structure.

**The t-distribution** in the embedding space (instead of Gaussian) has heavier tails, which helps avoid the "crowding problem" — in low dimensions, there is less room to place points, so distant clusters would be forced together without the heavy tails.

**Key parameter: perplexity.** Roughly controls the effective number of neighbors for each point. Typical values: 5-50. Too low: very local structure only. Too high: washes out local structure.

**Critical pitfalls of t-SNE:**
1. **Non-deterministic:** Different random seeds give different embeddings.
2. **Non-parametric:** You cannot embed new points without re-running the algorithm.
3. **Do not interpret cluster sizes:** t-SNE can expand or compress clusters arbitrarily.
4. **Do not interpret distances between clusters:** The relative distances between clusters are not meaningful.
5. **Use only for visualization, never for downstream ML.**

### 5.4 Connection to Deep Learning: Autoencoders

PCA is *linear* dimensionality reduction. An **autoencoder** is a neural network that learns a nonlinear encoding:

```
Input --> [Encoder] --> Bottleneck --> [Decoder] --> Reconstruction
  x          f(x)         z             g(z)          x_hat
                     (low-dim)

Loss = ||x - x_hat||^2

The bottleneck forces the network to learn a compressed representation.
```

**Key insight:** If the encoder and decoder are both single linear layers (with no activation function), and the loss is MSE, the autoencoder learns the same subspace as PCA. Neural networks generalize PCA to nonlinear manifolds.

This connection makes PCA conceptually important: it is the linear special case of the most general representation learning framework.

---

## 6. The ML Pipeline

### 6.1 Cross-Validation

**The problem with a single train/test split:** You get a single estimate of model performance. This estimate has high variance — a different split might give a very different number. You cannot trust it.

**K-fold cross-validation:**
1. Split data into K roughly equal folds.
2. For each fold: train on the other K-1 folds, evaluate on this fold.
3. Average the K evaluation scores. Also report the standard deviation.

```
Fold 1: [TEST] [Train] [Train] [Train] [Train]
Fold 2: [Train] [TEST] [Train] [Train] [Train]
Fold 3: [Train] [Train] [TEST] [Train] [Train]
Fold 4: [Train] [Train] [Train] [TEST] [Train]
Fold 5: [Train] [Train] [Train] [Train] [TEST]

Report: mean score +/- std across 5 folds.
```

**Typical K values:** 5 or 10. Larger K: less bias in the estimate (more training data per fold), but more variance and more computation.

**Stratified K-fold:** For classification, ensure each fold has approximately the same class distribution as the full dataset. Critical for imbalanced data.

```python
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = []
for train_idx, val_idx in skf.split(X, y):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    model.fit(X_train, y_train)
    score = model.score(X_val, y_val)
    scores.append(score)

print(f"Mean: {np.mean(scores):.4f} +/- {np.std(scores):.4f}")
```

**Nested cross-validation:** When you are both selecting hyperparameters AND estimating generalization performance, you need two levels of cross-validation:
- Outer loop: estimate generalization performance.
- Inner loop: select best hyperparameters.

This avoids optimistic bias from tuning hyperparameters on the same validation set used to estimate performance.

**Time series cross-validation:** For time-ordered data, you cannot use random splits (that would be data leakage from the future). Use forward chaining:

```
Fold 1: [Train] | [Test]
Fold 2: [Train Train] | [Test]
Fold 3: [Train Train Train] | [Test]
Fold 4: [Train Train Train Train] | [Test]
```

Training set always precedes test set in time.

### 6.2 Hyperparameter Tuning

#### Grid Search

Try every combination of hyperparameter values from a predefined grid.

```python
param_grid = {
    'max_depth': [3, 5, 7, 10],
    'n_estimators': [50, 100, 200, 500],
    'learning_rate': [0.01, 0.05, 0.1, 0.2]
}
# Total combinations: 4 * 4 * 4 = 64
```

**Problem:** Exponential growth. With 5 hyperparameters and 5 values each, that is 5^5 = 3,125 combinations. Each evaluated with 5-fold CV = 15,625 model fits.

#### Random Search

Sample hyperparameter combinations randomly from defined distributions.

```python
param_distributions = {
    'max_depth': randint(3, 15),
    'n_estimators': randint(50, 1000),
    'learning_rate': uniform(0.001, 0.3)
}
# Sample, say, 100 random combinations.
```

**Why random search is better (Bergstra and Bengio, 2012):**
- Not all hyperparameters are equally important. In grid search, if one hyperparameter does not matter, you waste trials varying it while keeping the important one fixed.
- Random search explores more distinct values of each hyperparameter.

```
Grid Search:                    Random Search:
hp2 ^                           hp2 ^
    | x  x  x  x  x               | x   x    x
    | x  x  x  x  x               |    x   x
    | x  x  x  x  x               | x      x   x
    | x  x  x  x  x               |   x  x    x
    | x  x  x  x  x               | x    x  x
    +-----------------> hp1        +-----------------> hp1

    Only 5 unique values of        13 unique values of each.
    each hp explored.              More coverage with same budget.
```

#### Bayesian Optimization

Build a probabilistic model (surrogate) of the hyperparameter-to-performance mapping. Use it to decide which hyperparameters to try next.

**The idea:**
1. Evaluate a few random points.
2. Fit a Gaussian Process (or tree-based model) to the observed (hyperparameters, score) pairs.
3. Use an acquisition function (e.g., Expected Improvement) to select the next point to evaluate. This balances exploitation (trying near the current best) with exploration (trying uncertain regions).
4. Evaluate the selected point. Update the surrogate. Repeat.

**In practice:** Use `optuna`, `hyperopt`, or `scikit-optimize`. These libraries handle the details. Bayesian optimization is most valuable when each evaluation is expensive (e.g., training a large model).

### 6.3 Feature Engineering

Feature engineering is the process of transforming raw data into features that better represent the underlying problem. It is often more impactful than model selection.

**Common transformations:**

| Technique | When to use | Example |
|-----------|-------------|---------|
| Log transform | Right-skewed features | Income, prices |
| Square root | Count data | Word counts |
| Binning | Nonlinear relationships | Age -> age groups |
| Interaction features | Combined effects | area = length * width |
| Ratio features | Relative measures | price_per_sqft = price / sqft |
| One-hot encoding | Nominal categories | Color -> is_red, is_blue, is_green |
| Target encoding | High-cardinality categories | Zip code -> avg target per zip |
| Polynomial features | Nonlinear relationships | x, x^2, x^3 |

**Feature selection methods:**

1. **Filter methods:** Score each feature independently (correlation with target, mutual information, chi-squared test). Fast, but ignores feature interactions.

2. **Wrapper methods:** Evaluate feature subsets by training a model (forward selection, backward elimination). Expensive, but captures interactions.

3. **Embedded methods:** Feature selection built into the model (L1 regularization, tree feature importances). Practical middle ground.

### 6.4 Data Leakage

Data leakage occurs when information from the test set (or the future) leaks into the training process. It gives inflated performance estimates that do not generalize.

**Common leakage traps:**

1. **Scaling before splitting.** If you fit the scaler on the entire dataset (including test data), the test set statistics influence the training data transformations.

```python
# WRONG
scaler.fit(X)                    # Sees test data!
X_scaled = scaler.transform(X)
X_train, X_test = split(X_scaled)

# RIGHT
X_train, X_test = split(X)
scaler.fit(X_train)              # Only sees training data
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

2. **Target encoding without cross-validation.** If you compute the average target value per category using the full training set (including the current fold's validation data), you are leaking.

3. **Time-based leakage.** Using features that would not be available at prediction time. Example: using "days until event" when you would not know the event date in advance.

4. **Duplicate or near-duplicate samples** appearing in both train and test sets (common with augmented data).

**How to detect leakage:**
- Suspiciously high performance (if your model is 99.5% accurate on a hard problem, be skeptical).
- A large gap between cross-validation performance and holdout performance.
- Feature importance showing that a seemingly irrelevant feature dominates (it might be leaking the target).

### 6.5 Algorithm Selection Guide

```
START
  |
  v
Is your data tabular (structured)?
  |           |
 Yes          No --> Consider neural networks (images, text, audio, etc.)
  |
  v
How much data do you have?
  |                    |
 < 1000 samples       > 1000 samples
  |                    |
  v                    v
Consider:              Do you need interpretability?
- Logistic Reg           |              |
- SVM (RBF)             Yes             No
- Small RF               |              |
                         v              v
                    Consider:       Consider:
                    - Logistic Reg  - Random Forest
                    - Decision Tree - Gradient Boosting (XGBoost/LightGBM)
                    - Small RF      - Stacking/Blending
                    - Linear SVM

For any tabular problem in 2026,
gradient boosting (XGBoost, LightGBM, CatBoost)
is the default starting point.
```

**The no-free-lunch theorem:** No algorithm is universally best. But in practice:
- **Tabular data:** Gradient boosting dominates.
- **Images:** Convolutional neural networks (or vision transformers).
- **Text:** Transformers (BERT, GPT variants).
- **Small data + clear features:** SVMs and logistic regression are competitive.
- **Need interpretability:** Logistic regression or shallow decision trees.
- **Need a quick baseline:** Random forest (minimal tuning needed).

### 6.6 Regularization (Cross-Cutting Topic)

Regularization is any technique that prevents a model from fitting the training data too closely. It is a weapon against overfitting (high variance).

**L2 regularization (Ridge):** Add ||w||^2 to the loss.
```
L_ridge = MSE + lambda * sum(w_i^2)
```
Pushes weights toward zero but never exactly to zero. Shrinks all features.

**L1 regularization (Lasso):** Add ||w||_1 to the loss.
```
L_lasso = MSE + lambda * sum(|w_i|)
```
Pushes some weights to exactly zero. Performs feature selection.

**Elastic Net:** Combination of L1 and L2.

**In deep learning,** regularization takes many forms: L2 (weight decay), dropout, early stopping, data augmentation, batch normalization. The concept is the same — prevent the model from memorizing the training data.

---

## Appendix A: Mathematical Notation Reference

| Symbol | Meaning |
|--------|---------|
| n | Number of training samples |
| d | Number of features |
| K | Number of classes |
| X | Feature matrix, shape (n, d) |
| y | Target vector, shape (n,) |
| w | Weight vector, shape (d,) |
| b | Bias (scalar) |
| y_hat | Predictions |
| lr, eta | Learning rate |
| lambda | Regularization strength |
| sigma() | Sigmoid function |
| @ | Matrix multiplication (in NumPy) |
| ||v|| | L2 norm of vector v |
| X^T | Transpose of X |
| dL/dw | Partial derivative of L with respect to w |

---

## Appendix B: NumPy Cheat Sheet for ML

```python
import numpy as np

# Matrix operations
X.T                           # transpose
X @ w                         # matrix multiplication
np.linalg.inv(A)             # matrix inverse
np.linalg.eigh(A)            # eigenvalue decomposition (symmetric)
np.linalg.norm(v)            # vector norm

# Statistics
np.mean(X, axis=0)           # mean per feature
np.std(X, axis=0)            # std per feature
np.var(X, axis=0)            # variance per feature

# Common patterns
np.random.permutation(n)     # random permutation of indices
np.random.choice(n, size=k, replace=True)  # bootstrap sampling
np.clip(x, a, b)             # clip values to [a, b]
np.argsort(-scores)          # indices that would sort (descending)
np.unique(y, return_counts=True)  # unique values and their counts

# Broadcasting
X - X.mean(axis=0)           # center features (mean subtraction)
X / X.std(axis=0)            # scale features

# Activation functions
sigmoid = lambda z: 1 / (1 + np.exp(-z))
relu = lambda z: np.maximum(0, z)
softmax = lambda z: np.exp(z - z.max(-1, keepdims=True)) / \
                    np.exp(z - z.max(-1, keepdims=True)).sum(-1, keepdims=True)
```

---

## Appendix C: Key Formulas Quick Reference

**Linear Regression (MSE):**
```
Loss:     L = (1/n) * ||Xw + b - y||^2
Gradient: dw = (2/n) * X^T(Xw + b - y)
Normal:   w = (X^T X)^{-1} X^T y
```

**Logistic Regression (BCE):**
```
Model:    p = sigmoid(Xw + b)
Loss:     L = -(1/n) * [y^T log(p) + (1-y)^T log(1-p)]
Gradient: dw = (1/n) * X^T(p - y)
```

**Softmax + Categorical Cross-Entropy:**
```
Model:    p_k = exp(z_k) / sum(exp(z_j))
Loss:     L = -(1/n) * sum_i log(p_{i, true_class})
```

**SVM:**
```
Objective: min (1/2)||w||^2 + C * sum(xi_i)
Kernel:    K(x,y) = exp(-gamma * ||x-y||^2)    [RBF]
```

**Decision Tree:**
```
Gini:    G = 1 - sum(p_k^2)
Entropy: H = -sum(p_k * log2(p_k))
IG:      H(parent) - weighted_avg(H(children))
```

**PCA:**
```
Covariance:  Sigma = (1/n) * X^T X   (centered X)
Decompose:   Sigma = V Lambda V^T
Project:     X_reduced = X @ V[:, :k]
```

---

## Appendix D: Recommended Reading

1. **"Pattern Recognition and Machine Learning" by Christopher Bishop** — The comprehensive reference. Dense but excellent.
2. **"The Elements of Statistical Learning" by Hastie, Tibshirani, Friedman** — Free online. Rigorous statistical perspective.
3. **"Hands-On Machine Learning" by Aurelien Geron** — Practical, code-focused. Good for building intuition with implementations.
4. **Andrew Ng's Machine Learning course (Stanford CS229)** — Lecture notes are freely available. Excellent for linear models and gradient descent.
5. **XGBoost paper (Chen and Guestrin, 2016)** — Read it. It is well-written and not too long. Understanding XGBoost's innovations is valuable.
6. **"Random Forests" by Leo Breiman (2001)** — The original paper. Short and remarkably clear for a foundational work.

---

*"Every concept in these notes will reappear in deep learning, usually in a more complex form. If you master them here, where the math is clean and the models are interpretable, you will have an enormous advantage when things get complicated."*
