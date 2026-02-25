# Module 01: Classical ML Foundations — Lesson Plan

**Duration:** 2 weeks (6 sessions, ~3 hours each)
**Prerequisites:** Python proficiency, basic linear algebra, introductory ML exposure
**Goal:** Build the mathematical and algorithmic foundations that every deep learning concept rests upon. You should leave this module able to derive gradient descent from scratch, explain the bias-variance tradeoff to a colleague, and know *why* each classical method works — not just how to call sklearn.

---

## Session 1: Linear Regression Deep Dive

**Date:** Week 1, Day 1
**Duration:** 3 hours

### Learning Objectives

By the end of this session, you will be able to:
1. Derive the ordinary least squares loss function and explain why we use squared error.
2. Implement gradient descent from scratch (batch, stochastic, mini-batch) and explain the tradeoffs.
3. Derive and implement the closed-form normal equation solution.
4. Explain why feature scaling matters and demonstrate its effect on convergence.
5. Extend linear regression to polynomial features and reason about overfitting.

### Session Outline

| Time | Topic | Details |
|------|-------|---------|
| 0:00 - 0:20 | **The Problem Setup** | What does "fit a line" actually mean? Define the hypothesis, parameters, and the design matrix. Notation that carries into deep learning. |
| 0:20 - 0:50 | **The Loss Landscape** | Mean Squared Error — derive it, plot it in 2D (single weight), show it's convex. Why convexity matters. Visualize the loss surface for 2 parameters. Contour plots. |
| 0:50 - 1:30 | **Gradient Descent: The Core Algorithm** | Derive the gradient of MSE. Batch gradient descent step-by-step. Learning rate: too high (divergence), too low (slow), just right. Implement from scratch in NumPy. Stochastic GD: why randomness helps. Mini-batch: the practical compromise. Convergence criteria. |
| 1:30 - 1:45 | **Break** | |
| 1:45 - 2:10 | **The Closed-Form Solution** | Normal equation derivation. When to use it vs. gradient descent. Computational complexity comparison. Matrix invertibility issues. |
| 2:10 - 2:35 | **Feature Scaling and Engineering** | Why unscaled features break gradient descent (elongated contours). Standardization vs. normalization. Polynomial features: extending the linear framework. The overfitting cliff. |
| 2:35 - 2:55 | **Hands-On Exercise** | Implement linear regression from scratch. Compare GD variants. Visualize convergence. |
| 2:55 - 3:00 | **Key Takeaways and Preview** | Gradient descent is the single most important algorithm you will use in deep learning. Everything else is variations on this theme. |

### Exercises

1. Implement batch gradient descent for linear regression in pure NumPy.
2. Plot the loss curve for 3 different learning rates on the same graph.
3. Implement mini-batch GD and compare convergence noise to batch GD.
4. Solve the same problem with the normal equation. Compare wall-clock time for n=100, n=10000, n=100000.
5. Add polynomial features (degree 2, 5, 10) and observe overfitting.

### Key Takeaways

- Gradient descent is not specific to linear regression — it is the universal optimization algorithm for ML/DL.
- The learning rate is the most important hyperparameter you will ever tune.
- Feature scaling is not optional — it is a prerequisite for stable optimization.
- Linear regression with polynomial features is still "linear" in the parameters.

---

## Session 2: Logistic Regression and Classification

**Date:** Week 1, Day 2
**Duration:** 3 hours

### Learning Objectives

By the end of this session, you will be able to:
1. Explain why linear regression fails for classification and how the sigmoid function fixes it.
2. Derive cross-entropy loss from maximum likelihood estimation.
3. Implement logistic regression with gradient descent from scratch.
4. Compute and interpret all standard classification metrics.
5. Extend binary logistic regression to multi-class via softmax.

### Session Outline

| Time | Topic | Details |
|------|-------|---------|
| 0:00 - 0:20 | **Why Not Linear Regression for Classification?** | The problem with predicting probabilities outside [0,1]. The sigmoid (logistic) function: shape, properties, derivative. Interpreting outputs as probabilities. |
| 0:20 - 0:55 | **Cross-Entropy Loss from First Principles** | Maximum likelihood estimation review. The likelihood of a Bernoulli model. Taking the log. Arriving at binary cross-entropy. Why it works better than MSE for classification (gradient magnitude argument). |
| 0:55 - 1:25 | **Gradient Descent for Logistic Regression** | Derive the gradient (it is surprisingly clean). Implement from scratch. Decision boundaries: what they are, how to plot them, linear vs. nonlinear. |
| 1:25 - 1:40 | **Break** | |
| 1:40 - 2:15 | **Evaluation Metrics Deep Dive** | Accuracy and its failures (class imbalance). Confusion matrix. Precision, recall, F1-score — when each matters. ROC curve and AUC — the complete story. Precision-recall curves. Threshold tuning. |
| 2:15 - 2:35 | **Multi-Class Classification** | One-vs-rest vs. softmax. The softmax function: generalizing sigmoid. Categorical cross-entropy. Brief connection to neural network output layers. |
| 2:35 - 2:55 | **Hands-On Exercise** | Build logistic regression from scratch. Compute all metrics manually. Plot ROC curve. |
| 2:55 - 3:00 | **Key Takeaways and Preview** | Cross-entropy loss and softmax will follow you into every classification network you ever build. |

### Exercises

1. Implement the sigmoid function. Verify numerically that sigmoid'(x) = sigmoid(x)(1 - sigmoid(x)).
2. Implement binary cross-entropy loss from scratch. Compare its gradient to MSE's gradient for classification.
3. Build a complete logistic regression classifier with gradient descent.
4. Implement precision, recall, F1, and ROC-AUC from scratch (no sklearn).
5. Plot the decision boundary for a 2D dataset. Add polynomial features and re-plot.

### Key Takeaways

- Cross-entropy loss comes from maximum likelihood estimation — it is not an arbitrary choice.
- The sigmoid and softmax functions map real numbers to valid probability distributions.
- Accuracy is a terrible metric for imbalanced datasets. Always look at the confusion matrix first.
- Every neural network classifier uses softmax + cross-entropy. You just derived why.

---

## Session 3: SVMs and Kernel Methods

**Date:** Week 1, Day 3
**Duration:** 3 hours

### Learning Objectives

By the end of this session, you will be able to:
1. Explain the maximum margin principle and why it leads to better generalization.
2. Identify support vectors and explain their role in the decision boundary.
3. Explain the kernel trick and why it is computationally brilliant.
4. Apply SVMs with different kernels and interpret the results.
5. Articulate when SVMs outperform neural networks.

### Session Outline

| Time | Topic | Details |
|------|-------|---------|
| 0:00 - 0:25 | **Maximum Margin Intuition** | Many hyperplanes can separate data — which is best? The margin concept. Why maximizing margin improves generalization (connection to bias-variance). The geometric derivation. |
| 0:25 - 0:55 | **The SVM Optimization Problem** | The constrained optimization formulation. Support vectors: the only data points that matter. Lagrange multipliers (intuition, not full derivation). The dual formulation preview. |
| 0:55 - 1:30 | **The Kernel Trick** | What happens when data is not linearly separable. Mapping to higher dimensions. The computational nightmare of explicit mapping. The kernel trick: compute dot products in high-dimensional space without ever going there. RBF kernel: infinite-dimensional feature space. Polynomial kernel. |
| 1:30 - 1:45 | **Break** | |
| 1:45 - 2:15 | **Soft Margin and Practical SVMs** | Real data is never perfectly separable. The C parameter: trading margin width for classification errors. The nu-SVM variant. Choosing kernels and hyperparameters. |
| 2:15 - 2:40 | **SVMs in Practice** | When SVMs beat neural networks (small data, tabular data, clear margin). When they do not (images, text, massive data). SVMs for regression (SVR). Computational scaling issues. |
| 2:40 - 2:55 | **Hands-On Exercise** | Apply SVMs with linear, polynomial, and RBF kernels. Visualize decision boundaries. Tune C and gamma. |
| 2:55 - 3:00 | **Key Takeaways and Preview** | The kernel trick idea (computing in implicit feature spaces) reappears in attention mechanisms and other DL architectures. |

### Exercises

1. Generate linearly separable 2D data. Fit a linear SVM, identify the support vectors, draw the margin.
2. Generate non-linearly separable data (e.g., concentric circles). Show that a linear SVM fails. Apply RBF kernel.
3. Vary C from 0.01 to 1000. Plot decision boundaries for each. Explain the bias-variance effect.
4. Vary gamma for the RBF kernel. Observe underfitting and overfitting.
5. Compare SVM, logistic regression, and (if time) a small neural net on a tabular dataset. When does each win?

### Key Takeaways

- SVMs find the decision boundary with the largest margin, which is a form of regularization.
- Support vectors are the critical training points — removing non-support-vectors does not change the model.
- The kernel trick lets you work in high (even infinite) dimensional feature spaces at the cost of computing a kernel function.
- SVMs remain competitive on small-to-medium tabular datasets.

---

## Session 4: Trees, Forests, and Boosting

**Date:** Week 2, Day 1
**Duration:** 3 hours

### Learning Objectives

By the end of this session, you will be able to:
1. Build a decision tree from scratch using information gain or Gini impurity.
2. Explain why random forests work (bagging + feature randomness = decorrelated trees).
3. Explain gradient boosting as gradient descent in function space.
4. Deeply understand the bias-variance tradeoff and how ensembles exploit it.
5. Know when to reach for tree-based methods vs. neural networks.

### Session Outline

| Time | Topic | Details |
|------|-------|---------|
| 0:00 - 0:35 | **Decision Trees from Scratch** | Recursive partitioning. Splitting criteria: information gain (entropy), Gini impurity — derive both, show they are similar. The greedy algorithm. When to stop splitting. Pruning. Trees for regression (variance reduction). |
| 0:35 - 1:05 | **The Bias-Variance Tradeoff — Deeply** | Decompose expected error into bias, variance, and irreducible noise. A single deep tree: low bias, high variance. A shallow tree: high bias, low variance. This is the most important concept in all of ML. Visualize it. |
| 1:05 - 1:30 | **Random Forests: Variance Reduction Through Ensembles** | Bagging: train on bootstrap samples. Why averaging reduces variance (the math). Feature randomness: decorrelating trees. Out-of-bag error as free validation. Feature importance. |
| 1:30 - 1:45 | **Break** | |
| 1:45 - 2:20 | **Gradient Boosting: Bias Reduction Through Ensembles** | Boosting vs. bagging: fundamentally different philosophies. Gradient boosting as gradient descent in function space. Each tree fits the residuals. Learning rate (shrinkage). XGBoost innovations: regularization, column sampling, histogram binning. |
| 2:20 - 2:40 | **Practical Ensemble Methods** | XGBoost vs. LightGBM vs. CatBoost. Hyperparameter tuning guide. Why gradient boosting dominates Kaggle tabular competitions. When neural networks beat trees. |
| 2:40 - 2:55 | **Hands-On Exercise** | Implement a decision tree split. Train RF and GBM on real data. Compare. |
| 2:55 - 3:00 | **Key Takeaways and Preview** | The ensemble principle (combining weak learners) appears in DL as dropout, model averaging, and mixture of experts. |

### Exercises

1. Implement the Gini impurity and information gain calculations from scratch.
2. Implement a basic decision tree that finds the best split for a single feature.
3. Train a Random Forest. Vary n_estimators from 1 to 500. Plot OOB error.
4. Train XGBoost. Perform learning rate vs. n_estimators tradeoff analysis.
5. On a real dataset: compare a single tree, random forest, gradient boosting, and logistic regression. Create a comparison table with all metrics.

### Key Takeaways

- A single decision tree is interpretable but unstable (high variance).
- Random forests reduce variance by averaging decorrelated trees.
- Gradient boosting reduces bias by sequentially fitting residuals.
- The bias-variance tradeoff is not just theory — it dictates your choice of model and hyperparameters.
- Tree-based ensembles are the default choice for structured/tabular data even in 2026.

---

## Session 5: Unsupervised Learning Essentials

**Date:** Week 2, Day 2
**Duration:** 3 hours

### Learning Objectives

By the end of this session, you will be able to:
1. Implement K-Means clustering and understand its failure modes.
2. Derive PCA from the variance maximization perspective.
3. Explain t-SNE intuitively and know when to use it (and when not to).
4. Connect dimensionality reduction to autoencoders in deep learning.
5. Apply unsupervised methods for exploratory data analysis and preprocessing.

### Session Outline

| Time | Topic | Details |
|------|-------|---------|
| 0:00 - 0:30 | **K-Means Clustering** | The algorithm step by step. The objective function (within-cluster sum of squares). Initialization matters: K-Means++. Choosing K: elbow method, silhouette scores. Failure modes: non-convex clusters, different densities, high dimensions. |
| 0:30 - 1:10 | **PCA: The Most Important Unsupervised Method** | The variance maximization formulation. Eigenvalue decomposition of the covariance matrix. What eigenvectors and eigenvalues tell you. Choosing the number of components (explained variance). PCA for visualization. PCA as a preprocessing step. Connection to SVD. |
| 1:10 - 1:30 | **t-SNE and Modern Visualization** | Why PCA fails for complex manifolds. t-SNE intuition: preserve local neighborhoods. Perplexity parameter. Why t-SNE is non-parametric and non-deterministic. UMAP as an alternative. Pitfalls: do not interpret distances or cluster sizes. |
| 1:30 - 1:45 | **Break** | |
| 1:45 - 2:15 | **Dimensionality Reduction Meets Deep Learning** | Linear dimensionality reduction (PCA) vs. nonlinear (autoencoders). The autoencoder as a generalization of PCA. Why this matters for representation learning. Preview of variational autoencoders. |
| 2:15 - 2:40 | **Other Unsupervised Methods** | Hierarchical clustering. DBSCAN for arbitrary cluster shapes. Gaussian mixture models and the EM algorithm (intuition). |
| 2:40 - 2:55 | **Hands-On Exercise** | Cluster a dataset with K-Means. Apply PCA. Visualize with t-SNE. Compare. |
| 2:55 - 3:00 | **Key Takeaways and Preview** | Representation learning is the core idea of deep learning. Unsupervised methods teach you to think about data structure. |

### Exercises

1. Implement K-Means from scratch. Show the cluster assignments updating over iterations.
2. Implement PCA via eigenvalue decomposition of the covariance matrix.
3. Apply PCA to a high-dimensional dataset. Plot explained variance ratio. Choose components.
4. Use t-SNE to visualize a labeled dataset (e.g., digits). Color by true label. Experiment with perplexity.
5. Compare PCA and t-SNE on the same dataset. When does each tell a better story?

### Key Takeaways

- K-Means is fast and intuitive but assumes convex, similarly-sized clusters.
- PCA finds the directions of maximum variance — it is the foundation of dimensionality reduction.
- t-SNE is for visualization only, not for downstream ML.
- Autoencoders generalize PCA to nonlinear dimensionality reduction.

---

## Session 6: The ML Pipeline

**Date:** Week 2, Day 3
**Duration:** 3 hours

### Learning Objectives

By the end of this session, you will be able to:
1. Design a proper cross-validation scheme and explain why k-fold beats a single train/test split.
2. Perform hyperparameter tuning with grid search, random search, and Bayesian optimization.
3. Engineer features that improve model performance.
4. Identify and prevent data leakage.
5. Choose the right algorithm for a given problem.

### Session Outline

| Time | Topic | Details |
|------|-------|---------|
| 0:00 - 0:30 | **Cross-Validation Done Right** | Why a single train/test split is dangerous. K-fold cross-validation. Stratified K-fold for classification. Leave-one-out (and why it is usually a bad idea). Nested cross-validation for unbiased model comparison. Time series cross-validation (forward chaining). |
| 0:30 - 1:00 | **Hyperparameter Tuning** | Grid search: exhaustive but exponential. Random search: why it is usually better (Bergstra and Bengio, 2012). Bayesian optimization: surrogate models and acquisition functions. Practical tuning strategies. How many trials are enough. |
| 1:00 - 1:25 | **Feature Engineering** | Domain knowledge matters. Numerical features: scaling, binning, interactions, log transforms. Categorical features: one-hot, target encoding, frequency encoding. Text features: TF-IDF, embeddings preview. Missing values: strategies and their tradeoffs. Feature selection: filter, wrapper, embedded methods. |
| 1:25 - 1:40 | **Break** | |
| 1:40 - 2:10 | **Data Leakage: The Silent Killer** | What data leakage is and why it destroys your model. Common leakage traps: scaling before splitting, using future data, target leakage in features. Leakage through feature engineering. How to detect leakage: suspiciously high performance, feature importance analysis. |
| 2:10 - 2:40 | **Algorithm Selection Guide** | Decision framework: data size, feature types, interpretability needs, training time. Linear models: when and why. Tree models: when and why. SVMs: when and why. Neural networks: when and why. The "no free lunch" theorem in practice. |
| 2:40 - 2:55 | **Capstone Exercise** | End-to-end pipeline: data exploration, feature engineering, model selection, hyperparameter tuning, evaluation. |
| 2:55 - 3:00 | **Module Wrap-Up** | What you have learned. How it all connects to deep learning. Preview of the next module. |

### Exercises

1. Implement k-fold cross-validation from scratch using NumPy.
2. Compare grid search vs. random search on a hyperparameter grid for Random Forest. Plot results.
3. Take a raw dataset. Engineer at least 5 meaningful features. Measure their impact.
4. Create a deliberately leaky pipeline. Show the inflated performance. Fix it and compare.
5. Given a novel dataset, write a 1-page analysis recommending an algorithm with justification.

### Key Takeaways

- Cross-validation is non-negotiable. A single train/test split gives you a single noisy estimate.
- Random search is almost always preferable to grid search for hyperparameter tuning.
- Feature engineering is where domain knowledge meets data science — it often matters more than model choice.
- Data leakage will give you beautiful results that mean nothing. Always audit your pipeline.
- There is no universally best algorithm. There is only the best algorithm for your data, your constraints, and your problem.

---

## Assessment and Progression

**Assignment 1:** Linear Regression from Scratch (due after Session 1)
**Assignment 2:** Logistic Regression and Evaluation Metrics (due after Session 2)
**Assignment 3:** Ensemble Methods Mini-Competition (due end of Week 2)

**Progression Criteria:** To move to Module 02 (Deep Learning Fundamentals), you must:
- Complete all three assignments with passing marks.
- Demonstrate that you can derive gradient descent and cross-entropy loss on a whiteboard (or in a live discussion).
- Articulate the bias-variance tradeoff clearly and correctly.
- Show a working end-to-end ML pipeline with proper validation.

---

*"The foundations are not the boring part. They are the part that lets you understand everything that comes after."*
