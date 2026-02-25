# Assignment 02: Calculus and Optimization — Calculus Meets Code

## Overview

This assignment bridges the gap between calculus on paper and gradient-based optimization
in practice. You will implement derivatives, the chain rule, and gradient descent from
scratch, then visualize what these concepts look like on real functions and loss landscapes.

The central exercise — implementing backpropagation by hand for a 3-layer computation
graph — is the single most important exercise in this module. If you understand that
exercise, you understand how neural networks learn. Everything else is engineering.

**Estimated time:** 8-10 hours
**Language:** Python (NumPy for core implementations, PyTorch for verification)
**Submission:** Jupyter notebook (.ipynb) with all code, outputs, visualizations, and written answers.

---

## Exercise 1: Numerical Differentiation

**Why this matters:** Numerical differentiation is your ground truth. When you implement
a new gradient computation (custom autograd function, new loss, etc.), you verify it
by comparing against numerical derivatives. If they disagree, your analytical gradient
is wrong. Always.

### Task

#### 1a. Implement Three Finite Difference Methods

```python
def forward_difference(f, x, h=1e-7):
    """
    f'(x) ~ [f(x + h) - f(x)] / h

    Accuracy: O(h) — first-order
    """
    # Your implementation
    pass

def central_difference(f, x, h=1e-7):
    """
    f'(x) ~ [f(x + h) - f(x - h)] / (2h)

    Accuracy: O(h^2) — second-order (much better!)
    """
    # Your implementation
    pass

def complex_step(f, x, h=1e-20):
    """
    f'(x) ~ Im[f(x + ih)] / h

    Accuracy: O(h^2) with NO subtractive cancellation.
    Works only for functions that can accept complex inputs.
    This is the gold standard for gradient checking.
    """
    # Your implementation
    # Hint: f must be called with x + 1j * h
    pass
```

#### 1b. Compare Accuracy

For f(x) = sin(x) at x = 1.0 (where the true derivative is cos(1.0)):

1. Compute the derivative using all three methods for h = 10^(-1), 10^(-2), ..., 10^(-15)
2. Plot the absolute error vs h on a log-log scale
3. Observe: forward difference has a V-shaped error curve (too large h = truncation error, too small h = floating point cancellation). Central difference is better. Complex step has no cancellation problem.

```python
import numpy as np
import matplotlib.pyplot as plt

f = np.sin
true_deriv = np.cos(1.0)
x = 1.0

hs = np.logspace(-1, -15, 30)
errors_forward = []
errors_central = []
errors_complex = []

for h in hs:
    errors_forward.append(abs(forward_difference(f, x, h) - true_deriv))
    errors_central.append(abs(central_difference(f, x, h) - true_deriv))
    errors_complex.append(abs(complex_step(f, x, h) - true_deriv))

plt.figure(figsize=(10, 6))
plt.loglog(hs, errors_forward, label='Forward difference')
plt.loglog(hs, errors_central, label='Central difference')
plt.loglog(hs, errors_complex, label='Complex step')
plt.xlabel('Step size h')
plt.ylabel('Absolute error')
plt.title('Numerical Differentiation: Accuracy vs Step Size')
plt.legend()
plt.grid(True)
plt.gca().invert_xaxis()
plt.savefig('numerical_diff_accuracy.png', dpi=150)
plt.show()
```

**Written question:** Why does forward difference accuracy WORSEN when h becomes very
small (below ~10^(-8))? Why does the complex step method not have this problem?

#### 1c. Gradient Checking for Common DL Functions

Implement the analytical derivatives for these functions and verify them numerically:

1. **ReLU**: f(x) = max(0, x). Derivative = 1 if x > 0, 0 if x < 0.
2. **Sigmoid**: f(x) = 1/(1 + exp(-x)). Derivative = f(x)(1 - f(x)).
3. **Softplus**: f(x) = log(1 + exp(x)). Derivative = sigmoid(x).
4. **Tanh**: f(x) = tanh(x). Derivative = 1 - tanh(x)^2.

For each function:
- Plot the function and its derivative on the same axes (range [-5, 5])
- Verify the analytical derivative matches numerical derivative at 10 random points
- Print the maximum absolute error

```python
# Template for each function
def check_gradient(f, f_prime, name, x_range=(-5, 5), n_points=10):
    """Verify analytical gradient against numerical gradient."""
    x_test = np.random.uniform(x_range[0], x_range[1], n_points)
    max_error = 0
    for x in x_test:
        analytical = f_prime(x)
        numerical = central_difference(f, x)
        error = abs(analytical - numerical)
        max_error = max(max_error, error)
    print(f"{name}: max error = {max_error:.2e}")
    assert max_error < 1e-5, f"Gradient check FAILED for {name}!"
```

### Deliverable for Exercise 1

- All three numerical differentiation methods implemented
- Log-log accuracy plot
- Analytical derivatives for all four activation functions, verified numerically
- Function + derivative plots for all four functions
- Written answer about floating point cancellation (3-5 sentences)

---

## Exercise 2: The Chain Rule — Backpropagation by Hand

**Why this matters:** This is the exercise. If you can compute gradients by hand through
a computation graph, verify them with PyTorch, and understand every intermediate value,
you understand backpropagation. There is no shortcut for this understanding.

### Task

#### 2a. A Simple Computation Graph

Consider this computation:

```
Input: x = 2.0

z1 = x^2          (squaring)
z2 = z1 + 3       (adding constant)
z3 = sin(z2)      (sine)
L  = z3^2         (squaring — "loss")
```

By hand (write it in a markdown cell in your notebook):
1. Compute the forward pass: what are z1, z2, z3, L?
2. Compute dL/dz3, dz3/dz2, dz2/dz1, dz1/dx
3. Apply the chain rule: dL/dx = dL/dz3 * dz3/dz2 * dz2/dz1 * dz1/dx
4. Compute the numerical value

Then verify with PyTorch:

```python
import torch

x = torch.tensor(2.0, requires_grad=True)
z1 = x ** 2
z2 = z1 + 3
z3 = torch.sin(z2)
L = z3 ** 2
L.backward()

print(f"PyTorch gradient: dL/dx = {x.grad.item()}")
print(f"Your gradient:    dL/dx = ???")  # Replace with your answer
```

#### 2b. A 3-Layer Neural Network — Full Backpropagation

This is the core exercise. Implement forward and backward passes for a 3-layer network
**by hand in NumPy**, then verify every gradient with PyTorch autograd.

Architecture:
- Input: x (scalar, 1 value)
- Layer 1: z1 = w1 * x + b1, a1 = sigmoid(z1)
- Layer 2: z2 = w2 * a1 + b2, a2 = sigmoid(z2)
- Layer 3: z3 = w3 * a2 + b3, y_pred = sigmoid(z3)
- Loss: L = (y_pred - y_true)^2

```python
import numpy as np

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_deriv(x):
    s = sigmoid(x)
    return s * (1.0 - s)

# Fixed parameters
x = 1.5
y_true = 0.8
w1, b1 = 0.5, 0.1
w2, b2 = -0.3, 0.2
w3, b3 = 0.7, -0.1

# ============ FORWARD PASS ============
# Compute z1, a1, z2, a2, z3, y_pred, L
# Show every intermediate value.

z1 = w1 * x + b1
a1 = sigmoid(z1)
z2 = w2 * a1 + b2
a2 = sigmoid(z2)
z3 = w3 * a2 + b3
y_pred = sigmoid(z3)
L = (y_pred - y_true) ** 2

print("=== FORWARD PASS ===")
print(f"z1 = {z1:.6f}, a1 = {a1:.6f}")
print(f"z2 = {z2:.6f}, a2 = {a2:.6f}")
print(f"z3 = {z3:.6f}, y_pred = {y_pred:.6f}")
print(f"L = {L:.6f}")

# ============ BACKWARD PASS ============
# Compute ALL gradients using the chain rule.
# For each gradient, write the chain rule formula in a comment.

# dL/dy_pred = 2 * (y_pred - y_true)
dL_dy_pred = 2 * (y_pred - y_true)

# dL/dz3 = dL/dy_pred * dy_pred/dz3 = dL/dy_pred * sigmoid'(z3)
dL_dz3 = dL_dy_pred * sigmoid_deriv(z3)

# dL/dw3 = dL/dz3 * dz3/dw3 = dL/dz3 * a2
dL_dw3 = dL_dz3 * a2

# dL/db3 = dL/dz3 * dz3/db3 = dL/dz3 * 1
dL_db3 = dL_dz3

# dL/da2 = dL/dz3 * dz3/da2 = dL/dz3 * w3
dL_da2 = dL_dz3 * w3

# Continue for all remaining gradients...
# YOUR CODE HERE: compute dL/dz2, dL/dw2, dL/db2, dL/da1, dL/dz1, dL/dw1, dL/db1, dL/dx

print("\n=== BACKWARD PASS ===")
print(f"dL/dw3 = {dL_dw3:.6f}")
print(f"dL/db3 = {dL_db3:.6f}")
# print all other gradients...
```

Now verify EVERY gradient with PyTorch:

```python
import torch

x_t = torch.tensor(1.5, dtype=torch.float64)
w1_t = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)
b1_t = torch.tensor(0.1, dtype=torch.float64, requires_grad=True)
w2_t = torch.tensor(-0.3, dtype=torch.float64, requires_grad=True)
b2_t = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
w3_t = torch.tensor(0.7, dtype=torch.float64, requires_grad=True)
b3_t = torch.tensor(-0.1, dtype=torch.float64, requires_grad=True)

z1_t = w1_t * x_t + b1_t
a1_t = torch.sigmoid(z1_t)
z2_t = w2_t * a1_t + b2_t
a2_t = torch.sigmoid(z2_t)
z3_t = w3_t * a2_t + b3_t
y_pred_t = torch.sigmoid(z3_t)
L_t = (y_pred_t - 0.8) ** 2

L_t.backward()

print("\n=== PYTORCH VERIFICATION ===")
print(f"dL/dw3: manual = {dL_dw3:.6f}, pytorch = {w3_t.grad.item():.6f}")
print(f"dL/db3: manual = {dL_db3:.6f}, pytorch = {b3_t.grad.item():.6f}")
# Compare all gradients. They MUST match to at least 6 decimal places.
```

**Written question:** In the backward pass, you computed dL/da2 = dL/dz3 * w3. Notice
that this gradient flows THROUGH the weight w3. If w3 is very small (close to 0), what
happens to the gradient flowing to earlier layers? If w3 is very large, what happens?
Relate this to the vanishing and exploding gradient problems. (4-6 sentences)

#### 2c. Visualize the Gradient Flow

Create a diagram (using matplotlib or text) showing the computation graph with:
- Forward pass values at each node
- Backward pass gradients on each edge
- Color-code the gradients: green for large (>0.1), yellow for medium (0.01-0.1), red for small (<0.01)

This visualization is how you develop intuition for where gradients vanish.

### Deliverable for Exercise 2

- Simple computation graph: hand-computed and verified gradients
- 3-layer network: ALL gradients hand-computed and verified against PyTorch (printed comparison table)
- Written answer about vanishing/exploding gradients
- Computation graph visualization with gradient magnitudes

---

## Exercise 3: Gradient Descent on 2D Functions

**Why this matters:** Visualizing gradient descent in 2D builds geometric intuition for
what happens in million-dimensional parameter space. The optimizer behaviors you see in
2D — spiraling, oscillation, getting stuck — all happen in high dimensions too.

### Task

#### 3a. Implement Gradient Descent

```python
def gradient_descent(f, grad_f, x0, lr=0.01, n_steps=100):
    """
    Vanilla gradient descent.

    Args:
        f: function to minimize, takes numpy array, returns scalar
        grad_f: gradient function, takes numpy array, returns numpy array
        x0: initial point (numpy array)
        lr: learning rate
        n_steps: number of steps
    Returns:
        trajectory: list of points visited (including x0)
        losses: list of function values at each point
    """
    trajectory = [x0.copy()]
    losses = [f(x0)]
    x = x0.copy()

    for _ in range(n_steps):
        g = grad_f(x)
        x = x - lr * g
        trajectory.append(x.copy())
        losses.append(f(x))

    return np.array(trajectory), np.array(losses)
```

#### 3b. Test Functions

Implement these classic optimization test functions and their gradients:

```python
# 1. Rosenbrock function (banana-shaped valley)
# f(x, y) = (1 - x)^2 + 100*(y - x^2)^2
# Minimum at (1, 1), f = 0
# This is HARD to optimize: the minimum is inside a narrow, curved valley.

def rosenbrock(xy):
    x, y = xy
    return (1 - x)**2 + 100 * (y - x**2)**2

def rosenbrock_grad(xy):
    x, y = xy
    dfdx = -2*(1 - x) + 200*(y - x**2)*(-2*x)
    dfdy = 200*(y - x**2)
    return np.array([dfdx, dfdy])


# 2. Rastrigin function (many local minima)
# f(x, y) = 20 + (x^2 - 10*cos(2*pi*x)) + (y^2 - 10*cos(2*pi*y))
# Global minimum at (0, 0), f = 0
# This is HARD because of the many local minima.

def rastrigin(xy):
    x, y = xy
    return 20 + (x**2 - 10*np.cos(2*np.pi*x)) + (y**2 - 10*np.cos(2*np.pi*y))

def rastrigin_grad(xy):
    x, y = xy
    dfdx = 2*x + 10*2*np.pi*np.sin(2*np.pi*x)
    dfdy = 2*y + 10*2*np.pi*np.sin(2*np.pi*y)
    return np.array([dfdx, dfdy])


# 3. Quadratic with different curvatures (ill-conditioned)
# f(x, y) = 0.5 * x^2 + 10 * y^2
# Minimum at (0, 0)
# The y-direction has 20x more curvature than x.
# Gradient descent oscillates in y and crawls in x.

def ill_conditioned_quadratic(xy):
    x, y = xy
    return 0.5 * x**2 + 10 * y**2

def ill_conditioned_grad(xy):
    x, y = xy
    return np.array([x, 20*y])
```

#### 3c. Visualize Trajectories

For each function, create a contour plot with the gradient descent trajectory overlaid:

```python
def plot_trajectory(f, trajectory, title, xlim, ylim, levels=50):
    """
    Plot contour lines of f with the optimization trajectory overlaid.

    - Use filled contour plot (plt.contourf) for the function
    - Plot the trajectory as connected dots (red line with dots at each step)
    - Mark the starting point (green star) and ending point (red star)
    - Mark the true minimum (black X) if known
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    x_range = np.linspace(xlim[0], xlim[1], 200)
    y_range = np.linspace(ylim[0], ylim[1], 200)
    X, Y = np.meshgrid(x_range, y_range)
    Z = np.array([[f(np.array([x, y])) for x in x_range] for y in y_range])

    ax.contourf(X, Y, Z, levels=levels, cmap='viridis')
    ax.plot(trajectory[:, 0], trajectory[:, 1], 'r.-', markersize=3, linewidth=0.8)
    ax.plot(trajectory[0, 0], trajectory[0, 1], 'g*', markersize=15, label='Start')
    ax.plot(trajectory[-1, 0], trajectory[-1, 1], 'r*', markersize=15, label='End')
    ax.set_title(title)
    ax.legend()
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    plt.tight_layout()
    return fig
```

Run gradient descent on each function with several learning rates. Show:
1. Rosenbrock: lr = 0.001, 0.0001 (large lr diverges), start at (-1, -1), 5000 steps
2. Rastrigin: lr = 0.01, start at (3, 3), 200 steps — observe getting stuck in local minimum
3. Ill-conditioned quadratic: lr = 0.01, start at (5, 1), 200 steps — observe oscillation

**Written question:** For the ill-conditioned quadratic, gradient descent oscillates
wildly in the y-direction while making slow progress in x. Why? What does the condition
number of the Hessian have to do with this? (4-6 sentences)

### Deliverable for Exercise 3

- Gradient descent implementation
- All three test functions with analytical gradients
- Contour plots with trajectories for each function (at least 2 learning rates each)
- Loss-vs-step plots for each run
- Written answer about ill-conditioning

---

## Exercise 4: Optimizer Showdown

**Why this matters:** You will use Adam for most training. But you should understand WHY
Adam works and what each piece does. Implementing the optimizers from scratch and seeing
them race on the same landscape builds this understanding.

### Task

#### 4a. Implement Four Optimizers

```python
class VanillaGD:
    def __init__(self, lr=0.01):
        self.lr = lr

    def step(self, params, grads):
        """
        params = params - lr * grads
        """
        return params - self.lr * grads


class Momentum:
    def __init__(self, lr=0.01, beta=0.9):
        self.lr = lr
        self.beta = beta
        self.v = None  # Velocity

    def step(self, params, grads):
        """
        v = beta * v + grads
        params = params - lr * v

        Momentum accumulates past gradients. If the gradient keeps pointing
        the same direction, velocity builds up (like a ball rolling downhill).
        If the gradient oscillates, velocity averages out the oscillations.
        """
        if self.v is None:
            self.v = np.zeros_like(params)
        self.v = self.beta * self.v + grads
        return params - self.lr * self.v


class RMSProp:
    def __init__(self, lr=0.01, beta=0.99, epsilon=1e-8):
        self.lr = lr
        self.beta = beta
        self.epsilon = epsilon
        self.s = None  # Running average of squared gradients

    def step(self, params, grads):
        """
        s = beta * s + (1 - beta) * grads^2
        params = params - lr * grads / (sqrt(s) + epsilon)

        RMSProp divides by the running RMS of gradients. Directions with
        large gradients get smaller effective learning rates. Directions
        with small gradients get larger effective learning rates. This
        adapts to the curvature.
        """
        if self.s is None:
            self.s = np.zeros_like(params)
        # Your implementation
        pass


class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None  # First moment (mean of gradients)
        self.v = None  # Second moment (mean of squared gradients)
        self.t = 0     # Time step

    def step(self, params, grads):
        """
        m = beta1 * m + (1 - beta1) * grads           # Update first moment
        v = beta2 * v + (1 - beta2) * grads^2         # Update second moment
        m_hat = m / (1 - beta1^t)                      # Bias correction
        v_hat = v / (1 - beta2^t)                      # Bias correction
        params = params - lr * m_hat / (sqrt(v_hat) + epsilon)

        Adam = Momentum + RMSProp + bias correction.
        The bias correction accounts for the fact that m and v are
        initialized to zero and need a few steps to "warm up."
        """
        if self.m is None:
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)
        self.t += 1
        # Your implementation
        pass
```

#### 4b. Race on the Ill-Conditioned Quadratic

Run all four optimizers on f(x, y) = 0.5*x^2 + 10*y^2, starting from (5, 1):

- Vanilla GD: lr = 0.05
- Momentum: lr = 0.05, beta = 0.9
- RMSProp: lr = 0.1, beta = 0.99
- Adam: lr = 0.1, beta1 = 0.9, beta2 = 0.999

Create two plots:
1. **Trajectory plot**: All four trajectories on the same contour plot, different colors
2. **Convergence plot**: Loss vs step for all four, log scale on y-axis

```python
optimizers = {
    'Vanilla GD': VanillaGD(lr=0.05),
    'Momentum': Momentum(lr=0.05, beta=0.9),
    'RMSProp': RMSProp(lr=0.1, beta=0.99),
    'Adam': Adam(lr=0.1, beta1=0.9, beta2=0.999)
}

results = {}
x0 = np.array([5.0, 1.0])

for name, opt in optimizers.items():
    x = x0.copy()
    trajectory = [x.copy()]
    losses = [ill_conditioned_quadratic(x)]

    for _ in range(200):
        g = ill_conditioned_grad(x)
        x = opt.step(x, g)
        trajectory.append(x.copy())
        losses.append(ill_conditioned_quadratic(x))

    results[name] = {
        'trajectory': np.array(trajectory),
        'losses': np.array(losses)
    }

# Your plotting code here
```

#### 4c. Race on Rosenbrock

Repeat the race on the Rosenbrock function, starting from (-1, -1), 5000 steps.
Tune learning rates for each optimizer (finding good learning rates is part of the exercise).

**Written question:** Which optimizer performed best on each function? Why does Momentum
help with the ill-conditioned quadratic? Why does Adam handle Rosenbrock better than
vanilla GD? Explain in terms of what each optimizer is doing with the gradient
information. (6-8 sentences)

#### 4d. Ablation: What Does Each Piece of Adam Do?

To understand Adam, remove its pieces one at a time:
1. Adam (full)
2. Adam without bias correction (set m_hat = m, v_hat = v)
3. Adam without the second moment (set v_hat = 1) — this should behave like Momentum
4. Adam without the first moment (set m_hat = grads) — this should behave like RMSProp

Run all four variants on the ill-conditioned quadratic and plot convergence. This shows
you what each component contributes.

### Deliverable for Exercise 4

- All four optimizers implemented
- Trajectory comparison plot (ill-conditioned quadratic)
- Convergence comparison plot (both functions)
- Ablation study plots
- Written analysis (6-8 sentences)

---

## Exercise 5: Convexity and Local Minima

**Why this matters:** Understanding convexity tells you whether gradient descent will
find THE answer or just AN answer. For neural networks, it is always just AN answer —
but understanding why that is okay is crucial.

### Task

#### 5a. Convex vs Non-Convex Functions

Plot and classify each function as convex, non-convex, or strictly convex:

```python
# 1. f(x) = x^2           (strictly convex)
# 2. f(x) = |x|           (convex but not strictly)
# 3. f(x) = x^4 - 2x^2   (non-convex: has two local minima)
# 4. f(x) = x^3           (non-convex: no minimum at all)
# 5. f(x) = log(1 + exp(x))  (strictly convex: the softplus function!)
```

For each function:
- Plot it on [-3, 3]
- Pick two points and verify (or violate) the convexity condition:
  f(t*a + (1-t)*b) <= t*f(a) + (1-t)*f(b) for t in [0, 1]
- Draw the line segment between f(a) and f(b) and the function curve

#### 5b. Finding Multiple Local Minima

For f(x) = x^4 - 2x^2:

1. Find all critical points analytically (set f'(x) = 0 and solve)
2. Classify each as local minimum, local maximum, or saddle (using f''(x))
3. Run gradient descent from 5 different starting points: x = -2, -0.5, 0, 0.5, 2
4. Show that different starting points converge to different minima

```python
def f(x):
    return x**4 - 2*x**2

def f_prime(x):
    return 4*x**3 - 4*x

def f_double_prime(x):
    return 12*x**2 - 4

# Find critical points
# f'(x) = 4x^3 - 4x = 4x(x^2 - 1) = 0
# x = 0, x = 1, x = -1

# Classify
for x_crit in [-1, 0, 1]:
    print(f"x = {x_crit}: f''(x) = {f_double_prime(x_crit)}, "
          f"{'minimum' if f_double_prime(x_crit) > 0 else 'maximum'}")

# GD from different starting points
starts = [-2.0, -0.5, 0.01, 0.5, 2.0]  # Note: 0 is a critical point itself
lr = 0.01
for x0 in starts:
    x = x0
    for _ in range(1000):
        x = x - lr * f_prime(x)
    print(f"Start: {x0:+.1f} -> converged to x = {x:.4f}, f(x) = {f(x):.4f}")
```

**Written question:** Neural networks have millions of parameters and highly non-convex
loss surfaces. Why does gradient descent still work? Give two reasons based on what you
have learned. (4-6 sentences)

### Deliverable for Exercise 5

- Plots of all 5 functions with convexity verification
- Critical point analysis for f(x) = x^4 - 2x^2
- GD convergence from different starting points (plot showing all trajectories)
- Written answer about why GD works for neural networks

---

## Exercise 6: Saddle Points and SGD Noise

**Why this matters:** In high-dimensional optimization (which neural network training is),
saddle points are far more common than local minima. Understanding how SGD interacts with
saddle points is essential for understanding why stochastic training works.

### Task

#### 6a. Visualizing a Saddle Point

The function f(x, y) = x^2 - y^2 has a saddle point at the origin.

1. Create a 3D surface plot (using `ax = fig.add_subplot(111, projection='3d')`)
2. Create a contour plot showing the characteristic "X" shape of the contour lines
3. Plot the gradient field (quiver plot) to show how the gradient points away from
   the saddle in the y-direction and toward it in the x-direction

```python
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(16, 6))

# 3D surface
ax1 = fig.add_subplot(121, projection='3d')
x = np.linspace(-2, 2, 100)
y = np.linspace(-2, 2, 100)
X, Y = np.meshgrid(x, y)
Z = X**2 - Y**2
ax1.plot_surface(X, Y, Z, cmap='coolwarm', alpha=0.8)
ax1.set_title('f(x,y) = x^2 - y^2 (Saddle Point)')

# Contour + gradient field
ax2 = fig.add_subplot(122)
ax2.contour(X, Y, Z, levels=20, cmap='coolwarm')
# Add gradient field with quiver
step = 5
Xq, Yq = X[::step, ::step], Y[::step, ::step]
U = 2 * Xq   # df/dx
V = -2 * Yq  # df/dy
ax2.quiver(Xq, Yq, -U, -V, alpha=0.5)  # Negative gradient (descent direction)
ax2.set_title('Contour + Negative Gradient Field')

plt.tight_layout()
plt.savefig('saddle_point.png', dpi=150)
plt.show()
```

#### 6b. GD Gets Stuck, SGD Escapes

Run both pure gradient descent and "stochastic" gradient descent (with added noise)
starting near the saddle point:

```python
def run_optimization(start, lr, n_steps, noise_std=0.0, seed=42):
    """
    Run gradient descent on f(x,y) = x^2 - y^2 with optional noise.

    noise_std > 0 simulates SGD (stochastic noise in gradient estimates).
    """
    np.random.seed(seed)
    x, y = start
    trajectory = [(x, y)]

    for _ in range(n_steps):
        grad_x = 2 * x + np.random.normal(0, noise_std)
        grad_y = -2 * y + np.random.normal(0, noise_std)
        x -= lr * grad_x
        y -= lr * grad_y
        trajectory.append((x, y))

    return np.array(trajectory)

# Pure GD: starts near saddle, gets stuck (or moves very slowly)
traj_gd = run_optimization(start=(0.01, 0.01), lr=0.1, n_steps=50, noise_std=0.0)

# SGD: noise kicks it off the saddle
traj_sgd = run_optimization(start=(0.01, 0.01), lr=0.1, n_steps=50, noise_std=0.1)

# Plot both trajectories on the contour plot
# Your plotting code here
```

#### 6c. The Effect of Noise Magnitude

Run SGD with different noise levels: noise_std = 0.0, 0.01, 0.05, 0.1, 0.5, 1.0.
For each, run 100 times from the same starting point and measure:
1. How many steps to escape the saddle (defined as |x| + |y| > 1)
2. The final function value after 200 steps

Plot: noise level vs average escape time.

**Written question:** There is an optimal amount of noise. Too little and you get stuck.
Too much and you overshoot. How does this relate to batch size in SGD? (A smaller batch
= more noise in the gradient estimate. A larger batch = less noise.) What does this
suggest about batch size selection in practice? (4-6 sentences)

#### 6d. A Higher-Dimensional Saddle Point

Create a function with a saddle point in 10 dimensions:

```python
def high_dim_saddle(x):
    """
    f(x) = sum_{i=0}^{4} x_i^2 - sum_{i=5}^{9} x_i^2

    Saddle at origin: minimum in first 5 dims, maximum in last 5.
    """
    return np.sum(x[:5]**2) - np.sum(x[5:]**2)

def high_dim_saddle_grad(x):
    grad = np.zeros_like(x)
    grad[:5] = 2 * x[:5]
    grad[5:] = -2 * x[5:]
    return grad
```

Start from a random point near the origin. Run GD and SGD. Track the norm of x over
time. Does SGD escape faster? How does the escape direction relate to the maximum
directions of the Hessian?

**Written question:** At the saddle point, the Hessian is diag(2, 2, 2, 2, 2, -2, -2, -2, -2, -2).
The negative eigenvalues correspond to "escape directions." In a real neural network
with millions of parameters, what fraction of eigenvalues would need to be negative for
the critical point to be a saddle point rather than a local minimum? (3-5 sentences)

### Deliverable for Exercise 6

- 3D surface plot and contour/quiver plot of the saddle point
- GD vs SGD trajectory comparison
- Noise magnitude experiment with escape time plot
- High-dimensional saddle point experiment
- Written answers (two questions, 3-6 sentences each)

---

## Grading Criteria

| Criterion | Weight | What We Look For |
|-----------|--------|------------------|
| Correctness | 30% | All gradient computations match PyTorch. Optimizer implementations produce correct results. |
| Understanding | 25% | Chain rule exercise shows clear understanding of backpropagation. Written answers demonstrate conceptual depth. |
| Visualizations | 20% | Clear, informative plots. Trajectories are visible. Contour lines are meaningful. |
| Code Quality | 15% | Well-structured, documented code. Clear variable names. Modular functions. |
| Completeness | 10% | All exercises attempted. All deliverables present. |

---

## Stretch Goals

If you finish early and want to push further:

1. **Implement automatic differentiation**: Build a simple autograd engine that can
   compute gradients through a computation graph. Define a `Value` class with forward
   and backward methods. This is what Andrej Karpathy's micrograd does — implement it
   yourself. Support: add, multiply, power, sigmoid, ReLU.

2. **Implement L-BFGS**: This quasi-Newton method approximates the inverse Hessian using
   gradient history. Implement it and compare to Adam on the Rosenbrock function. L-BFGS
   should converge much faster because it uses curvature information.

3. **Learning rate finder**: Implement the learning rate range test (Smith, 2017). Start
   with a very small learning rate, increase it exponentially over one epoch, and plot loss
   vs learning rate. The optimal learning rate is just before the loss starts increasing.
   Test this on a real neural network (e.g., a small MLP on MNIST).

4. **Gradient noise scale**: For a real neural network training on MNIST, measure the
   ratio of gradient variance to gradient magnitude across training. This "gradient noise
   scale" (McCandlish et al., 2018) predicts the optimal batch size. Implement the
   measurement and see if larger batch sizes correspond to lower noise scales later in
   training.

5. **Hessian spectrum visualization**: For a small trained neural network, compute the top
   eigenvalues of the Hessian using the Lanczos algorithm (via power iteration on the
   Hessian-vector product, which PyTorch can compute efficiently). Plot the spectrum.
   Verify that most eigenvalues are near zero (the loss landscape is nearly flat in most
   directions).
