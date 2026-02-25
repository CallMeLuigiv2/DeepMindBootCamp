# Assignment 2: Hooks and Debugging Toolkit

## Overview

PyTorch hooks are the instrumentation layer of the framework. They let you observe and modify the forward and backward passes without touching the model code. This is indispensable for debugging, visualization, transfer learning, and implementing techniques like gradient reversal and pruning.

Most PyTorch users never register a hook. After this assignment, you will use them instinctively whenever you need to understand what a model is doing internally.

---

## Part 1: Feature Extraction with Forward Hooks

### Background

Transfer learning often requires extracting features from intermediate layers of a pretrained model. The standard approach is to modify the model architecture, but this is fragile — it requires understanding the model's internals and modifying code you do not own. Forward hooks provide a cleaner solution: attach a hook to any layer and capture its output during inference.

### Task

1. **Load a pretrained ResNet-18** from `torchvision.models`.

2. **Implement a `FeatureExtractor` class** that uses forward hooks to extract outputs from specified layers:
   - The constructor takes a model and a list of layer names (e.g., `['layer1', 'layer2', 'layer3', 'layer4']`).
   - It registers forward hooks on each specified layer.
   - The `__call__` method runs the model on an input and returns a dictionary mapping layer names to their outputs.
   - It has a `close()` method that removes all hooks.

3. **Extract features from all four residual stages** of ResNet-18 for a batch of random images (shape: `[4, 3, 224, 224]`).

4. **Print the shape of each extracted feature map.** Verify they match expected ResNet-18 dimensions:
   - `layer1`: `[4, 64, 56, 56]`
   - `layer2`: `[4, 128, 28, 28]`
   - `layer3`: `[4, 256, 14, 14]`
   - `layer4`: `[4, 512, 7, 7]`

5. **Use the extracted features** to build a simple nearest-neighbor classifier:
   - Extract `layer4` features for a small dataset (100 images from CIFAR-10, resized to 224x224).
   - Global average pool the features to get a 512-dimensional vector per image.
   - Classify test images by finding the training image with the closest feature vector (cosine similarity).
   - Report accuracy.

### Deliverables

- `feature_extractor.py` with the `FeatureExtractor` class.
- Script or notebook demonstrating feature extraction and nearest-neighbor classification.

---

## Part 2: Gradient Flow Visualizer

### Background

Vanishing and exploding gradients are among the most common training failures. The symptoms are subtle: the loss plateaus (vanishing) or becomes NaN (exploding). By the time you notice, you have wasted compute. A gradient flow visualizer catches these problems immediately by showing gradient magnitudes at every layer.

### Task

1. **Build a `GradientFlowVisualizer` class** that:
   - Accepts a model in its constructor.
   - Registers backward hooks on every layer that has parameters (Linear, Conv2d, etc.).
   - After each backward pass, stores the mean and max gradient magnitude for each layer's parameters.
   - Has a `plot()` method that creates a bar chart showing gradient magnitude per layer.

2. **Demonstrate on a deliberately problematic network:**
   - Create a deep network (20+ layers) without any normalization (no BatchNorm, no LayerNorm). Use sigmoid activations to induce vanishing gradients.
   - Train for a few steps on a simple task.
   - Plot the gradient flow. You should see gradients vanishing (approaching zero) in the early layers.

3. **Fix the problem:**
   - Add BatchNorm or switch to ReLU activations.
   - Plot the gradient flow again. The gradients should be more uniform across layers.

4. **Create a comparative visualization:**
   - Side-by-side gradient flow plots for: (a) deep sigmoid network, (b) deep sigmoid + BatchNorm, (c) deep ReLU network, (d) deep ReLU + residual connections.
   - Use matplotlib for plotting. Each plot should have layer names on the x-axis and gradient magnitude on the y-axis (log scale).

### Deliverables

- `gradient_flow.py` with the `GradientFlowVisualizer` class.
- Four gradient flow plots showing the effect of normalization, activation function, and residual connections.
- Written analysis (1-2 paragraphs) explaining what the plots reveal.

---

## Part 3: Gradient Reversal Layer

### Background

Domain adaptation is the problem of training a model on one domain (e.g., synthetic data) and deploying it on another (e.g., real data). The Domain-Adversarial Neural Network (DANN) approach uses a gradient reversal layer: a layer that is the identity in the forward pass but negates gradients in the backward pass.

This forces the feature extractor to learn domain-invariant features: features that are useful for the task but indistinguishable between domains.

### Task

1. **Implement `GradientReversalFunction`** as a custom autograd Function:
   - Forward: identity (return input unchanged).
   - Backward: negate the gradient and multiply by a scaling factor `lambda_val`.

2. **Wrap it in a `GradientReversalLayer` nn.Module.**

3. **Build a DANN architecture:**
   - Feature extractor: 2-3 linear layers with ReLU.
   - Task classifier: 1 linear layer (predicts the class label).
   - Domain classifier: gradient reversal layer followed by 1 linear layer (predicts source vs target domain).

4. **Train on a synthetic domain adaptation problem:**
   - Source domain: MNIST digits 0-4 with Gaussian noise added.
   - Target domain: MNIST digits 0-4 without noise (or vice versa).
   - The task is to classify digits. The domain classifier tries to distinguish source from target.
   - The gradient reversal ensures the feature extractor learns noise-invariant features.

5. **Compare performance:**
   - Train a standard classifier on source domain, test on target domain (baseline).
   - Train the DANN on source + target (labels only from source), test on target domain.
   - Report accuracy for both. The DANN should outperform the baseline on the target domain.

### Deliverables

- `gradient_reversal.py` with the function and module.
- Training script for the DANN.
- Accuracy comparison table (baseline vs DANN).

---

## Part 4: Activation Monitor

### Background

Neural networks can develop pathologies during training that are invisible from the loss curve alone: dead ReLUs (neurons that never activate), saturated sigmoids (neurons stuck at 0 or 1), and exploding activations. An activation monitor catches these problems by logging statistics about every layer's output during training.

### Task

1. **Implement an `ActivationMonitor` class** that:
   - Registers forward hooks on all activation layers (ReLU, Sigmoid, Tanh) and linear/conv layers.
   - For each layer, logs: mean, standard deviation, minimum, maximum, fraction of zeros, fraction of negative values.
   - Stores statistics over time (per epoch or per N steps).
   - Has a `report()` method that prints a summary table.
   - Has a `plot_over_time()` method that shows how statistics evolve during training.

2. **Detect dead ReLUs:**
   - Create a network with ReLU activations and a very high learning rate (to kill neurons quickly).
   - Train for several epochs.
   - Use the activation monitor to identify which ReLU layers have a high fraction of zeros (dead neurons).
   - A layer with >90% zeros is likely dead.

3. **Detect saturated sigmoids:**
   - Create a network with Sigmoid activations and large initial weights.
   - Train for a few steps.
   - Use the activation monitor to identify layers where the mean activation is close to 0 or 1 (saturated).

4. **Create a summary dashboard:**
   - A single visualization that shows activation statistics for all layers over training time.
   - Use a heatmap where rows are layers, columns are training steps, and color intensity represents the fraction of zero activations (for ReLU) or the standard deviation (for any activation).

### Deliverables

- `activation_monitor.py` with the `ActivationMonitor` class.
- Demonstrations of dead ReLU detection and sigmoid saturation detection.
- The summary dashboard visualization.

---

## Part 5: Simple Pruning with Hooks

### Background

Neural network pruning removes unnecessary weights (setting them to zero) to reduce model size and computation. The simplest form is magnitude pruning: zero out the smallest weights. Hooks provide a clean way to apply pruning masks before each forward pass, without modifying the model code.

### Task

1. **Implement a `MagnitudePruner` class** that:
   - Takes a model and a prune ratio (e.g., 0.2 for 20% pruning).
   - For each Linear and Conv2d layer, computes a binary mask: 1 for weights above the magnitude threshold, 0 for weights below.
   - Registers a forward pre-hook on each prunable layer that zeroes out masked weights before every forward pass.
   - Has a method `update_masks()` to recompute masks (call after each epoch to do iterative pruning).
   - Has a method `sparsity_report()` that prints the sparsity of each layer.

2. **Prune a trained network:**
   - Train a small network on MNIST to >97% accuracy.
   - Apply magnitude pruning at increasing ratios: 10%, 20%, 30%, 50%, 70%, 90%.
   - For each ratio, measure test accuracy WITHOUT retraining.
   - Plot: pruning ratio vs accuracy. You should see graceful degradation until a critical point, then a sharp drop.

3. **Prune and fine-tune:**
   - Prune at 50% and fine-tune for 5 epochs.
   - Compare accuracy to: (a) unpruned model, (b) pruned without fine-tuning.
   - Fine-tuning should recover most of the lost accuracy.

4. **Iterative pruning:**
   - Instead of pruning 50% at once, prune 10% per epoch for 5 epochs (reaching ~50% total sparsity).
   - Compare accuracy to one-shot 50% pruning. Iterative pruning should perform better.

### Deliverables

- `pruning.py` with the `MagnitudePruner` class.
- Pruning ratio vs accuracy plot.
- Comparison table: unpruned vs one-shot vs fine-tuned vs iterative.

---

## Evaluation Criteria

### Passing

- The `FeatureExtractor` correctly extracts features from specified layers and cleans up hooks.
- The gradient flow visualizer produces correct plots that clearly show vanishing/exploding gradients.
- The gradient reversal layer has a correct backward pass and produces measurable domain adaptation improvement.
- The activation monitor detects dead ReLUs and saturated sigmoids.
- The pruner applies masks correctly and the accuracy vs sparsity curve is reasonable.
- All hooks are properly removed when they are no longer needed.

### Distinction

All of the above, plus:
- Comparative visualizations are publication-quality.
- Written analysis demonstrates deep understanding of the observed phenomena.
- The DANN experiment shows clear, statistically meaningful improvement.
- The pruning experiments include iterative pruning and fine-tuning.

---

## Stretch Goals

1. **TensorBoard integration:** Modify the `ActivationMonitor` and `GradientFlowVisualizer` to log statistics to TensorBoard using `torch.utils.tensorboard.SummaryWriter`. Log histograms of activations and gradients. This is how monitoring works in real research codebases.

2. **Hook-based attention visualization:** Load a pretrained Vision Transformer (ViT). Use hooks to extract the attention weights from each head of each layer. Visualize the attention maps overlaid on the input image. This reveals what the model attends to.

3. **Gradient-weighted Class Activation Mapping (Grad-CAM):** Using hooks, implement Grad-CAM on a pretrained ResNet:
   - Register forward hooks on the last convolutional layer to capture activations.
   - Register backward hooks to capture gradients of the target class score with respect to those activations.
   - Compute the weighted combination of activation channels.
   - Visualize the resulting heatmap overlaid on the input image.
   - This is a classic explainability technique and a perfect exercise in combining forward and backward hooks.
