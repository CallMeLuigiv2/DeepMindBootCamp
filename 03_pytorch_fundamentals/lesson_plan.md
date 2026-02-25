# Module 3: PyTorch Fundamentals — Lesson Plan

**Weeks 4-5 | 6 Sessions | The Craftsman's Toolkit**

---

## Philosophy

PyTorch is not a library you "use." It is an instrument you master. Every serious deep learning
researcher and engineer thinks in tensors, reasons about computational graphs, and can write a
training loop from memory the way a pianist plays scales. By the end of this module, you will
have that fluency.

We do not skim surfaces here. You will understand *why* PyTorch works the way it does, not just
*how* to call its functions. When something breaks at 2 AM during a training run, you will know
exactly where to look.

**Prerequisites:** Strong Python. Comfort with NumPy. Linear algebra fundamentals (Module 1-2).

**Outcome:** You can build any training pipeline from scratch — data loading, model definition,
training loop, evaluation, checkpointing, logging — without copying from a tutorial.

---

## Session 1: Tensors — The Atoms of Deep Learning

**Duration:** 2.5 hours (1h lecture, 1h live-coding, 30min exercises)

### Objectives

By the end of this session, the apprentice will:

1. Understand tensors as typed, device-aware, strided views over contiguous memory — not merely
   "multi-dimensional arrays."
2. Navigate the dtype system and know when precision matters (float32 vs float16 vs bfloat16).
3. Move tensors between CPU and GPU fluently and write device-agnostic code.
4. Master advanced indexing, slicing, and boolean masking.
5. Distinguish between view, reshape, and contiguous — and know when each is required.
6. Predict the output shape of any broadcasting operation before running it.
7. Understand memory layout through strides and storage.

### Time Allocation

| Block | Duration | Content |
|-------|----------|---------|
| 1 | 20 min | What tensors really are: typed + device + strided + autograd-aware. Mental model. |
| 2 | 15 min | dtype system: float32, float64, int64, bool, float16/bfloat16. Casting rules. |
| 3 | 15 min | Device management: CPU vs CUDA. `.to(device)`, `device='cuda:0'`, device-agnostic patterns. |
| 4 | 10 min | Tensor creation: `torch.zeros`, `ones`, `randn`, `arange`, `linspace`, `from_numpy`, `tensor` vs `as_tensor`. |
| 5 | 20 min | Indexing and slicing: basic, advanced (integer array indexing), boolean masking, `torch.where`. |
| 6 | 20 min | Reshaping: `view` vs `reshape` vs `contiguous`. Memory layout, strides, storage. Live demo with `.stride()` and `.storage()`. |
| 7 | 15 min | Broadcasting: the rules, shape inference, common pitfalls. Predict-before-you-run exercises. |
| 8 | 10 min | In-place operations (`add_`, `mul_`): why they exist, why they can silently break autograd. |
| 9 | 25 min | Live-coding: build a batch of image tensors, apply transformations, move to GPU, benchmark. |

### Live-Coding Exercises

1. **Tensor Anatomy Explorer:** Create tensors of various shapes. Inspect `.dtype`, `.device`,
   `.shape`, `.stride()`, `.storage()`, `.is_contiguous()`. Transpose a tensor and observe how
   strides change without data movement.

2. **Broadcasting Prediction Game:** Given two tensor shapes, predict the output shape of their
   addition. Then verify. Do 10 examples, increasing in difficulty.

3. **Image Batch Manipulation:** Load a batch of "images" (random tensors of shape
   `[B, C, H, W]`). Extract all red channels. Flip images horizontally using indexing. Crop the
   center 50%. Do all of this without loops.

4. **Memory Layout Investigation:** Create a tensor, take a transpose, check `is_contiguous()`.
   Try calling `.view()` on it. Observe the error. Fix it with `.contiguous()`. Understand why.

### Key Takeaways

- A tensor is not an array. It is a view over a storage with a shape, stride, dtype, and device.
- `view` requires contiguous memory; `reshape` does not (it may copy). Know which you are using.
- Broadcasting is powerful but dangerous. If you cannot predict the output shape, you do not
  understand your data flow.
- In-place operations save memory but can corrupt the computational graph. Use them only when
  you are certain autograd is not involved.
- Device management is not optional. Every tensor operation requires operands on the same device.

---

## Session 2: Autograd — The Engine of Learning

**Duration:** 2.5 hours (1h lecture, 1h live-coding, 30min exercises)

### Objectives

By the end of this session, the apprentice will:

1. Understand the dynamic computational graph and how it differs from static graphs.
2. Trace how `backward()` computes gradients through the graph via the chain rule.
3. Use `requires_grad`, `.grad`, `detach()`, and `torch.no_grad()` correctly.
4. Understand gradient accumulation and why `zero_grad()` is mandatory.
5. Distinguish leaf tensors from non-leaf tensors and know why it matters.
6. Implement a custom autograd Function with `forward()` and `backward()`.
7. Grasp the concept of gradient checkpointing and when it is useful.

### Time Allocation

| Block | Duration | Content |
|-------|----------|---------|
| 1 | 20 min | Dynamic computational graphs: built on-the-fly, destroyed after backward. Why this matters. |
| 2 | 20 min | The mechanics: `requires_grad=True`, `backward()`, `.grad`. Trace through `y = w*x + b` by hand. |
| 3 | 15 min | Gradient accumulation: why gradients add up, why `zero_grad()` exists, and when accumulation is useful. |
| 4 | 15 min | `detach()` and `torch.no_grad()`: stopping gradient flow. Use cases: inference, freezing layers. |
| 5 | 15 min | Leaf vs non-leaf tensors: `.is_leaf`, `.retain_grad()`, `.grad_fn`. Why non-leaf gradients are discarded. |
| 6 | 10 min | `retain_graph=True`: when you need it, why it is expensive, how to avoid it. |
| 7 | 20 min | Custom autograd Functions: `torch.autograd.Function`, `ctx.save_for_backward`. Implement Swish. |
| 8 | 10 min | Gradient checkpointing concept: trade compute for memory. When models do not fit in GPU memory. |
| 9 | 25 min | Live-coding: build a linear regression from raw tensors and autograd. No nn.Module yet. |

### Live-Coding Exercises

1. **Manual Backprop Trace:** Define `y = w * x + b`, then `loss = (y - target)**2`. Call
   `loss.backward()`. Print `w.grad`, `b.grad`. Verify by hand using calculus. Draw the
   computational graph on paper (or whiteboard).

2. **Gradient Accumulation Experiment:** Run backward twice without zeroing gradients. Observe
   that gradients double. Then demonstrate the correct pattern with `zero_grad()`.

3. **Custom Activation Function:** Implement the Swish activation (`x * sigmoid(x)`) as a
   `torch.autograd.Function`. Verify gradients match using `torch.autograd.gradcheck`.

4. **Linear Regression from Scratch:** Using only tensors and autograd (no nn.Module, no
   optimizer), train a linear model on synthetic data. Manual SGD: `w -= lr * w.grad`.

### Key Takeaways

- PyTorch builds a new computational graph on every forward pass. This is what makes it
  "dynamic" — your graph can change based on data or control flow.
- `backward()` destroys the graph by default. If you need to call it twice, use
  `retain_graph=True`, but ask yourself why you need to.
- Gradients accumulate. This is by design (useful for gradient accumulation across mini-batches),
  but you must zero them explicitly before each optimization step.
- `torch.no_grad()` is for inference. `detach()` is for stopping gradient flow through a
  specific tensor. They are not interchangeable.
- If you can implement a custom autograd Function, you truly understand how PyTorch computes
  gradients.

---

## Session 3: nn.Module — Building Blocks

**Duration:** 2.5 hours (1h lecture, 1h live-coding, 30min exercises)

### Objectives

By the end of this session, the apprentice will:

1. Understand nn.Module as the fundamental abstraction for neural network components.
2. Implement custom modules with proper `__init__` and `forward` methods.
3. Navigate the parameter system: `parameters()`, `named_parameters()`, `children()`.
4. Know the difference between parameters and buffers.
5. Use `state_dict()` for saving and loading models correctly.
6. Build nested, composable module hierarchies.
7. Use common layers: Linear, Conv2d, BatchNorm, Dropout, and understand their parameters.

### Time Allocation

| Block | Duration | Content |
|-------|----------|---------|
| 1 | 20 min | The Module abstraction: `__init__`, `forward`, why you never call `forward()` directly. |
| 2 | 15 min | Parameter registration: `nn.Parameter`, why it matters, what `parameters()` returns. |
| 3 | 15 min | Common layers: `Linear` (weight shapes, bias), `Conv2d` (kernel_size, stride, padding, output size formula). |
| 4 | 10 min | `BatchNorm2d`, `Dropout`, `ReLU`, `LayerNorm` — what they do in train vs eval mode. |
| 5 | 15 min | `Sequential` vs custom modules: when each is appropriate. `ModuleList` and `ModuleDict`. |
| 6 | 15 min | Buffers vs parameters: `register_buffer` for non-learnable state (e.g., running mean in BatchNorm). |
| 7 | 15 min | `state_dict()`, `load_state_dict()`: saving, loading, strict mode, handling mismatched keys. |
| 8 | 10 min | `train()` vs `eval()`: what changes (Dropout, BatchNorm). A common source of bugs. |
| 9 | 25 min | Live-coding: build a CNN from scratch for image classification. |

### Live-Coding Exercises

1. **Module Anatomy:** Build a simple `nn.Module` with two linear layers and a ReLU. Print all
   parameters, their shapes, and their names. Count total parameters.

2. **Custom Module with Buffers:** Build a module that normalizes inputs using a running mean
   and variance stored as buffers. Save and load the model. Verify buffers persist.

3. **CNN Architecture:** Build a CNN for 32x32 images: Conv2d -> BatchNorm -> ReLU -> MaxPool,
   repeated twice, then flatten and two Linear layers. Calculate the output shape at each layer
   by hand, then verify.

4. **Module Surgery:** Load a pretrained model (or a model you saved). Freeze all layers except
   the last one. Replace the last layer. Verify only the new layer has `requires_grad=True`.

### Key Takeaways

- `nn.Module` is not magic. It is a container that tracks parameters and provides a `forward`
  method. Understanding this demystifies everything.
- Always call `model(x)`, never `model.forward(x)`. The `__call__` method runs hooks and other
  machinery.
- `train()` and `eval()` toggle behavior for Dropout and BatchNorm. Forgetting `model.eval()`
  during inference is one of the most common bugs in deep learning code.
- `state_dict()` is the correct way to save models. Save the state dict, not the model object.
- Parameters are tensors with `requires_grad=True` that are registered with the module. Buffers
  are tensors that are part of the state but are not optimized.

---

## Session 4: Data Pipeline

**Duration:** 2.5 hours (1h lecture, 1h live-coding, 30min exercises)

### Objectives

By the end of this session, the apprentice will:

1. Implement custom Dataset classes (both map-style and iterable-style).
2. Configure DataLoader for optimal performance: batching, shuffling, num_workers, pin_memory.
3. Write custom `collate_fn` for variable-length data.
4. Apply data augmentation using torchvision transforms.
5. Use torchvision.datasets for standard benchmarks.
6. Handle common data formats: images, CSV, JSON, custom binary.

### Time Allocation

| Block | Duration | Content |
|-------|----------|---------|
| 1 | 15 min | The Dataset abstraction: `__len__`, `__getitem__`. Map-style vs iterable-style. |
| 2 | 15 min | DataLoader: batching, shuffling, drop_last. What happens under the hood. |
| 3 | 15 min | Performance: `num_workers` (multiprocessing), `pin_memory` (page-locked memory for GPU), `prefetch_factor`. |
| 4 | 15 min | `collate_fn`: default behavior, custom collation for variable-length sequences, padding. |
| 5 | 15 min | Transforms: `torchvision.transforms` (Compose, ToTensor, Normalize, RandomHorizontalFlip, etc.). The v2 API. |
| 6 | 10 min | `torchvision.datasets`: CIFAR-10, MNIST, ImageNet. Quick setup. |
| 7 | 10 min | Custom data formats: loading images from directories, CSV files, handling large datasets. |
| 8 | 10 min | Common pitfalls: forgetting to shuffle training data, data leakage in transforms, workers on Windows. |
| 9 | 25 min | Live-coding: build a complete data pipeline for a custom image dataset. |

### Live-Coding Exercises

1. **Custom Dataset:** Create a Dataset class that generates synthetic regression data
   (y = 3x + noise). Use it with a DataLoader. Iterate through batches.

2. **Image Dataset with Augmentation:** Build a Dataset for images stored in folders. Apply
   augmentation (random crop, horizontal flip, color jitter) for training, only resize + normalize
   for validation. Show the difference.

3. **Variable-Length Collation:** Create a dataset of variable-length sequences (simulating NLP
   data). Write a `collate_fn` that pads sequences to the maximum length in the batch.

4. **DataLoader Benchmarking:** Compare data loading speed with `num_workers=0, 2, 4, 8`. Measure
   with `pin_memory=True` vs `False` when transferring to GPU. Find the optimal configuration.

### Key Takeaways

- The Dataset defines *what* data you have. The DataLoader defines *how* you iterate over it.
- `num_workers > 0` uses multiprocessing to load data in parallel. On Windows, this requires
  the `if __name__ == '__main__'` guard. Start with `num_workers=2` and increase.
- `pin_memory=True` pre-allocates page-locked memory, making CPU-to-GPU transfers faster. Always
  use it when training on GPU.
- Data augmentation is applied per-sample, on-the-fly. Different augmentation for training vs
  validation is critical. Never augment validation data (except resize/normalize).
- The `collate_fn` is where you handle batches of non-uniform data. Master it for NLP and any
  task with variable-length inputs.

---

## Session 5: The Training Loop

**Duration:** 3 hours (1h lecture, 1.5h live-coding, 30min exercises)

### Objectives

By the end of this session, the apprentice will:

1. Write the canonical training loop from memory: forward, loss, backward, step, zero_grad.
2. Choose the correct loss function for the task (MSE, CrossEntropy, BCE, etc.).
3. Understand and configure optimizers: SGD (with momentum), Adam, AdamW.
4. Implement learning rate scheduling: StepLR, CosineAnnealing, OneCycleLR, ReduceLROnPlateau.
5. Write a proper validation loop (no gradients, eval mode).
6. Implement early stopping and model checkpointing.
7. Add gradient clipping for training stability.

### Time Allocation

| Block | Duration | Content |
|-------|----------|---------|
| 1 | 20 min | The canonical loop: every line explained. Why the order matters. |
| 2 | 15 min | Loss functions: MSE for regression, CrossEntropyLoss (includes softmax!) for classification, BCEWithLogitsLoss for binary/multi-label. |
| 3 | 15 min | Optimizers: SGD (momentum, weight decay), Adam (adaptive lr), AdamW (decoupled weight decay). |
| 4 | 15 min | Learning rate schedulers: StepLR, CosineAnnealingLR, OneCycleLR, ReduceLROnPlateau. When to use each. |
| 5 | 15 min | Validation loop: `model.eval()`, `torch.no_grad()`, computing metrics. Why you must not call backward. |
| 6 | 10 min | Early stopping: tracking best validation loss, patience, restoring best model. |
| 7 | 10 min | Model checkpointing: what to save (model, optimizer, epoch, loss, scheduler). |
| 8 | 10 min | Gradient clipping: `clip_grad_norm_` vs `clip_grad_value_`. When and why. |
| 9 | 40 min | Live-coding: train a model from scratch with all the above. |

### Live-Coding Exercises

1. **The Canonical Loop:** Write the training loop from scratch. Train a simple network on
   synthetic data. Print loss every epoch. Then refactor to add a validation loop.

2. **Loss Function Exploration:** Train the same model with MSE vs CrossEntropy on a
   classification task. Observe the difference. Demonstrate the common mistake of applying
   softmax before CrossEntropyLoss.

3. **Optimizer Comparison:** Train with SGD (no momentum), SGD (with momentum), Adam, and AdamW.
   Plot training curves. Observe convergence speed differences.

4. **Learning Rate Finder:** Implement a simple LR range test: train for a few hundred steps with
   exponentially increasing learning rate. Plot loss vs learning rate. Find the optimal LR.

5. **Full Training Script:** Combine everything: training loop, validation, early stopping,
   checkpointing, LR scheduling. This becomes the template for all future work.

### Key Takeaways

- The training loop order is: `zero_grad` -> `forward` -> `loss` -> `backward` -> `step`. This
  is not arbitrary. `zero_grad` clears stale gradients. `forward` builds the graph. `loss`
  computes the scalar. `backward` fills `.grad`. `step` updates parameters.
- `CrossEntropyLoss` expects raw logits, not probabilities. It applies `log_softmax` internally.
  Applying softmax before it is a bug that silently degrades performance.
- AdamW is usually the right default optimizer for deep learning. SGD with momentum can
  generalize better but requires more tuning.
- Always validate after each epoch. The validation loss tells you when you are overfitting.
  The training loss alone tells you almost nothing about generalization.
- Save checkpoints that include the optimizer state and scheduler state. You need these to
  resume training correctly.

---

## Session 6: Putting It All Together

**Duration:** 3 hours (30min lecture, 2h project work, 30min review)

### Objectives

By the end of this session, the apprentice will:

1. Build a complete end-to-end pipeline: data loading, model, training, evaluation, saving.
2. Add TensorBoard (or W&B) logging to a training run.
3. Write reproducible training code with proper seeding and deterministic settings.
4. Debug common training issues: NaN loss, exploding gradients, learning rate too high/low.
5. Have a reusable training template for all future projects.

### Time Allocation

| Block | Duration | Content |
|-------|----------|---------|
| 1 | 15 min | The end-to-end checklist: every component of a training pipeline. |
| 2 | 15 min | Logging: TensorBoard (`SummaryWriter`, scalars, images, histograms). W&B overview. |
| 3 | 10 min | Reproducibility: `torch.manual_seed`, `torch.cuda.manual_seed_all`, `torch.backends.cudnn.deterministic`, `torch.use_deterministic_algorithms`. |
| 4 | 10 min | Common bugs and debugging: NaN gradients, shape mismatches, forgetting eval(), wrong loss function. |
| 5 | 120 min | Project work: build a full pipeline for CIFAR-10 or Fashion-MNIST with all components. |
| 6 | 10 min | Code review and discussion: what went well, what was hard, what to improve. |

### Live-Coding Exercises

1. **TensorBoard Integration:** Add logging to the training loop from Session 5. Log training
   loss, validation loss, learning rate, and sample predictions. Launch TensorBoard and inspect.

2. **Reproducibility Test:** Train a model twice with the same seed. Verify identical results.
   Then train without seeding and observe the difference.

3. **Debugging Challenge:** Given a broken training script (with 5 deliberate bugs), find and
   fix all bugs. Bugs include: missing `model.eval()`, softmax before CrossEntropyLoss, not
   zeroing gradients, wrong input shape, and data on wrong device.

4. **Full Pipeline:** Build the complete pipeline from scratch. This is the capstone exercise.

### Key Takeaways

- A training pipeline is a system, not a script. Each component (data, model, optimizer,
  scheduler, logger, checkpointer) should be modular and testable.
- Reproducibility is not optional. Set seeds, use deterministic algorithms, log your
  hyperparameters, save your code version (git hash).
- The most common bugs in deep learning are silent: wrong loss function, missing eval mode,
  data on the wrong device. They do not crash — they just produce bad results.
- You now have every tool needed to train any model in PyTorch. The rest of this course is about
  choosing the right architectures, data, and training strategies. The engineering is solved.

---

## Weekly Schedule

### Week 4

| Day | Session | Focus |
|-----|---------|-------|
| Day 1 | Session 1 | Tensors |
| Day 2 | Practice | Assignment 01: Tensor Operations |
| Day 3 | Session 2 | Autograd |
| Day 4 | Practice | Assignment 02: Autograd Deep Dive |
| Day 5 | Session 3 | nn.Module |

### Week 5

| Day | Session | Focus |
|-----|---------|-------|
| Day 1 | Session 4 | Data Pipeline |
| Day 2 | Session 5 | The Training Loop |
| Day 3 | Practice | Begin Assignment 03: Full Pipeline |
| Day 4 | Session 6 | Putting It All Together |
| Day 5 | Practice | Complete Assignment 03, code review |

---

## Assessment Criteria

The apprentice has completed this module when they can:

1. Write any tensor operation without consulting documentation.
2. Explain what `backward()` does to any computational graph.
3. Build a custom `nn.Module` with proper parameter registration.
4. Set up a DataLoader with optimal performance settings.
5. Write a complete training loop from memory — including validation, checkpointing, and logging.
6. Debug common training failures by reasoning about what each component does.

If any of these feel shaky, repeat the relevant session before moving on.
