# Assignment 03: Build a Complete Training Pipeline from Scratch

**Module 3, Sessions 3-6 | Estimated time: 8-12 hours**

---

## Objective

This is the capstone assignment for the PyTorch Fundamentals module. You will build a complete,
production-quality training pipeline from scratch. No starter code. No templates. Every line
of code will be yours, and you will understand every line.

When you finish this assignment, you will have a reusable training template that you can adapt
for any future project. This is the foundation of your engineering practice.

---

## The Task

Train a convolutional neural network on **CIFAR-10** to achieve **at least 85% test accuracy**
using a simple architecture (no pretrained models, no architectures with more than ~500K
parameters).

CIFAR-10 contains 60,000 32x32 color images in 10 classes: airplane, automobile, bird, cat,
deer, dog, frog, horse, ship, truck. 50,000 training images and 10,000 test images.

85% on CIFAR-10 with a small CNN requires that every component of your pipeline works correctly:
data augmentation, proper normalization, good architecture, appropriate optimizer and learning
rate schedule, and no bugs.

---

## Requirements

Your submission must include the following files:

### 1. `model.py` — The CNN Architecture

Build a CNN using `nn.Module`. Requirements:

- At least 3 convolutional layers with increasing channel counts (e.g., 32 -> 64 -> 128).
- Use `BatchNorm2d` after each convolutional layer.
- Use `MaxPool2d` or strided convolutions for downsampling.
- Use `Dropout` for regularization.
- A fully connected classifier head.
- A method to count total parameters.
- Total parameter count should be under 500K (forces you to be efficient).

```python
# Skeleton (fill in the details)
class CIFAR10Net(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # Define layers here

    def forward(self, x):
        # Define forward pass here
        pass

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
```

Document your architecture choices: Why these layer sizes? Why this depth? How did you calculate
the spatial dimensions at each layer?

### 2. `dataset.py` — Data Pipeline

Requirements:

- Use `torchvision.datasets.CIFAR10` for downloading and loading data.
- Split the training set into train (45,000) and validation (5,000) using `random_split` with
  a fixed seed.
- Define separate transforms for training and validation:
  - **Training:** `RandomCrop(32, padding=4)`, `RandomHorizontalFlip()`, `ToTensor()`,
    `Normalize()`. You may add more augmentation.
  - **Validation/Test:** `ToTensor()`, `Normalize()` only.
- Compute or use the CIFAR-10 channel-wise mean and standard deviation for normalization:
  `mean = (0.4914, 0.4822, 0.4465)`, `std = (0.2470, 0.2435, 0.2616)`.
- Return `DataLoader` objects with appropriate settings:
  - `batch_size` from arguments.
  - `num_workers` from arguments (default 2, with proper Windows compatibility).
  - `pin_memory=True` when using GPU.
  - `shuffle=True` for training, `False` for validation and test.
  - `drop_last=True` for training.

```python
def get_data_loaders(batch_size=128, num_workers=2, data_dir='./data', pin_memory=True):
    """
    Returns: train_loader, val_loader, test_loader
    """
    pass
```

### 3. `train.py` — The Training Script

This is the main script. It must be runnable from the command line:

```bash
python train.py --epochs 100 --batch-size 128 --lr 0.01 --weight-decay 1e-4
```

Requirements:

**Command-line arguments (using argparse):**
- `--epochs` (default: 100)
- `--batch-size` (default: 128)
- `--lr` (default: 0.01)
- `--weight-decay` (default: 1e-4)
- `--optimizer` (choices: sgd, adam, adamw; default: adamw)
- `--scheduler` (choices: cosine, step, onecycle, plateau; default: cosine)
- `--seed` (default: 42)
- `--save-dir` (default: ./checkpoints)
- `--log-dir` (default: ./runs)
- `--num-workers` (default: 2)
- `--patience` (default: 15, for early stopping)
- `--grad-clip` (default: 1.0, max gradient norm; 0 to disable)

**The training loop must include:**

1. **Reproducibility:** Set all random seeds at the start.
2. **Device management:** Detect and use GPU if available.
3. **Model instantiation:** Create the model and move to device. Print parameter count.
4. **Optimizer selection:** Based on the `--optimizer` argument.
5. **Scheduler selection:** Based on the `--scheduler` argument.
6. **Loss function:** `nn.CrossEntropyLoss`.
7. **Training epoch function:** Iterate over training batches, compute loss, backward, clip, step.
   Return average loss and accuracy.
8. **Validation epoch function:** Iterate over validation batches with `model.eval()` and
   `torch.no_grad()`. Return average loss and accuracy.
9. **TensorBoard logging:** Log train loss, val loss, train accuracy, val accuracy, and learning
   rate at every epoch.
10. **Model checkpointing:** Save the best model (by validation loss) and the latest model. The
    checkpoint must include: epoch, model state dict, optimizer state dict, scheduler state dict,
    best validation loss, and best validation accuracy.
11. **Early stopping:** Stop training if validation loss does not improve for `--patience` epochs.
12. **Final evaluation:** After training, load the best model and evaluate on the test set. Print
    the test accuracy.
13. **Summary:** At the end of training, print a summary: best epoch, best validation accuracy,
    test accuracy, total training time.

**Code structure guidelines:**

```python
def train_one_epoch(model, loader, criterion, optimizer, device, grad_clip=0):
    """Train for one epoch. Return average loss and accuracy."""
    pass

def validate(model, loader, criterion, device):
    """Validate. Return average loss and accuracy."""
    pass

def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()

    train_loader, val_loader, test_loader = get_data_loaders(...)
    model = CIFAR10Net().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(args, model)
    scheduler = get_scheduler(args, optimizer, train_loader)

    # Training loop
    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(...)
        val_loss, val_acc = validate(...)
        scheduler_step(...)
        log_metrics(...)
        checkpoint_if_best(...)
        if early_stopping(...):
            break

    # Final evaluation
    evaluate_on_test(...)
    print_summary(...)

if __name__ == '__main__':
    main()
```

### 4. `utils.py` — Utility Functions

Extract reusable utilities:

- `set_seed(seed)` — Set all random seeds.
- `get_device()` — Return the best available device.
- `count_parameters(model)` — Return the number of trainable parameters.
- `EarlyStopping` class.
- Any other helper functions you find useful.

### 5. `requirements.txt`

List all required packages with version numbers:

```
torch>=2.0.0
torchvision>=0.15.0
tensorboard>=2.12.0
```

Include only what is actually needed. Do not dump your entire environment.

### 6. `evaluate.py` — Standalone Evaluation Script

A script that loads a saved checkpoint and evaluates it on the test set:

```bash
python evaluate.py --checkpoint ./checkpoints/best_model.pth
```

Print:
- Test accuracy (overall)
- Per-class accuracy
- Confusion matrix (as a printed table)

---

## Accuracy Target

**Minimum: 85% test accuracy.**

This is achievable with:
- A well-designed small CNN (~300K-500K parameters)
- Proper data augmentation (RandomCrop + HorizontalFlip at minimum)
- AdamW or SGD with momentum
- Cosine annealing or OneCycleLR
- 80-100 epochs of training

If you cannot reach 85%, debug methodically:
1. Is your model training at all? (Does training loss decrease?)
2. Is your model overfitting? (Training accuracy >> validation accuracy?)
3. Are your transforms correct? (Visualize augmented images.)
4. Is your normalization correct? (Wrong mean/std will hurt.)
5. Is your learning rate appropriate? (Try a range test.)

---

## Evaluation Criteria

### Correctness (40%)

- The training script runs without errors.
- All components work together: data loading, model, training loop, checkpointing, logging.
- The model achieves at least 85% test accuracy.
- Checkpoints can be loaded and evaluation reproduced.
- Setting the same seed produces the same results.

### Code Quality (30%)

- Clean, readable code with meaningful variable names.
- Functions are modular and single-purpose.
- Proper error handling (e.g., check that checkpoint file exists before loading).
- Comments explain *why*, not *what* (the code should be clear enough that *what* is obvious).
- No dead code, no commented-out experiments.
- Follows Python conventions (PEP 8).

### Engineering Practices (20%)

- Proper argument parsing with sensible defaults.
- TensorBoard logging works and produces useful visualizations.
- Checkpointing saves everything needed to resume training.
- Early stopping is correctly implemented.
- The code is device-agnostic (works on CPU and GPU).
- The `if __name__ == '__main__'` guard is present (required for Windows multiprocessing).

### Understanding (10%)

- Architecture choices are documented and justified.
- You can explain every line of the training loop.
- The README or comments explain what the code does and how to use it.

---

## Submission Checklist

Before submitting, verify:

```
[ ] python train.py runs without errors (with default arguments)
[ ] Training reaches at least 85% test accuracy
[ ] python evaluate.py --checkpoint ./checkpoints/best_model.pth works
[ ] TensorBoard logs are generated in ./runs/
[ ] Setting --seed 42 produces identical results across two runs
[ ] The code works on CPU (even if slower)
[ ] requirements.txt lists all dependencies
[ ] Total model parameters < 500K
[ ] No hardcoded absolute paths (use relative paths or arguments)
[ ] No API keys, passwords, or personal data in the code
```

---

## Stretch Goals

### Stretch 1: Weights and Biases Integration

Add W&B logging alongside (or instead of) TensorBoard:

```bash
python train.py --logger wandb --wandb-project cifar10-training
```

Log: loss curves, accuracy curves, learning rate, sample predictions (as W&B Images),
confusion matrix, model architecture summary.

### Stretch 2: Mixed Precision Training

Add an `--amp` flag that enables Automatic Mixed Precision:

```bash
python train.py --amp
```

Use `torch.amp.autocast` and `torch.amp.GradScaler`. Compare training speed with and
without AMP. Report the speedup.

### Stretch 3: Learning Rate Range Test

Implement a learning rate range test (Smith, 2017):

```bash
python lr_finder.py --min-lr 1e-7 --max-lr 10 --num-steps 200
```

Plot loss vs learning rate. Use this to select the optimal learning rate for your final
training run.

### Stretch 4: Hyperparameter Search

Write a script that searches over:
- Learning rate: [1e-4, 1e-3, 1e-2, 1e-1]
- Weight decay: [0, 1e-4, 1e-3, 1e-2]
- Optimizer: [sgd, adamw]

Run each combination for 30 epochs. Report results in a table. Identify the best configuration.

### Stretch 5: Push to 90%+

Using the same parameter budget (<500K), try to push test accuracy above 90%. Techniques to
explore:
- Cutout / CutMix / MixUp data augmentation
- Label smoothing
- Squeeze-and-Excitation blocks
- Depthwise separable convolutions (MobileNet-style)
- Knowledge distillation from a larger model

Document what worked and what did not.

### Stretch 6: Training Resume

Add a `--resume` flag:

```bash
python train.py --resume ./checkpoints/latest_model.pth
```

This should load the checkpoint and continue training from where it left off, with the correct
epoch, optimizer state, scheduler state, and best validation loss. Verify that resuming produces
the same results as uninterrupted training (with the same seed).

---

## Architecture Hints

If you are stuck on the architecture, here is a reasonable starting point. Do not copy this
verbatim; understand it and modify it.

```
Input: (B, 3, 32, 32)

Block 1: Conv2d(3, 32, 3, padding=1) -> BN -> ReLU -> Conv2d(32, 32, 3, padding=1) -> BN -> ReLU -> MaxPool(2)
  Output: (B, 32, 16, 16)

Block 2: Conv2d(32, 64, 3, padding=1) -> BN -> ReLU -> Conv2d(64, 64, 3, padding=1) -> BN -> ReLU -> MaxPool(2)
  Output: (B, 64, 8, 8)

Block 3: Conv2d(64, 128, 3, padding=1) -> BN -> ReLU -> Conv2d(128, 128, 3, padding=1) -> BN -> ReLU -> MaxPool(2)
  Output: (B, 128, 4, 4)

Flatten: (B, 128 * 4 * 4) = (B, 2048)
Dropout(0.3)
Linear(2048, 256) -> ReLU -> Dropout(0.3)
Linear(256, 10)

Parameter count: approximately 400K
```

This architecture should comfortably reach 85%+ with proper training.

---

## Timeline Suggestion

| Phase | Hours | What to do |
|-------|-------|------------|
| 1 | 1-2 | Write `model.py` and `dataset.py`. Verify shapes, test data loading. |
| 2 | 2-3 | Write the core training loop in `train.py`. Get basic training working (loss decreases). |
| 3 | 1-2 | Add validation, logging, checkpointing, early stopping. |
| 4 | 1-2 | Add argument parsing, seed management, all the engineering. |
| 5 | 1-2 | Tune to reach 85%. Debug if needed. |
| 6 | 1 | Write `evaluate.py`. Clean up code. Verify reproducibility. |

---

*This assignment is the bridge between learning PyTorch and using PyTorch. Every deep learning
project you will ever work on — from research prototypes to production systems — follows this
same structure. Build it well, and it will serve you for years.*
