# Assignment 3: Transfer Learning Project

## Overview

This assignment moves from pedagogical exercises to real-world engineering. You will take a
pretrained model, adapt it to a domain-specific classification task, compare transfer learning
strategies rigorously, and build interpretability into your pipeline with Grad-CAM. This is
the workflow you would follow at DeepMind, Google, or any production ML team when given a new
vision task.

Transfer learning is not optional knowledge — it is the default approach for the vast
majority of real-world computer vision problems. Very few practitioners train from scratch
anymore. The question is never "should I use transfer learning?" but "which pretrained model
and which fine-tuning strategy?"

**Estimated time:** 15-20 hours

---

## Step 0: Choose Your Task

Select a domain-specific image classification problem. The task should be:
- **Not ImageNet:** The whole point is to transfer to a different domain.
- **At least 5 classes.**
- **At least 100 images per class** (more is better, but the whole point is that transfer
  learning works with limited data).
- **Visually interesting:** You should be able to look at Grad-CAM outputs and judge whether
  the model is attending to the right features.

### Recommended Datasets

Pick ONE of the following (or find your own, subject to the criteria above):

1. **Oxford Flowers 102** — 102 flower species, ~8,000 images. Classic fine-grained
   classification task. Available via `torchvision.datasets.Flowers102`.

2. **Food-101** — 101 food categories, 1,000 images per class. Use a subset (e.g., 10-20
   classes) for faster iteration. Available via `torchvision.datasets.Food101`.

3. **Stanford Cars** — 196 car classes (make, model, year). Fine-grained classification.

4. **EuroSAT** — 10 land use classes from satellite imagery. 27,000 images.
   Available via `torchvision.datasets.EuroSAT`.

5. **ISIC Skin Lesion** — Dermatology images, multiple skin condition classes.
   Medical imaging domain with a meaningful real-world application.

6. **Intel Image Classification** — 6 classes (buildings, forest, glacier, mountain, sea,
   street), ~25,000 images. Available on Kaggle.

7. **Your own dataset.** If you have domain expertise in any area, collect or find a dataset
   in that domain. This makes the project far more meaningful.

### Dataset Preparation

Regardless of which dataset you choose:

```python
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as T

# Define transforms (different for train and val/test!)
train_transform = T.Compose([
    T.RandomResizedCrop(224, scale=(0.08, 1.0)),
    T.RandomHorizontalFlip(),
    T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Split into train/val/test (if the dataset does not already have splits)
# Typical split: 70% train, 15% val, 15% test
# Use the SAME random seed for reproducibility
```

**Important:** Always use ImageNet normalization statistics (mean and std) when using
ImageNet-pretrained models. The pretrained features expect inputs in this range.

---

## Step 1: Baseline — Training from Scratch

Before using transfer learning, establish a baseline by training a small CNN from scratch.
This gives you a lower bound to compare against.

### Task

Build a simple CNN (5-8 conv layers) and train it from scratch on your chosen dataset.

```python
class SimpleCNN(nn.Module):
    """Baseline: small CNN trained from scratch."""
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Block 2
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Block 3
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Block 4
            nn.Conv2d(128, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
```

Train for 50 epochs. Record the best validation accuracy. This is your baseline.

---

## Step 2: Strategy 1 — Feature Extraction (Frozen Backbone)

### Task

Use a pretrained ResNet-50 (or EfficientNet-B0) as a frozen feature extractor. Only train
the new classification head.

```python
import torchvision.models as models

def create_feature_extractor(num_classes, backbone='resnet50'):
    """Create a model with frozen pretrained backbone."""
    if backbone == 'resnet50':
        model = models.resnet50(weights='IMAGENET1K_V2')
        # Freeze ALL backbone parameters
        for param in model.parameters():
            param.requires_grad = False
        # Replace classification head
        num_features = model.fc.in_features  # 2048
        model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, num_classes)
        )
    elif backbone == 'efficientnet_b0':
        model = models.efficientnet_b0(weights='IMAGENET1K_V1')
        for param in model.parameters():
            param.requires_grad = False
        num_features = model.classifier[1].in_features  # 1280
        model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, num_classes)
        )
    return model
```

### Training Configuration

```python
# Only trainable parameters are in the new head
trainable_params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.Adam(trainable_params, lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
```

Train for 30 epochs. This should be fast (few trainable parameters).

### Record

- Number of trainable parameters vs total parameters
- Best validation accuracy
- Training time
- Training curve (loss and accuracy)

---

## Step 3: Strategy 2 — Fine-tuning Last Layers

### Task

Unfreeze the last residual block (layer4 in ResNet, or the last few blocks in EfficientNet)
and the classification head. Train with a smaller learning rate.

```python
def create_finetuned_model(num_classes, unfreeze_from='layer4'):
    """Create a model with partial fine-tuning."""
    model = models.resnet50(weights='IMAGENET1K_V2')

    # Freeze everything first
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze layer4
    for param in model.layer4.parameters():
        param.requires_grad = True

    # Replace and unfreeze classification head
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(num_features, num_classes)
    )
    # model.fc parameters are unfrozen by default (newly created)

    return model
```

### Training Configuration

```python
# Use a smaller learning rate than Strategy 1
optimizer = torch.optim.Adam([
    {'params': model.layer4.parameters(), 'lr': 1e-4},
    {'params': model.fc.parameters(), 'lr': 1e-3},
])
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
```

Train for 30 epochs.

### Record

Same metrics as Strategy 1.

---

## Step 4: Strategy 3 — Full Fine-tuning with Differential Learning Rates

### Task

Unfreeze the entire network. Use progressively larger learning rates for later layers.

```python
def create_full_finetune_model(num_classes):
    """Create a model for full fine-tuning with differential LRs."""
    model = models.resnet50(weights='IMAGENET1K_V2')

    # Replace classification head
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(num_features, num_classes)
    )

    # All parameters are trainable
    return model

model = create_full_finetune_model(num_classes)

# Differential learning rates: early layers get smaller LR
optimizer = torch.optim.Adam([
    {'params': model.conv1.parameters(),   'lr': 1e-6},
    {'params': model.bn1.parameters(),     'lr': 1e-6},
    {'params': model.layer1.parameters(),  'lr': 1e-5},
    {'params': model.layer2.parameters(),  'lr': 5e-5},
    {'params': model.layer3.parameters(),  'lr': 1e-4},
    {'params': model.layer4.parameters(),  'lr': 5e-4},
    {'params': model.fc.parameters(),      'lr': 1e-3},
])
```

### Training Protocol

Use a two-phase training approach:

**Phase 1 (Warmup):** Train only the classification head for 5 epochs (freeze backbone).
This gives the random head a chance to "warm up" without corrupting pretrained features.

**Phase 2 (Full fine-tuning):** Unfreeze all layers with differential learning rates.
Train for 25 more epochs with cosine annealing.

```python
# Phase 1: Head warmup
for param in model.parameters():
    param.requires_grad = False
for param in model.fc.parameters():
    param.requires_grad = True
warmup_optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-3)
train(model, warmup_optimizer, epochs=5)

# Phase 2: Full fine-tuning
for param in model.parameters():
    param.requires_grad = True
full_optimizer = torch.optim.Adam([
    {'params': model.conv1.parameters(),   'lr': 1e-6},
    {'params': model.bn1.parameters(),     'lr': 1e-6},
    {'params': model.layer1.parameters(),  'lr': 1e-5},
    {'params': model.layer2.parameters(),  'lr': 5e-5},
    {'params': model.layer3.parameters(),  'lr': 1e-4},
    {'params': model.layer4.parameters(),  'lr': 5e-4},
    {'params': model.fc.parameters(),      'lr': 1e-3},
])
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(full_optimizer, T_max=25)
train(model, full_optimizer, epochs=25, scheduler=scheduler)
```

---

## Step 5: Data Augmentation Study

### Task

For your best-performing strategy, compare augmentation pipelines:

**Pipeline A: Minimal**
```python
T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), T.Normalize(...)])
```

**Pipeline B: Standard**
```python
T.Compose([
    T.RandomResizedCrop(224),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize(...)
])
```

**Pipeline C: Aggressive**
```python
T.Compose([
    T.RandomResizedCrop(224, scale=(0.08, 1.0)),
    T.RandomHorizontalFlip(),
    T.ColorJitter(0.4, 0.4, 0.4, 0.1),
    T.RandomRotation(15),
    T.ToTensor(),
    T.Normalize(...),
    T.RandomErasing(p=0.25),
])
```

Train with each pipeline and compare validation accuracy. How much does augmentation help?

---

## Step 6: Grad-CAM Visualization

### Task

Implement Grad-CAM (or use a library like `pytorch-grad-cam`) to visualize what your model
focuses on when making predictions.

### Implementation

```python
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


class GradCAM:
    """Grad-CAM implementation for model interpretability."""

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register hooks
        self.forward_hook = target_layer.register_forward_hook(self._save_activation)
        self.backward_hook = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, target_class=None):
        """Generate Grad-CAM heatmap for a given input and target class."""
        self.model.eval()
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # Zero gradients
        self.model.zero_grad()

        # Backward pass for target class
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1.0
        output.backward(gradient=one_hot)

        # Compute weights (global average pooling of gradients)
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)

        # Weighted combination of activation maps
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)  # Only positive contributions

        # Normalize to [0, 1]
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        # Upsample to input size
        cam = F.interpolate(cam, size=input_tensor.shape[2:],
                            mode='bilinear', align_corners=False)

        return cam.squeeze().cpu().numpy(), target_class

    def remove_hooks(self):
        self.forward_hook.remove()
        self.backward_hook.remove()
```

### Visualization Requirements

For at least 10 images from your test set (choose a mix of correct and incorrect predictions):

1. **Show the original image, the Grad-CAM heatmap, and the overlay.**

```python
def visualize_gradcam(image_tensor, cam, predicted_class, true_class, class_names):
    """Visualize Grad-CAM results."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Unnormalize the image for display
    img = image_tensor.squeeze().cpu().numpy().transpose(1, 2, 0)
    img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    img = np.clip(img, 0, 1)

    # Original
    axes[0].imshow(img)
    axes[0].set_title(f"True: {class_names[true_class]}")
    axes[0].axis('off')

    # Heatmap
    axes[1].imshow(cam, cmap='jet')
    axes[1].set_title(f"Pred: {class_names[predicted_class]}")
    axes[1].axis('off')

    # Overlay
    axes[2].imshow(img)
    axes[2].imshow(cam, cmap='jet', alpha=0.5)
    correct = "CORRECT" if predicted_class == true_class else "WRONG"
    axes[2].set_title(f"Overlay ({correct})")
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(f'gradcam_{true_class}_{predicted_class}.png',
                dpi=150, bbox_inches='tight')
    plt.show()
```

2. **For correctly classified images:** Does the model attend to the right regions? (e.g., for
   flower classification, is it looking at the petals rather than the background?)

3. **For misclassified images:** What is the model looking at? Can you explain the mistake?
   (e.g., is it attending to a distracting background element?)

4. **Compare Grad-CAM across strategies:** Take the same 5 images and generate Grad-CAM from
   Strategy 1 (frozen), Strategy 2 (partial fine-tune), and Strategy 3 (full fine-tune). Do
   different strategies attend to different regions?

---

## Step 7: Comprehensive Evaluation

### Task

For each strategy (1, 2, 3) and the from-scratch baseline, report:

### Results Table

| Metric | From Scratch | Strategy 1 | Strategy 2 | Strategy 3 |
|--------|-------------|-----------|-----------|-----------|
| Trainable params | | | | |
| Total params | | | | |
| Best val accuracy | | | | |
| Test accuracy | | | | |
| Training time | | | | |
| Epochs to 90% of best | | | | |

### Per-Class Analysis

Create a confusion matrix for your best model. Which classes are most confused? Why?

```python
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

def evaluate_model(model, test_loader, class_names):
    """Full evaluation with confusion matrix and per-class metrics."""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, preds = outputs.max(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Classification report
    print(classification_report(all_labels, all_preds,
                                target_names=class_names))

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.show()
```

---

## Step 8: Report

Write a presentation-quality report (minimum 1,000 words, excluding code) covering:

### Report Structure

**1. Introduction (100 words)**
- What task did you choose and why?
- What is the dataset (classes, size, characteristics)?

**2. Methodology (300 words)**
- What pretrained model did you use and why?
- Describe each transfer learning strategy in your own words.
- What data augmentation pipeline did you use and why?
- What training hyperparameters did you choose and why?

**3. Results (300 words)**
- Complete results table with all four approaches.
- Training curves for all strategies (overlaid on one plot if possible).
- Per-class analysis: which classes are hard and why?

**4. Grad-CAM Analysis (200 words)**
- What does the model focus on for correct predictions?
- What goes wrong for misclassified images?
- Do different strategies produce different attention patterns?

**5. Discussion (200 words)**
- Which strategy worked best and why?
- When would each strategy be appropriate?
- What would you do differently with more data? With less data?
- What are the limitations of your approach?

**6. Conclusion (100 words)**
- Key findings.
- Practical recommendations for someone facing a similar task.

### Visualizations Required

At minimum, your report should include:
- [ ] Sample images from each class
- [ ] Training curves (all strategies, overlaid)
- [ ] Augmentation comparison results
- [ ] Grad-CAM visualizations (at least 10 images)
- [ ] Confusion matrix
- [ ] Grad-CAM comparison across strategies (at least 5 images)

---

## Deliverables

Submit the following:

1. **Code:** Jupyter notebook or Python scripts, fully runnable and well-commented
2. **Report:** PDF, minimum 1,000 words with all required visualizations
3. **Trained models:** Save the best model checkpoint for your best strategy
   ```python
   torch.save({
       'model_state_dict': model.state_dict(),
       'optimizer_state_dict': optimizer.state_dict(),
       'best_val_acc': best_val_acc,
       'epoch': best_epoch,
       'class_names': class_names,
   }, 'best_model.pth')
   ```
4. **Grad-CAM images:** All Grad-CAM visualizations as separate PNG files

### Grading Criteria

| Component | Weight | Criteria |
|-----------|--------|----------|
| Task selection and data prep | 10% | Appropriate dataset, correct splits, proper normalization |
| Three transfer strategies | 25% | Correctly implemented, properly compared, reasonable results |
| Data augmentation study | 10% | At least 3 pipelines compared, results discussed |
| Grad-CAM implementation | 20% | Working Grad-CAM with meaningful visualizations and analysis |
| Evaluation rigor | 15% | Proper train/val/test splits, confusion matrix, per-class analysis |
| Report quality | 20% | Clear writing, complete analysis, presentation-quality figures |

### Common Pitfalls

- **Using the wrong normalization.** If you use an ImageNet-pretrained model, you MUST
  normalize with ImageNet statistics, even if your dataset is very different (medical images,
  satellite imagery). The pretrained features expect inputs in that range.

- **Not using model.eval() during inference.** BatchNorm and Dropout behave differently in
  train vs eval mode. Always call model.eval() for validation and testing.

- **Reporting validation accuracy instead of test accuracy.** You tune hyperparameters on the
  validation set, then report final numbers on the held-out test set. Never tune on the test set.

- **Unfreezing too many layers too early.** If you immediately unfreeze the entire backbone
  with a large learning rate, you will destroy the pretrained features. Always warm up the
  head first.

- **Not setting different learning rates.** When fine-tuning, early layers should have much
  smaller learning rates than later layers. The early features (edges, textures) are already
  good; you do not want to change them much.

- **Skipping the from-scratch baseline.** Without the baseline, you cannot quantify the
  benefit of transfer learning. The comparison is the whole point.

---

## Stretch Goals

1. **Progressive resizing.** Train with images at 128x128 first, then increase to 224x224,
   then to 320x320. Does this improve final accuracy or training speed? Compare to training
   at 224x224 throughout.

2. **Test-time augmentation (TTA).** At test time, apply multiple augmentations to each image
   and average the predictions. How much does TTA improve accuracy?

   ```python
   def predict_with_tta(model, image, n_augments=10):
       model.eval()
       predictions = []
       for _ in range(n_augments):
           augmented = train_transform(image).unsqueeze(0)
           with torch.no_grad():
               pred = F.softmax(model(augmented), dim=1)
           predictions.append(pred)
       # Also add clean prediction
       clean = val_transform(image).unsqueeze(0)
       with torch.no_grad():
           predictions.append(F.softmax(model(clean), dim=1))
       return torch.stack(predictions).mean(dim=0)
   ```

3. **Try a different backbone.** Compare ResNet-50, EfficientNet-B0, EfficientNet-B3, and
   a Vision Transformer (ViT-B/16). Which performs best on your task? Does the relative
   ranking match expectations from ImageNet benchmarks?

4. **Implement Mixup or CutMix.** Add these advanced augmentation strategies to your best
   pipeline. Do they help for transfer learning, or mainly for training from scratch?

5. **Learning rate warmup experiment.** Compare training with and without learning rate warmup
   (linearly increase LR from 0 to target over the first 5 epochs). Does warmup help
   stabilize fine-tuning?

6. **Feature space analysis.** Extract features from the penultimate layer for all test
   images. Use t-SNE or UMAP to visualize the feature space. Compare the feature clusters
   from Strategy 1 vs Strategy 3 — does fine-tuning produce better-separated clusters?

   ```python
   from sklearn.manifold import TSNE

   def extract_features(model, dataloader):
       model.eval()
       features, labels = [], []
       # Hook into the layer before the final FC
       hook_output = []
       def hook(module, input, output):
           hook_output.append(input[0].detach())
       handle = model.fc.register_forward_hook(hook)

       with torch.no_grad():
           for inputs, targets in dataloader:
               _ = model(inputs)
               features.append(hook_output[-1].cpu().numpy())
               labels.append(targets.numpy())
               hook_output.clear()

       handle.remove()
       return np.concatenate(features), np.concatenate(labels)
   ```

7. **Knowledge distillation.** Use your best fine-tuned model (large ResNet-50) as a teacher
   to train a smaller student model (ResNet-18 or MobileNet). Does distillation help the
   student outperform direct fine-tuning of the student architecture?
