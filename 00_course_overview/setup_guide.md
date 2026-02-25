# Environment Setup Guide

This document walks you through setting up a complete deep learning development environment from scratch. Follow every step. Do not skip sections because you think you already have something installed — version mismatches and partial installations are a leading cause of wasted hours.

By the end of this guide, you will have a verified, GPU-accelerated PyTorch environment ready for serious work.

---

## Table of Contents

1. [Python Installation](#1-python-installation)
2. [Virtual Environment Setup](#2-virtual-environment-setup)
3. [PyTorch Installation (CUDA-Enabled)](#3-pytorch-installation-cuda-enabled)
4. [Essential Libraries](#4-essential-libraries)
5. [IDE Setup](#5-ide-setup)
6. [GPU Setup](#6-gpu-setup)
7. [Git Configuration](#7-git-configuration)
8. [Verification](#8-verification)
9. [Troubleshooting](#9-troubleshooting)

---

## 1. Python Installation

You need Python 3.10 or later. We recommend managing Python versions explicitly rather than relying on your system Python.

### Option A: pyenv (Recommended for Linux/macOS)

```bash
# Install pyenv
curl https://pyenv.run | bash

# Add to your shell profile (~/.bashrc, ~/.zshrc, etc.)
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"

# Restart your shell, then install Python
pyenv install 3.11.7
pyenv global 3.11.7

# Verify
python --version
# Should output: Python 3.11.7
```

### Option A (Windows Variant): pyenv-win

```powershell
# Install pyenv-win via PowerShell
Invoke-WebRequest -UseBasicParsing -Uri "https://raw.githubusercontent.com/pyenv-win/pyenv-win/master/pyenv-win/install-pyenv-win.ps1" -OutFile "./install-pyenv-win.ps1"; &"./install-pyenv-win.ps1"

# Restart your terminal, then:
pyenv install 3.11.7
pyenv global 3.11.7

# Verify
python --version
```

### Option B: Conda / Miniconda

If you prefer conda (common in deep learning workflows):

```bash
# Download Miniconda installer from https://docs.conda.io/en/latest/miniconda.html
# Run the installer, then:

conda create -n deeplearning python=3.11 -y
conda activate deeplearning

# Verify
python --version
```

**Important:** If using conda, always ensure your `deeplearning` environment is activated before installing packages or running code. Your terminal prompt should show `(deeplearning)` at the beginning.

---

## 2. Virtual Environment Setup

If you are using pyenv (not conda), create a dedicated virtual environment:

```bash
# Navigate to your project root
cd /path/to/RoadToMlExpert

# Create the virtual environment
python -m venv .venv

# Activate it
# Linux/macOS:
source .venv/bin/activate
# Windows (Git Bash):
source .venv/Scripts/activate
# Windows (PowerShell):
.venv\Scripts\Activate.ps1

# Verify you are in the venv
which python
# Should point to your .venv directory
```

Add `.venv/` to your `.gitignore` file. Virtual environments should never be committed.

---

## 3. PyTorch Installation (CUDA-Enabled)

This is the most important step and the one most likely to go wrong. Pay attention to version compatibility.

### Step 1: Determine Your CUDA Version

If you have an NVIDIA GPU:

```bash
# Check your NVIDIA driver and supported CUDA version
nvidia-smi
```

Look for the "CUDA Version" in the top-right corner of the output. This tells you the maximum CUDA version your driver supports.

### Step 2: Install PyTorch

Go to [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/) and use the configuration selector. Below are common installation commands:

**With CUDA 12.1 (pip):**

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**With CUDA 12.4 (pip):**

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

**With CUDA (conda):**

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

**CPU only (if you have no GPU or are setting up a secondary machine):**

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Step 3: Verify PyTorch and CUDA

```python
import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

    # Quick computation test
    x = torch.randn(1000, 1000, device="cuda")
    y = torch.randn(1000, 1000, device="cuda")
    z = x @ y
    print(f"GPU matrix multiply successful. Result shape: {z.shape}")
```

If `torch.cuda.is_available()` returns `False` but you have an NVIDIA GPU, see the [Troubleshooting](#9-troubleshooting) section.

---

## 4. Essential Libraries

Install all required libraries in one command:

```bash
pip install \
    numpy \
    matplotlib \
    scikit-learn \
    torchvision \
    torchaudio \
    torchtext \
    tensorboard \
    wandb \
    jupyter \
    jupyterlab \
    ipywidgets \
    tqdm \
    pandas \
    seaborn \
    einops \
    tiktoken
```

### What Each Library Is For

| Library | Purpose |
|---------|---------|
| **numpy** | Numerical computing. You will implement things from scratch in NumPy before using PyTorch. |
| **matplotlib** | Plotting. Essential for visualizing loss curves, data distributions, learned features. |
| **scikit-learn** | Classical ML baselines, data preprocessing, evaluation metrics. |
| **torchvision** | Datasets (MNIST, CIFAR, ImageNet), pre-trained models, image transforms. |
| **torchaudio** | Audio datasets and transforms. Used in later modules. |
| **torchtext** | Text datasets and processing utilities. Used for NLP modules. |
| **tensorboard** | Training visualization. Log losses, metrics, histograms, images during training. |
| **wandb** | Experiment tracking. More powerful than TensorBoard for comparing runs and collaboration. |
| **jupyter / jupyterlab** | Interactive notebooks for exploration and prototyping. |
| **ipywidgets** | Interactive widgets for Jupyter. Useful for parameter exploration. |
| **tqdm** | Progress bars. Small but essential for long training runs. |
| **pandas** | Data manipulation. Used for loading and processing tabular data. |
| **seaborn** | Statistical visualization built on matplotlib. Cleaner default aesthetics. |
| **einops** | Tensor operations with readable syntax. Invaluable for attention mechanisms and reshaping. |
| **tiktoken** | Tokenizer library. Used in NLP modules. |

### Weights & Biases Setup

After installing `wandb`, create a free account and authenticate:

```bash
wandb login
```

You will be prompted for an API key. Get it from [wandb.ai/authorize](https://wandb.ai/authorize).

---

## 5. IDE Setup

### VS Code (Recommended)

Visual Studio Code with the right extensions provides an excellent deep learning development experience.

#### Install VS Code

Download from [code.visualstudio.com](https://code.visualstudio.com/).

#### Required Extensions

Open VS Code and install these extensions (Ctrl+Shift+X to open the Extensions panel):

1. **Python** (Microsoft) — Python language support, IntelliSense, linting, debugging.
2. **Jupyter** (Microsoft) — Run Jupyter notebooks directly in VS Code.
3. **Pylance** (Microsoft) — Fast, feature-rich Python language server. Installed automatically with the Python extension.

#### Recommended Extensions

4. **GitLens** — Enhanced Git integration. See blame annotations, compare branches.
5. **Remote - SSH** — If you will be developing on a remote GPU server.
6. **Vim** (vscodevim) — If you use Vim keybindings. Optional but worth learning.

#### VS Code Settings

Add these to your `settings.json` (Ctrl+Shift+P, then "Preferences: Open Settings (JSON)"):

```json
{
    "python.defaultInterpreterPath": "${workspaceFolder}/.venv/Scripts/python",
    "editor.formatOnSave": true,
    "editor.rulers": [88, 120],
    "files.trimTrailingWhitespace": true,
    "jupyter.askForKernelRestart": false,
    "python.analysis.typeCheckingMode": "basic"
}
```

Adjust `python.defaultInterpreterPath` based on your OS:
- **Windows:** `${workspaceFolder}/.venv/Scripts/python`
- **Linux/macOS:** `${workspaceFolder}/.venv/bin/python`
- **Conda:** Use the path shown by `which python` when your environment is active.

#### Selecting the Python Interpreter

1. Open VS Code in your project directory.
2. Press Ctrl+Shift+P and type "Python: Select Interpreter."
3. Choose the interpreter from your virtual environment or conda environment.

---

## 6. GPU Setup

### Option A: Local NVIDIA GPU (Recommended if Available)

A local GPU provides the fastest iteration cycle. Even a mid-range consumer GPU (RTX 3060 with 12GB VRAM, or better) is sufficient for everything in this course.

#### Step 1: Install NVIDIA Drivers

- Download the latest driver for your GPU from [nvidia.com/drivers](https://www.nvidia.com/Download/index.aspx).
- Install and reboot.
- Verify: `nvidia-smi` should display your GPU model and driver version.

#### Step 2: Install CUDA Toolkit

- Download from [developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads).
- Choose the version that matches your PyTorch installation (e.g., CUDA 12.1 or 12.4).
- **Important:** If you installed PyTorch via pip with a `--index-url` specifying a CUDA version, the CUDA runtime is bundled with PyTorch. You may not need to install the full CUDA toolkit separately. However, installing it provides `nvcc` and other tools useful for debugging.

#### Step 3: Install cuDNN

- Download from [developer.nvidia.com/cudnn](https://developer.nvidia.com/cudnn) (requires free NVIDIA Developer account).
- Follow the installation instructions for your OS.
- cuDNN dramatically accelerates convolution and recurrent operations.

#### Step 4: Verify

```bash
# Check driver
nvidia-smi

# Check CUDA compiler (if full toolkit installed)
nvcc --version
```

Then run the PyTorch verification script from Section 3.

### Option B: Google Colab (Free Tier)

If you do not have a local GPU, Google Colab provides free (limited) GPU access.

- Go to [colab.research.google.com](https://colab.research.google.com/).
- Create a new notebook.
- Go to Runtime > Change runtime type > Select "T4 GPU."
- Run `!nvidia-smi` to verify GPU access.

**Limitations of Colab:**
- Sessions time out after inactivity (typically 90 minutes).
- Runtime is not persistent — you lose installed packages and data when the session ends.
- GPU allocation is not guaranteed during high-demand periods.
- Limited to 12--15 GB VRAM (T4 GPU).

**Colab is acceptable for this course** but introduces friction. If you plan to do research beyond this course, invest in a local GPU or a cloud setup.

### Option C: Cloud GPU (AWS, GCP, Lambda Labs, Vast.ai)

For more reliable GPU access than Colab:

- **Lambda Labs** — Simple, ML-focused cloud GPUs. Good pricing for A100/H100 instances.
- **Vast.ai** — Marketplace for renting consumer GPUs. Very affordable.
- **AWS (EC2 p3/p4 instances)** — Enterprise-grade but more complex setup.
- **GCP (Vertex AI or Compute Engine with GPUs)** — Good integration with TensorBoard.

For this course, a single GPU with at least 8GB VRAM is sufficient. You do not need multi-GPU setups until Module 4.

---

## 7. Git Configuration

Every piece of code you write should be version-controlled. No exceptions.

### Initial Setup

```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"

# Recommended settings
git config --global init.defaultBranch main
git config --global core.autocrlf input   # Use 'true' on Windows
git config --global pull.rebase true
```

On Windows, set `core.autocrlf` to `true` instead of `input`:

```bash
git config --global core.autocrlf true
```

### Repository Setup

```bash
cd /path/to/RoadToMlExpert
git init

# Create .gitignore
```

### Recommended .gitignore

Your `.gitignore` should include at minimum:

```
# Python
__pycache__/
*.py[cod]
*.egg-info/
.eggs/
dist/
build/

# Virtual environments
.venv/
venv/
env/

# Jupyter
.ipynb_checkpoints/

# IDE
.vscode/
.idea/

# Data (large files should not be in git)
data/
*.csv
*.h5
*.hdf5
*.pkl
*.pt
*.pth

# OS
.DS_Store
Thumbs.db

# Experiment tracking
wandb/

# Logs
*.log
runs/
```

**Important:** Never commit model checkpoints, datasets, or API keys to git. Use `.gitignore` aggressively.

### Commit Discipline

- Commit frequently with meaningful messages.
- Each commit should represent a single logical change.
- Write commit messages in the imperative: "Add LSTM implementation" not "Added LSTM implementation."
- When you complete a module, tag it: `git tag -a module-01 -m "Completed Module 1: Foundations"`

---

## 8. Verification

Run this verification script to confirm everything is working. Save it as `verify_setup.py` in your project root and run it:

```python
"""
Deep Learning Apprenticeship — Environment Verification Script

Run this script to verify that your development environment is correctly
configured. Every check should pass before you begin the course.

Usage:
    python verify_setup.py
"""

import sys
import importlib


def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    msg = f"  [{status}] {name}"
    if detail:
        msg += f" -- {detail}"
    print(msg)
    return condition


def main():
    print("=" * 60)
    print("  Deep Learning Environment Verification")
    print("=" * 60)
    all_passed = True

    # -----------------------------------------------------------
    # 1. Python version
    # -----------------------------------------------------------
    print("\n1. Python")
    v = sys.version_info
    py_ok = v.major == 3 and v.minor >= 10
    all_passed &= check(
        "Python version >= 3.10",
        py_ok,
        f"Found {v.major}.{v.minor}.{v.micro}",
    )

    # -----------------------------------------------------------
    # 2. Core libraries
    # -----------------------------------------------------------
    print("\n2. Core Libraries")
    core_libs = [
        "numpy",
        "matplotlib",
        "sklearn",
        "pandas",
        "seaborn",
        "tqdm",
        "einops",
    ]
    for lib in core_libs:
        try:
            mod = importlib.import_module(lib)
            ver = getattr(mod, "__version__", "version unknown")
            all_passed &= check(lib, True, ver)
        except ImportError:
            all_passed &= check(lib, False, "NOT INSTALLED")

    # -----------------------------------------------------------
    # 3. PyTorch ecosystem
    # -----------------------------------------------------------
    print("\n3. PyTorch Ecosystem")
    try:
        import torch

        all_passed &= check("torch", True, torch.__version__)
    except ImportError:
        all_passed &= check("torch", False, "NOT INSTALLED")
        print("  *** PyTorch is required. Cannot continue GPU checks. ***")
        torch = None

    torch_extras = ["torchvision", "torchaudio"]
    for lib in torch_extras:
        try:
            mod = importlib.import_module(lib)
            ver = getattr(mod, "__version__", "version unknown")
            all_passed &= check(lib, True, ver)
        except ImportError:
            all_passed &= check(lib, False, "NOT INSTALLED")

    # -----------------------------------------------------------
    # 4. GPU / CUDA
    # -----------------------------------------------------------
    print("\n4. GPU / CUDA")
    if torch is not None:
        cuda_available = torch.cuda.is_available()
        all_passed &= check("CUDA available", cuda_available)

        if cuda_available:
            check(
                "CUDA version",
                True,
                torch.version.cuda,
            )
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1e9
            check("GPU", True, f"{gpu_name} ({gpu_mem:.1f} GB)")

            # Quick computation test
            try:
                x = torch.randn(512, 512, device="cuda")
                y = torch.randn(512, 512, device="cuda")
                z = x @ y
                torch.cuda.synchronize()
                check("GPU computation", True, "Matrix multiply OK")
            except Exception as e:
                all_passed &= check("GPU computation", False, str(e))
        else:
            print("  [INFO] No CUDA GPU detected. CPU-only mode.")
            print("         This is fine if you plan to use Colab or cloud GPUs.")
    else:
        print("  [SKIP] PyTorch not installed, skipping GPU checks.")

    # -----------------------------------------------------------
    # 5. Experiment tracking
    # -----------------------------------------------------------
    print("\n5. Experiment Tracking")
    for lib in ["tensorboard", "wandb"]:
        try:
            mod = importlib.import_module(lib)
            ver = getattr(mod, "__version__", "version unknown")
            all_passed &= check(lib, True, ver)
        except ImportError:
            all_passed &= check(lib, False, "NOT INSTALLED")

    # -----------------------------------------------------------
    # 6. Jupyter
    # -----------------------------------------------------------
    print("\n6. Jupyter")
    for lib in ["jupyter", "jupyterlab", "ipywidgets"]:
        try:
            mod = importlib.import_module(
                lib.replace("-", "_").replace("lab", "lab")
                if lib != "jupyterlab"
                else "jupyterlab"
            )
            ver = getattr(mod, "__version__", "version unknown")
            all_passed &= check(lib, True, ver)
        except ImportError:
            all_passed &= check(lib, False, "NOT INSTALLED")

    # -----------------------------------------------------------
    # Summary
    # -----------------------------------------------------------
    print("\n" + "=" * 60)
    if all_passed:
        print("  All checks passed. Your environment is ready.")
    else:
        print("  Some checks failed. Review the output above and fix")
        print("  any issues before starting the course.")
    print("=" * 60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
```

Run it:

```bash
python verify_setup.py
```

Every check should show `[PASS]`. If any show `[FAIL]`, address them before starting the course material.

---

## 9. Troubleshooting

### PyTorch says CUDA is not available, but I have an NVIDIA GPU

1. **Check your driver:** Run `nvidia-smi`. If this fails, your NVIDIA driver is not installed or not in your PATH.
2. **Version mismatch:** The CUDA version in your PyTorch installation must be compatible with your driver. The driver's CUDA version (shown in `nvidia-smi`) must be >= the CUDA version PyTorch was compiled with.
3. **Wrong PyTorch build:** You may have installed the CPU-only version of PyTorch. Uninstall and reinstall with the correct `--index-url`.
4. **Conda conflicts:** If using conda, ensure you installed `pytorch-cuda` from the correct channels.

### Import errors after installing packages

- Make sure your virtual environment or conda environment is activated.
- Run `which python` (or `where python` on Windows) to verify you are using the correct interpreter.
- Reinstall the problematic package: `pip install --force-reinstall <package>`.

### Jupyter notebook cannot find the kernel

```bash
# Register your virtual environment as a Jupyter kernel
python -m ipykernel install --user --name=deeplearning --display-name="Deep Learning"
```

### Out of GPU memory

- Reduce your batch size.
- Use `torch.cuda.empty_cache()` to free unused memory.
- Use `del tensor_name` to explicitly delete tensors you no longer need.
- Check for accidental gradient accumulation: wrap inference code in `with torch.no_grad():`.
- Use mixed precision training (`torch.cuda.amp`) to halve memory usage.

### Windows-specific issues

- Use Git Bash or WSL2 for a Unix-like shell experience. PowerShell works but some commands differ.
- If using WSL2, install the NVIDIA driver on Windows (not inside WSL). WSL2 will use the Windows driver automatically.
- File paths: Use forward slashes in Python code regardless of OS. Python handles the conversion.

---

## What Not to Do

- **Do not install packages globally.** Always use a virtual environment or conda environment.
- **Do not use Python 2.** It has been end-of-life since 2020.
- **Do not install TensorFlow "just in case."** We use PyTorch exclusively. Installing both can cause CUDA version conflicts.
- **Do not skip the verification step.** Finding out your environment is broken during Week 3 is far more disruptive than spending 10 minutes verifying it now.
- **Do not store datasets in your git repository.** Use a separate `data/` directory listed in `.gitignore`, or download datasets programmatically.

---

Your environment is your workshop. Take the time to set it up properly. A well-configured environment removes friction and lets you focus on what matters: understanding deep learning.
