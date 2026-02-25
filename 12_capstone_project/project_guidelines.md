# Module 12 -- Capstone Project: Guidelines

## Overview

This is the final challenge. Everything you have learned across eleven modules -- mathematical foundations, PyTorch mechanics, convolutional and recurrent architectures, Transformers, generative models, performance engineering, and the discipline of reading research papers -- converges here. The capstone is not an exercise. It is a project. You will define a problem, design a solution, implement it, run experiments, analyze results, and communicate your findings. The standard you should hold yourself to: could this be submitted as a workshop paper or presented credibly to a team of ML researchers?

The capstone spans 2-3 weeks and demands sustained, focused effort. You will write a proposal, build a codebase, run experiments that may fail and require rethinking, produce a conference-style report, and deliver a presentation. This mirrors the actual workflow of a research engineer at a lab like DeepMind. There are no shortcuts. The quality of your capstone is the single most honest measure of what you have internalized from this course.

Do not pick a project that is too easy. Do not pick one that is impossible. Pick one that stretches you -- one where you are not entirely sure you can pull it off, but you have a credible plan to try.

---

## Timeline

### Week 1: Proposal and Literature Review

**Days 1-2: Topic Selection**
- Review the project ideas list (see `project_ideas.md`) or formulate your own topic.
- Assess feasibility: Do you have access to the data? Do you have enough compute? Can the core experiment be run in under 24 hours of GPU time?
- Discuss your idea with a mentor or peer. Get honest feedback on scope.

**Days 3-5: Literature Review and Proposal Writing**
- Read a minimum of 5 relevant papers. At least 2 should be foundational to the area, and at least 2 should be recent (published within the last 18 months).
- For each paper, write a 2-3 sentence summary: what they did, what the key result was, and how it relates to your project.
- Write your proposal (see Proposal Requirements below).
- The proposal is a contract with yourself. Treat it seriously.

**Deliverable**: Written proposal document (2-3 pages). Submit for feedback before proceeding.

### Week 2: Implementation and Experimentation

**Days 1-2: Core Infrastructure**
- Set up your repository structure (see Code Submission Requirements).
- Implement data loading and preprocessing.
- Implement your model architecture.
- Write a minimal training loop and verify it runs on a small data subset.
- Commit early and commit often. Your git history is part of the evaluation.

**Days 3-4: Training and Initial Experiments**
- Train your baseline model(s).
- Train your proposed approach.
- Log everything: losses, metrics, hyperparameters, timing.
- If something is not working, diagnose it systematically. Check gradients. Visualize intermediate outputs. Compare against known results.

**Days 5-7: Iteration and Ablation Studies**
- Run ablation studies: remove or modify one component at a time to understand what matters.
- Tune hyperparameters methodically (grid search or Bayesian optimization, not random guessing).
- If your original approach fails, execute your backup plan. A well-documented negative result is far more valuable than a fabricated positive one.

**Deliverable**: Working codebase with trained models and logged experiment results.

### Week 3: Analysis, Writing, and Presentation

**Days 1-2: Analysis**
- Compile all experimental results into tables and figures.
- Perform error analysis: where does your model fail? Why?
- Create visualizations: training curves, attention maps, feature visualizations, sample outputs, confusion matrices -- whatever is appropriate for your project.
- Identify the key findings. What is the story your results tell?

**Days 3-5: Report Writing**
- Write your report following the conference-style format (see Report Requirements).
- Every claim must be backed by evidence (a number, a figure, a reference).
- Have someone else read your draft. Revise based on their confusion, not your intention.

**Days 6-7: Presentation Preparation**
- Build your slide deck (15 minutes of content).
- Prepare a live demo of your system.
- Practice your presentation at least twice. Time yourself.
- Anticipate questions and prepare answers.

**Deliverable**: Final report, presentation slides, polished code repository.

---

## Proposal Requirements

Your proposal should be 2-3 pages and must include every one of the following sections:

### 1. Problem Statement
What problem are you solving? State it precisely. A vague problem leads to a vague project.

*Good*: "I will implement a class-conditional diffusion model on CIFAR-10 and evaluate whether classifier-free guidance improves FID scores compared to classifier guidance."

*Bad*: "I want to explore generative AI for images."

### 2. Motivation
Why does this problem matter? Who cares about the answer? This can be practical (a real application) or scientific (understanding something about how models work). One strong paragraph is sufficient.

### 3. Approach
Describe your method in technical detail. Include:
- The model architecture (with a diagram -- even a hand-drawn one is fine at the proposal stage).
- The loss function(s) you will use.
- The training procedure (optimizer, learning rate schedule, batch size, number of epochs).
- Any novel components or modifications you are introducing.

### 4. Dataset
- What data will you use? How large is it?
- How will you split it (train/validation/test)?
- Any preprocessing or augmentation?
- If you are creating a new dataset, describe the collection and annotation process.

### 5. Evaluation Metrics
- What metrics will you report? (Accuracy, F1, FID, BLEU, perplexity, etc.)
- Why are these the right metrics for your problem?
- What baseline numbers do you expect, and what would constitute a good result?

### 6. Timeline
A week-by-week breakdown of what you will accomplish. Be specific. "Implement model" is not a plan. "Implement the U-Net backbone with sinusoidal time embeddings and verify it produces correct output shapes on a dummy batch" is a plan.

### 7. Risk Assessment
What could go wrong? Every project has risks. Identify at least three:
- **Technical risk**: The model might not converge, the dataset might be too noisy, training might take too long.
- **Scope risk**: The project might be too ambitious for the timeline.
- **Resource risk**: You might not have enough compute.

For each risk, state your mitigation strategy or backup plan. A project that fails gracefully and documents why is far better than one that quietly pretends everything worked.

---

## Implementation Requirements

Your code must meet the following standards. These are not optional. Sloppy code undermines everything else.

### Reproducibility
- **Random seeds**: Set seeds for Python, NumPy, and PyTorch. Document the seed used for every reported result.
- **Configuration files**: Use YAML or JSON config files for all hyperparameters. No magic numbers buried in code.
- **Environment**: Provide a `requirements.txt` or `environment.yml` with pinned versions.
- **Determinism**: Where possible, use `torch.backends.cudnn.deterministic = True`. Document any sources of non-determinism.

### Code Quality
- **Modularity**: Separate data loading, model definition, training logic, and evaluation into distinct files or modules.
- **Documentation**: Every function should have a docstring. Every non-obvious design choice should have a comment explaining why, not what.
- **Type hints**: Use Python type hints for function signatures.
- **Naming**: Variables and functions should have descriptive names. `x` is acceptable for tensor inputs in forward passes. `temp` is never acceptable.

### Version Control
- Use git throughout development, not just at the end.
- Commit messages should be meaningful: "Fix learning rate schedule bug causing divergence after epoch 50" not "update".
- Your git history should tell the story of your project.

### Training and Evaluation
- **Train/validation/test splits**: Use a held-out test set that you evaluate on exactly once, at the very end. Use the validation set for all development decisions.
- **Baselines**: Implement at least one meaningful baseline. A random baseline and a simple non-neural baseline are both useful. Published results on the same dataset count if you cite them.
- **Ablation studies**: Systematically vary your model to understand which components matter. Remove one thing at a time. Report results for each ablation.
- **Logging**: Use TensorBoard, Weights & Biases, or at minimum CSV logs for all training runs. You must be able to produce training curves.

---

## Report Requirements

Your report should be 6-8 pages in a conference-style format (single column is acceptable; use reasonable margins and 11-12pt font). The report is the primary artifact that communicates what you did and what you found.

### Abstract (200 words maximum)
State the problem, your approach, and your key result in 200 words or fewer. A reader should know exactly what you did and whether it worked after reading only the abstract.

### 1. Introduction
- What problem are you addressing?
- Why is it important or interesting?
- What is your approach at a high level?
- What are your contributions? (List 2-3 specific contributions as bullet points.)

### 2. Related Work
- Discuss at least 5 relevant papers.
- Do not just summarize each paper in isolation. Organize by theme and explain how your work relates to and differs from prior work.
- Demonstrate that you understand the landscape of the field your project sits in.

### 3. Method
- Describe your approach in enough detail that someone could re-implement it.
- Include an architecture diagram.
- State your loss function with the actual mathematical expression.
- Describe training details: optimizer, learning rate (and schedule), batch size, number of epochs, hardware used, training time.

### 4. Experiments
- **Dataset**: Size, source, splits, any preprocessing.
- **Baselines**: What you are comparing against and why.
- **Main results**: A clear table comparing your approach to baselines on your chosen metrics.
- **Ablation studies**: A table or set of figures showing the contribution of each component.
- **Qualitative results**: Where appropriate, show example outputs, visualizations, or case studies.

### 5. Analysis
This is the section that separates good work from great work.
- What worked and why do you think it worked?
- What did not work, and what did you try to fix it?
- Show failure cases. Analyze them.
- If you have visualizations (attention maps, t-SNE embeddings, gradient visualizations), include them here with interpretation.
- Be honest. Honest analysis of a partial success is worth more than a polished presentation of inflated results.

### 6. Conclusion
- Summarize your findings in 1-2 paragraphs.
- State the limitations of your work clearly.
- Suggest 2-3 concrete directions for future work.

### References
- Use a consistent citation format (author-year or numbered).
- Cite every paper you reference. Cite the datasets you use. Cite the libraries you use (PyTorch, etc.).

---

## Code Submission Requirements

Your GitHub repository must have the following structure (adapt as appropriate for your project):

```
project-name/
    README.md              # Project overview, setup instructions, how to reproduce results
    requirements.txt       # Python dependencies with pinned versions
    configs/
        default.yaml       # Default hyperparameters
        experiment_1.yaml  # Config for specific experiments
    data/
        README.md          # Data download instructions (do NOT commit large data files)
    src/
        data/
            dataset.py     # Dataset classes and data loading
            transforms.py  # Data augmentation and preprocessing
        models/
            model.py       # Model architecture(s)
            layers.py      # Custom layers or modules
        training/
            train.py       # Training loop
            losses.py      # Loss functions
            optimizer.py   # Optimizer and scheduler setup
        evaluation/
            evaluate.py    # Evaluation scripts
            metrics.py     # Metric computation
        utils/
            visualization.py  # Plotting and visualization utilities
            logging.py     # Experiment logging
    scripts/
        train.sh           # Shell script to reproduce training
        evaluate.sh        # Shell script to reproduce evaluation
    notebooks/
        analysis.ipynb     # Analysis and visualization notebook
    tests/
        test_model.py      # Unit tests for model (at minimum, shape checks)
        test_data.py       # Unit tests for data pipeline
    results/
        figures/           # Generated figures for the report
    report/
        report.pdf         # Final report
        slides.pdf         # Presentation slides
```

### README Requirements
Your README must include:
1. **Project title and one-line description.**
2. **Setup instructions**: How to create the environment and install dependencies.
3. **Data**: How to download and prepare the dataset.
4. **Training**: Exact command to reproduce training. Example: `python src/training/train.py --config configs/default.yaml --seed 42`
5. **Evaluation**: Exact command to evaluate a trained model.
6. **Results**: A summary table of your main results.
7. **Pretrained models**: Link to download pretrained weights, or clear instructions for how long training takes.

---

## Presentation

### Format
- 15 minutes of presentation followed by 5 minutes of Q&A.
- You may use slides, a whiteboard, or any combination.

### Required Content
1. **Problem and motivation** (2 minutes): What are you solving and why should the audience care?
2. **Related work** (2 minutes): Brief overview of the landscape. What has been done before?
3. **Method** (4 minutes): Your approach, with a clear architecture diagram. This is the core of the talk.
4. **Results** (4 minutes): Key experimental results. Lead with your main result, then ablations.
5. **Demo** (2 minutes): Show your system working on real inputs. A live demo is strongly preferred. If a live demo is not feasible, show recorded outputs.
6. **Reflection** (1 minute): What did you learn? What would you do differently? Be candid about failures.

### Presentation Advice
- Practice your talk. Then practice it again.
- Do not read from your slides. Your slides should support your talk, not replace it.
- One key idea per slide. If a slide has more than 30 words, it has too many words.
- Figures and diagrams are almost always better than text.
- Anticipate the hard questions: "Why didn't you try X?" "How does this compare to Y?" "What happens when Z?"
- If you do not know the answer to a question, say so. Then say what you would do to find out.

---

## Evaluation Rubric (Summary)

The detailed rubric is in `evaluation_rubric.md`. Here is the high-level breakdown:

| Category              | Points | What We Are Looking For                                          |
|-----------------------|--------|------------------------------------------------------------------|
| Technical Depth       | 30     | Sophistication of approach, depth of understanding               |
| Implementation Quality| 25     | Clean code, reproducibility, proper engineering                  |
| Experimental Rigor    | 20     | Baselines, ablations, honest reporting, proper metrics           |
| Writing & Presentation| 15     | Clear communication, good figures, engaging presentation         |
| Originality           | 10     | Creative contribution, novel insight, independent thinking       |
| **Total**             |**100** |                                                                  |

---

## Final Advice

The capstone is not about getting the highest accuracy number. It is about demonstrating that you can:

1. **Formulate** a well-defined problem.
2. **Design** a thoughtful solution grounded in the literature.
3. **Implement** it cleanly and correctly.
4. **Evaluate** it rigorously and honestly.
5. **Communicate** what you found and what it means.

These are the five skills that separate someone who has taken an ML course from someone who can do ML research. The capstone is where you prove you have all five.

Start now. The three weeks will pass faster than you think.
