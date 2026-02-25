# Deep Learning Apprenticeship: Full Syllabus

**Course Title:** Foundations and Frontiers of Deep Learning
**Format:** Self-directed apprenticeship with structured milestones
**Duration:** 16 weeks (one semester), extensible based on mastery pace
**Prerequisite Knowledge:** Linear algebra, multivariable calculus, basic probability, and working proficiency in Python

---

## Course Philosophy

This is not a university course. There are no exams, no curves, and no partial credit. This apprenticeship is modeled after how researchers at labs like DeepMind actually develop deep competence: by building things from scratch, reading real papers, and iterating until understanding is genuine rather than performative.

You advance when you demonstrate mastery, not when a calendar says you should.

---

## Learning Objectives

By the end of this apprenticeship, you will be able to:

1. Implement core neural network architectures from scratch in NumPy and PyTorch.
2. Read, understand, and reproduce results from foundational deep learning papers.
3. Debug training pipelines systematically rather than by trial and error.
4. Reason about why architectural choices work, not just that they work.
5. Design experiments, interpret results, and communicate findings clearly.
6. Navigate the research literature independently and critically.

---

## Grading Philosophy: Mastery-Based Progression

There are no letter grades. Each module has a set of **mastery criteria** — concrete, observable demonstrations of understanding. You either meet them or you don't yet.

**What mastery looks like:**

- You can implement the concept without reference material.
- You can explain it clearly to someone who has never seen it (Feynman test).
- You can identify when and why it would or would not be appropriate to use.
- You can predict how changes to the system would affect behavior.

**What mastery does not look like:**

- Copying code from a tutorial and getting it to run.
- Memorizing formulas without understanding their derivation.
- Getting a "good enough" result and moving on.

If you find yourself unable to meet the mastery criteria for a module, that is valuable information. Go back. Revisit prerequisites. Ask questions. The goal is depth, not speed.

---

## Week-by-Week Breakdown

### Module 1: Foundations (Weeks 1--3)

**Week 1 — Mathematical Foundations and NumPy Fluency**

- Linear algebra review: vectors, matrices, eigendecomposition, SVD
- Calculus review: gradients, Jacobians, the chain rule in vector form
- Probability review: distributions, expectation, Bayes' theorem
- NumPy drills: vectorized operations, broadcasting, avoiding loops

*Learning objectives:* Compute gradients by hand for multi-layer compositions. Implement matrix operations in NumPy without loops. Explain why broadcasting works the way it does.

**Week 2 — The Perceptron to the MLP**

- The perceptron: linear classifiers, decision boundaries
- Activation functions: sigmoid, tanh, ReLU and the vanishing gradient problem
- Multi-layer perceptrons: universal approximation theorem (intuition and implications)
- Implement a 2-layer MLP from scratch in NumPy (forward pass, backward pass, training loop)

*Learning objectives:* Hand-derive backpropagation for a 2-layer network. Implement SGD from scratch. Explain the universal approximation theorem and why depth matters despite it.

**Week 3 — Backpropagation and Computational Graphs**

- Backpropagation as reverse-mode automatic differentiation
- Computational graphs: nodes, edges, forward and backward passes
- Implement a simple autograd engine (scalar-valued, then tensor-valued)
- Introduction to PyTorch: tensors, autograd, nn.Module

*Learning objectives:* Build a working autograd system. Trace gradients through a computational graph by hand. Transition fluently between NumPy and PyTorch.

---

### Module 2: Core Architectures (Weeks 4--7)

**Week 4 — Optimization**

- SGD, momentum, Nesterov momentum
- Adaptive methods: Adagrad, RMSProp, Adam
- Learning rate schedules: step decay, cosine annealing, warmup
- Loss landscapes: saddle points, local minima, and why they matter less than you think
- Implement optimizers from scratch, then compare against `torch.optim`

*Learning objectives:* Implement Adam from scratch. Explain why adaptive learning rates help. Diagnose common optimization pathologies from loss curves.

**Week 5 — Regularization and Generalization**

- Overfitting, underfitting, and the bias-variance tradeoff
- L1/L2 regularization: Bayesian interpretation
- Dropout: theory and practice
- Batch normalization, layer normalization, group normalization
- Data augmentation as implicit regularization
- Early stopping

*Learning objectives:* Implement dropout and batch normalization from scratch. Explain the Bayesian interpretation of weight decay. Design a regularization strategy for a given problem.

**Week 6 — Convolutional Neural Networks**

- Convolution as a mathematical operation and as a design prior
- Convolutional layers, pooling, stride, padding
- Classic architectures: LeNet, AlexNet, VGG, ResNet
- Residual connections: why they work (skip connections and gradient flow)
- Paper read: "Deep Residual Learning for Image Recognition" (He et al., 2015)
- Implement a CNN from scratch, then build ResNet in PyTorch

*Learning objectives:* Implement convolution (forward and backward) from scratch. Reproduce key results from the ResNet paper. Explain why residual connections solve the degradation problem.

**Week 7 — Recurrent Neural Networks and Sequence Modeling**

- Vanilla RNNs: unrolling, backpropagation through time
- The vanishing/exploding gradient problem in sequences
- LSTMs and GRUs: gating mechanisms
- Bidirectional RNNs, sequence-to-sequence models
- Implement an LSTM from scratch

*Learning objectives:* Implement BPTT for a vanilla RNN. Explain how LSTM gates control information flow. Train a character-level language model.

---

### Module 3: Modern Deep Learning (Weeks 8--11)

**Week 8 — Attention and Transformers**

- Attention as a soft dictionary lookup
- Scaled dot-product attention, multi-head attention
- The Transformer architecture: encoder, decoder, positional encoding
- Paper read: "Attention Is All You Need" (Vaswani et al., 2017)
- Implement a Transformer from scratch in PyTorch

*Learning objectives:* Implement multi-head attention from scratch. Explain why scaling by $\sqrt{d_k}$ is necessary. Build and train a Transformer for a sequence task.

**Week 9 — Natural Language Processing with Transformers**

- Word embeddings: Word2Vec, GloVe (historical context)
- Contextual embeddings and language modeling
- BERT: masked language modeling and pre-training/fine-tuning
- GPT: autoregressive language modeling
- Paper read: "BERT: Pre-training of Deep Bidirectional Transformers" (Devlin et al., 2019)
- Fine-tune a pre-trained Transformer on a downstream task

*Learning objectives:* Explain the difference between autoregressive and masked language models. Fine-tune a pre-trained model. Evaluate NLP models critically.

**Week 10 — Generative Models I: VAEs and Autoencoders**

- Autoencoders: representation learning, bottleneck architectures
- Variational Autoencoders: the ELBO, reparameterization trick
- Latent space interpolation and disentanglement
- Paper read: "Auto-Encoding Variational Bayes" (Kingma & Welling, 2013)
- Implement a VAE from scratch

*Learning objectives:* Derive the ELBO. Implement the reparameterization trick. Train a VAE on MNIST and explore the latent space.

**Week 11 — Generative Models II: GANs and Diffusion**

- Generative Adversarial Networks: minimax formulation, training dynamics
- Mode collapse and training instability
- Wasserstein GAN and spectral normalization
- Diffusion models: forward process, reverse process, denoising score matching
- Paper read: "Generative Adversarial Nets" (Goodfellow et al., 2014)
- Implement a basic GAN, then a simple diffusion model

*Learning objectives:* Implement and train a GAN from scratch. Explain mode collapse and mitigation strategies. Articulate the key insight behind diffusion models.

---

### Module 4: Research Skills (Weeks 12--14)

**Week 12 — Training at Scale**

- Distributed training: data parallelism, model parallelism
- Mixed precision training
- Gradient accumulation and large batch training
- Hyperparameter search: grid, random, Bayesian optimization
- Experiment tracking with TensorBoard and Weights & Biases

*Learning objectives:* Set up a training pipeline with mixed precision. Use W&B for experiment tracking. Make principled hyperparameter choices.

**Week 13 — Debugging and Evaluation**

- Systematic debugging of neural networks (Karpathy's "Recipe for Training Neural Networks")
- Ablation studies: how to isolate what matters
- Evaluation metrics beyond accuracy: precision, recall, calibration, fairness
- Reproducibility: seeds, deterministic training, experiment logging
- How to write up experimental results

*Learning objectives:* Debug a deliberately broken training pipeline. Design and execute an ablation study. Write a clear experimental report.

**Week 14 — Reading and Reproducing Papers**

- How to read a research paper (the 3-pass method)
- Identifying key contributions vs. incremental improvements
- Reproducing a paper from scratch: choosing a paper, setting expectations, common pitfalls
- Begin a paper reproduction project (to be completed in Weeks 15--16)

*Learning objectives:* Read and summarize three papers independently. Begin a faithful reproduction of a selected paper.

---

### Module 5: Capstone (Weeks 15--16)

**Weeks 15--16 — Paper Reproduction Project**

- Complete the reproduction of a selected research paper
- Write a detailed report: motivation, methodology, results, discrepancies, lessons learned
- Present findings (written or recorded)

*Learning objectives:* Produce a working implementation of a published result. Write a report that honestly documents both successes and failures. Demonstrate the ability to work independently on a research-adjacent task.

---

## Office Hours Expectations

This is a mentorship, and mentorship requires honest dialogue.

**How to use office hours effectively:**

1. Come with a specific, well-formed question. "It doesn't work" is not a question.
2. Show what you have tried. Include code, error messages, and your hypothesis about what is going wrong.
3. Demonstrate that you have spent meaningful time struggling before asking. The struggle is where learning happens.
4. Be prepared to be asked "what do you think is happening?" — the goal is to develop your debugging intuition, not to hand you answers.

**What to expect:**

- Socratic questioning over direct answers.
- Honest feedback. If your understanding is shallow, you will hear that — kindly, but clearly.
- Pointers to resources rather than solutions, unless you are genuinely stuck after sustained effort.

---

## Recommended Reading List

### Primary Textbooks

1. **Deep Learning** by Ian Goodfellow, Yoshua Bengio, and Aaron Courville
   - The canonical reference. Dense but comprehensive. Read the math carefully.
   - Freely available at [deeplearningbook.org](https://www.deeplearningbook.org/)

2. **Dive into Deep Learning** by Aston Zhang, Zachary C. Lipton, Mu Li, and Alexander J. Smola
   - Excellent interactive textbook with runnable code.
   - Freely available at [d2l.ai](https://d2l.ai/)

3. **PyTorch Documentation**
   - Not a textbook, but you should be able to navigate it fluently.
   - [pytorch.org/docs](https://pytorch.org/docs/stable/index.html)

### Foundational Papers (in approximate reading order)

- Rosenblatt, 1958 — "The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain"
- Rumelhart, Hinton & Williams, 1986 — "Learning representations by back-propagating errors"
- LeCun et al., 1998 — "Gradient-Based Learning Applied to Document Recognition"
- Krizhevsky, Sutskever & Hinton, 2012 — "ImageNet Classification with Deep Convolutional Neural Networks" (AlexNet)
- He et al., 2015 — "Deep Residual Learning for Image Recognition" (ResNet)
- Hochreiter & Schmidhuber, 1997 — "Long Short-Term Memory"
- Bahdanau, Cho & Bengio, 2014 — "Neural Machine Translation by Jointly Learning to Align and Translate"
- Vaswani et al., 2017 — "Attention Is All You Need"
- Devlin et al., 2019 — "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"
- Kingma & Welling, 2013 — "Auto-Encoding Variational Bayes"
- Goodfellow et al., 2014 — "Generative Adversarial Nets"
- Ho, Jain & Abbeel, 2020 — "Denoising Diffusion Probabilistic Models"

### Supplementary Resources

- **Mathematics for Machine Learning** by Deisenroth, Faisal, and Ong — for shoring up mathematical foundations.
- **The Little Book of Deep Learning** by Francois Fleuret — a concise, well-written overview.
- **Andrej Karpathy's blog posts** — especially "A Recipe for Training Neural Networks" and "The Unreasonable Effectiveness of Recurrent Neural Networks."
- **3Blue1Brown's Neural Networks series** — for visual intuition building.
- **Distill.pub articles** — for interactive, beautifully explained deep learning concepts (particularly on attention, feature visualization, and momentum).

---

## A Final Note

The researchers you admire did not get there by taking courses. They got there by doing the work — reading papers they did not yet understand, implementing things that broke, staring at loss curves, and slowly building an intuition that no shortcut can provide.

This syllabus is a scaffold. The real learning happens in the gap between reading about an idea and making it work. Lean into that gap.
