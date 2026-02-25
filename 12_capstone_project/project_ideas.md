# Module 12 -- Capstone Project: Project Ideas

This document contains a curated list of capstone project ideas organized by domain and difficulty level. Each idea has been selected because it exercises a meaningful combination of skills from the course while remaining feasible within the 2-3 week capstone timeline.

**Difficulty Levels:**
- **Intermediate**: Relies primarily on well-established techniques covered in the course. A strong implementation with good experimental analysis is expected.
- **Advanced**: Requires combining multiple techniques or adapting methods to new settings. Demands deeper understanding and more careful engineering.
- **Research-level**: Involves reproducing or extending recent work. May require reading several additional papers and making independent design decisions. Suitable for apprentices who want to push beyond the course material.

Choose a project that challenges you. If you are confident you can finish it in one week, it is probably too easy.

---

## Computer Vision

### 1. Neural Style Transfer with Controllable Style Strength

**Difficulty**: Intermediate

**Description**: Implement the neural style transfer method of Gatys et al. (2016), then extend it with perceptual losses and a user-controllable style/content tradeoff parameter. The goal is not merely to reproduce the original paper but to build a system where a user can smoothly interpolate between content preservation and style application, and to analyze how different layers of the CNN contribute to style vs. content.

**Key Techniques**: Convolutional neural networks, feature extraction from pretrained models (VGG), optimization-based image generation, perceptual loss functions, Gram matrix computation.

**Suggested Papers**:
- Gatys, Ecker, Bethge. "A Neural Algorithm of Artistic Style" (2016)
- Johnson, Alahi, Fei-Fei. "Perceptual Losses for Real-Time Style Transfer and Super-Resolution" (2016)
- Huang and Belongie. "Arbitrary Style Transfer in Real-Time with Adaptive Instance Normalization" (2017)

**Dataset**: Any collection of content images (e.g., MS COCO validation set) and style images (e.g., WikiArt). No labels needed.

**Evaluation Metrics**: Qualitative visual quality, content loss vs. style loss tradeoff curves, runtime comparison between optimization-based and feed-forward approaches, user study if feasible (informal is fine).

**What a Good Result Looks Like**: A working system that produces visually compelling style transfers across a variety of content/style pairs, with a clear demonstration of the controllable tradeoff. The report should include a systematic analysis of which VGG layers matter most for style vs. content, and a comparison between at least two approaches (e.g., optimization-based vs. feed-forward).

---

### 2. Image Super-Resolution with Diffusion Models

**Difficulty**: Advanced

**Description**: Implement a small-scale diffusion-based super-resolution system. Train a denoising diffusion probabilistic model (DDPM) conditioned on low-resolution images to produce high-resolution outputs. Compare against a simple bicubic baseline and a CNN-based approach (e.g., SRCNN or EDSR). Work at a manageable scale (e.g., 64x64 to 128x128) to keep training feasible.

**Key Techniques**: Diffusion models, U-Net architecture, noise scheduling, conditional generation, image quality metrics, convolutional networks.

**Suggested Papers**:
- Ho, Jain, Abbeel. "Denoising Diffusion Probabilistic Models" (2020)
- Saharia et al. "Image Super-Resolution via Iterative Refinement (SR3)" (2022)
- Dong et al. "Image Super-Resolution Using Deep Convolutional Networks (SRCNN)" (2015)

**Dataset**: CelebA (faces, 64x64 to 128x128) or DIV2K (natural images, work at reduced resolution). Create low-resolution inputs by downsampling with bicubic interpolation.

**Evaluation Metrics**: PSNR, SSIM, LPIPS (perceptual similarity), FID on generated high-res images. Report inference time per image.

**What a Good Result Looks Like**: The diffusion-based approach should produce visibly sharper and more detailed outputs than bicubic upsampling, with competitive or superior perceptual quality compared to the CNN baseline. The report should include a clear analysis of the tradeoff between sample quality and inference speed (number of diffusion steps), with visual comparisons across methods.

---

### 3. Few-Shot Image Classification

**Difficulty**: Advanced

**Description**: Implement a few-shot learning system using either Prototypical Networks or Model-Agnostic Meta-Learning (MAML) and evaluate on the mini-ImageNet benchmark. The core challenge is building the episodic training framework and understanding why meta-learning outperforms naive fine-tuning when data is scarce. Compare at least two approaches (e.g., Prototypical Networks vs. fine-tuning a pretrained feature extractor).

**Key Techniques**: Meta-learning, episodic training, metric learning, embedding spaces, transfer learning, convolutional feature extractors.

**Suggested Papers**:
- Snell, Swersky, Zemel. "Prototypical Networks for Few-Shot Learning" (2017)
- Finn, Abbeel, Levine. "Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks" (2017)
- Vinyals et al. "Matching Networks for One Shot Learning" (2016)
- Chen et al. "A Closer Look at Few-Shot Classification" (2019)

**Dataset**: mini-ImageNet (100 classes, 600 images per class, standard 64/16/20 class split for train/val/test). Alternatively, Omniglot for a simpler starting point.

**Evaluation Metrics**: 5-way 1-shot accuracy, 5-way 5-shot accuracy (with 95% confidence intervals over 1000 episodes). Compare against a fine-tuning baseline and a nearest-neighbor baseline on pretrained features.

**What a Good Result Looks Like**: Prototypical Networks achieving approximately 49-52% on 5-way 1-shot and 65-70% on 5-way 5-shot on mini-ImageNet (these are in the range of published results). A clear analysis of the learned embedding space (t-SNE visualizations), and a discussion of when and why meta-learning helps compared to transfer learning.

---

### 4. Vision Transformer vs. CNN: A Systematic Comparison

**Difficulty**: Intermediate

**Description**: Implement both a Vision Transformer (ViT) and a ResNet-style CNN, train them on the same datasets (CIFAR-10 and at least one other, e.g., CIFAR-100 or Tiny ImageNet), and perform a systematic comparison. Go beyond accuracy: analyze what each architecture learns by visualizing attention maps (ViT) vs. feature maps and Grad-CAM (CNN). Investigate how performance changes with dataset size and augmentation.

**Key Techniques**: Transformer architecture (self-attention, positional encoding, patch embedding), convolutional neural networks (residual connections, batch normalization), attention visualization, Grad-CAM, data augmentation.

**Suggested Papers**:
- Dosovitskiy et al. "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" (2021)
- He et al. "Deep Residual Learning for Image Recognition" (2016)
- Raghu et al. "Do Vision Transformers See Like Convolutional Neural Networks?" (2021)

**Dataset**: CIFAR-10, CIFAR-100, and optionally Tiny ImageNet (200 classes, 64x64 images). To study data efficiency, create subsets of varying sizes.

**Evaluation Metrics**: Test accuracy on each dataset, accuracy vs. training set size curves, training time comparison, parameter count comparison, and qualitative comparison of learned representations (attention maps vs. Grad-CAM).

**What a Good Result Looks Like**: Clear evidence that the CNN is more data-efficient at small scale while the ViT benefits more from augmentation and larger datasets (consistent with the literature). Insightful visualizations showing that ViT attention captures global structure early while CNNs build local-to-global hierarchically. A fair comparison where both models have comparable parameter counts.

---

## Natural Language Processing

### 5. Build a Small GPT

**Difficulty**: Intermediate

**Description**: Implement a GPT-style autoregressive language model from scratch in PyTorch. Train it at the character level or with byte-pair encoding (BPE) on a focused domain corpus (e.g., Shakespeare, Python source code, scientific abstracts, or song lyrics). The emphasis is on understanding every component of the Transformer decoder: masked self-attention, positional encoding, layer normalization, and the autoregressive generation process. Experiment with model size, context length, and decoding strategies.

**Key Techniques**: Transformer decoder, masked self-attention, positional encoding (sinusoidal and learned), layer normalization, autoregressive generation, temperature sampling, top-k and nucleus (top-p) sampling, BPE tokenization.

**Suggested Papers**:
- Radford et al. "Language Models are Unsupervised Multitask Learners (GPT-2)" (2019)
- Vaswani et al. "Attention Is All You Need" (2017)
- Karpathy. "The Unreasonable Effectiveness of Recurrent Neural Networks" (2015, blog post -- for character-level motivation)

**Dataset**: Shakespeare complete works (~1MB text), Python source code from a GitHub scrape, or a subset of scientific abstracts from arXiv. Keep the dataset under 100MB to ensure training is feasible.

**Evaluation Metrics**: Validation perplexity (primary metric), bits-per-character (for character-level models), qualitative sample quality at different temperatures, training loss curves across model sizes.

**What a Good Result Looks Like**: A model that generates coherent domain-specific text (e.g., text that sounds like Shakespeare, or syntactically valid Python). A clear scaling analysis showing how perplexity improves with model size and training data. Demonstration of different sampling strategies and their effect on output quality. An ablation study removing one Transformer component at a time (e.g., no positional encoding, no layer norm).

---

### 6. Efficient Transformer for Long Documents

**Difficulty**: Advanced

**Description**: Implement and compare two or more efficient Transformer variants designed for long sequences: choose from Linformer, Performer, Longformer, or BigBird. Train them on a long-document task (e.g., document classification, long-range sequence modeling on LRA benchmark tasks, or summarization). Analyze the accuracy-efficiency tradeoff: how much quality do you sacrifice for linear-time attention?

**Key Techniques**: Standard self-attention (quadratic baseline), linear attention approximations, sparse attention patterns, random feature maps (for Performer), sliding window attention (for Longformer), benchmarking and profiling.

**Suggested Papers**:
- Wang et al. "Linformer: Self-Attention with Linear Complexity" (2020)
- Choromanski et al. "Rethinking Attention with Performers" (2021)
- Beltagy, Peters, Cohan. "Longformer: The Long-Document Transformer" (2020)
- Tay et al. "Long Range Arena: A Benchmark for Efficient Transformers" (2021)

**Dataset**: Long Range Arena (LRA) benchmark tasks (ListOps, byte-level text classification, byte-level document retrieval, image classification on sequential pixels, Pathfinder). Alternatively, IMDB reviews or a document classification dataset with long texts.

**Evaluation Metrics**: Accuracy on task, wall-clock training time, memory usage (peak GPU memory), throughput (sequences per second) as a function of sequence length. Plot accuracy and speed against standard Transformer baseline.

**What a Good Result Looks Like**: Clear demonstration that efficient variants achieve comparable accuracy to standard attention on shorter sequences, with significant speed and memory advantages on long sequences. A plot showing how memory and time scale with sequence length for each method. An honest discussion of the failure modes of each efficient attention mechanism.

---

### 7. Question Answering System

**Difficulty**: Intermediate

**Description**: Fine-tune a Transformer-based model on the SQuAD 2.0 extractive question answering task. Start from a pretrained BERT or DistilBERT checkpoint, implement the QA head, and train on SQuAD. Then introduce at least one custom improvement: this could be a better answer verification mechanism for unanswerable questions, data augmentation for the training set, or an ensemble approach. Analyze error patterns systematically.

**Key Techniques**: Transfer learning, Transformer fine-tuning, span extraction, handling unanswerable questions, tokenization and alignment between tokens and character spans, learning rate warmup.

**Suggested Papers**:
- Devlin et al. "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" (2019)
- Rajpurkar et al. "Know What You Don't Know: Unanswerable Questions for SQuAD" (2018)
- Clark and Gardner. "Simple and Effective Multi-Paragraph Reading Comprehension" (2018)

**Dataset**: SQuAD 2.0 (100,000+ answerable questions, 50,000+ unanswerable questions, based on Wikipedia passages).

**Evaluation Metrics**: Exact Match (EM) and F1 score on the SQuAD 2.0 dev set. Report separately for answerable and unanswerable questions.

**What a Good Result Looks Like**: A fine-tuned DistilBERT achieving approximately 65-70 EM / 68-73 F1 on SQuAD 2.0 dev (or a fine-tuned BERT-base achieving approximately 73-76 EM / 76-79 F1). A custom improvement that yields a measurable (even if small) gain. A detailed error analysis showing common failure modes (e.g., questions requiring multi-hop reasoning, numerical reasoning, or coreference resolution).

---

### 8. Multilingual Sentiment Analysis

**Difficulty**: Advanced

**Description**: Build a sentiment analysis system that works across multiple languages using a shared multilingual Transformer backbone (e.g., mBERT or XLM-RoBERTa). Train on English sentiment data and evaluate zero-shot transfer to other languages. Then explore few-shot adaptation: how much target-language data is needed to match monolingual performance? Analyze what the shared representation learns across languages.

**Key Techniques**: Multilingual Transformers, cross-lingual transfer learning, zero-shot and few-shot transfer, fine-tuning strategies, representation analysis (CKA, probing classifiers, or embedding visualization).

**Suggested Papers**:
- Conneau et al. "Unsupervised Cross-lingual Representation Learning at Scale (XLM-R)" (2020)
- Devlin et al. "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" (2019)
- Pires, Schlinger, Garrette. "How Multilingual is Multilingual BERT?" (2019)

**Dataset**: Multilingual Amazon Reviews (English, German, French, Japanese, Chinese, Spanish) or the XNLI dataset for cross-lingual natural language inference. Use English for training and at least 3 other languages for evaluation.

**Evaluation Metrics**: Accuracy and macro-F1 per language. Zero-shot accuracy (train on English only, test on other languages). Few-shot accuracy curves (10, 50, 100, 500 target-language examples). Compare against monolingual baselines.

**What a Good Result Looks Like**: Zero-shot transfer that performs well above random (e.g., 70%+ accuracy on binary sentiment for related languages like German and French, lower for distant languages like Japanese). Clear few-shot learning curves showing diminishing returns. Visualization of cross-lingual embedding space showing language clusters and their alignment.

---

## Generative Models

### 9. Conditional Image Generation with Diffusion

**Difficulty**: Advanced

**Description**: Implement a class-conditional Denoising Diffusion Probabilistic Model (DDPM) on CIFAR-10. Implement both classifier guidance and classifier-free guidance, and compare them. This project requires implementing the full diffusion pipeline: the forward noising process, the reverse denoising process, the noise schedule, and the conditioning mechanism. Work at 32x32 resolution to keep training tractable.

**Key Techniques**: Diffusion models, U-Net with time conditioning, class conditioning (embedding concatenation, cross-attention, or adaptive group normalization), classifier guidance, classifier-free guidance, FID computation, noise scheduling.

**Suggested Papers**:
- Ho, Jain, Abbeel. "Denoising Diffusion Probabilistic Models" (2020)
- Dhariwal and Nichol. "Diffusion Models Beat GANs on Image Synthesis" (2021)
- Ho and Salimans. "Classifier-Free Diffusion Guidance" (2022)

**Dataset**: CIFAR-10 (50,000 training images, 10 classes, 32x32 resolution).

**Evaluation Metrics**: FID score (primary), Inception Score, per-class generation quality (visual inspection), diversity of samples. Compare unconditional vs. class-conditional generation.

**What a Good Result Looks Like**: Recognizable CIFAR-10 class images generated conditionally (cars look like cars, horses look like horses). FID below 50 would be reasonable for a from-scratch implementation at this scale. A clear comparison between classifier guidance and classifier-free guidance with different guidance scales. A visualization of the denoising process showing how an image emerges from noise over the reverse diffusion steps.

---

### 10. Music Generation with Transformers

**Difficulty**: Advanced

**Description**: Build a GPT-style autoregressive model that generates MIDI music sequences. Tokenize MIDI events (note-on, note-off, velocity, time-shift) into a sequence of discrete tokens, then train a Transformer decoder to predict the next token. Generate novel musical pieces and evaluate their musicality. Experiment with different tokenization schemes and conditioning (e.g., genre or instrument).

**Key Techniques**: Transformer decoder, autoregressive generation, custom tokenization for structured data, sequence modeling, relative positional encoding, MIDI processing.

**Suggested Papers**:
- Huang et al. "Music Transformer: Generating Music with Long-Term Structure" (2019)
- Oore et al. "This Time with Feeling: Learning Expressive Musical Performance" (2020)
- Payne. "MuseNet" (2019, OpenAI blog post)

**Dataset**: MAESTRO dataset (MIDI recordings of piano performances, ~200 hours) or Lakh MIDI Dataset (large collection of MIDI files across genres). Start with a single-instrument subset.

**Evaluation Metrics**: Validation negative log-likelihood, qualitative listening evaluation (do the outputs sound musical?), self-similarity analysis (does the music have repeating structure?), pitch and rhythm distribution analysis (compare statistics of generated vs. real music).

**What a Good Result Looks Like**: Generated pieces of 30-60 seconds that sound like plausible (if imperfect) music, with some recognizable structure (phrases, repetition, tension/resolution). A comparison between different model sizes and context lengths showing how longer context improves musical coherence. Audio samples included in the submission.

---

### 11. Text-to-Image with CLIP Guidance

**Difficulty**: Research-level

**Description**: Build a simple text-guided image generation system using CLIP as a guidance signal. Start with either a diffusion model or a GAN generator, and optimize the generated image to maximize CLIP similarity with a text prompt. This is a simplified version of systems like DALL-E 2 or Stable Diffusion, focused on understanding how text-image alignment can guide generation.

**Key Techniques**: CLIP model (contrastive learning between text and images), diffusion models or GANs, gradient-based optimization of images through a frozen CLIP model, augmentation-based regularization, prompt engineering.

**Suggested Papers**:
- Radford et al. "Learning Transferable Visual Models From Natural Language Supervision (CLIP)" (2021)
- Ramesh et al. "Hierarchical Text-Conditional Image Generation with CLIP Latents (DALL-E 2)" (2022)
- Crowson et al. "VQGAN+CLIP: Open Domain Image Generation and Editing with Natural Language Guidance" (2022)

**Dataset**: No training dataset needed for the CLIP-guided optimization approach (you use a pretrained CLIP model). For a diffusion-based approach, you may use CIFAR-10 or a subset of LAION for training the generator.

**Evaluation Metrics**: CLIP score (cosine similarity between generated image and text prompt), qualitative visual quality, diversity of outputs for the same prompt, comparison of different prompts and guidance strengths.

**What a Good Result Looks Like**: Generated images that are recognizably related to the text prompt, even if not photorealistic. A systematic study of how guidance strength affects quality vs. diversity. Honest documentation of which prompts work well and which fail. Comparison of at least two generation approaches (e.g., direct optimization vs. diffusion with CLIP guidance).

---

## Reinforcement Learning and Other Topics

### 12. Deep Q-Network for Atari

**Difficulty**: Intermediate

**Description**: Implement a Deep Q-Network (DQN) from scratch and train it to play Atari games (e.g., Pong, Breakout, or Space Invaders). Implement the core components that made DQN work: experience replay, target networks, and frame stacking. Then implement at least one improvement (Double DQN, Dueling DQN, or prioritized experience replay) and measure its effect.

**Key Techniques**: Q-learning, deep Q-networks, experience replay buffers, target networks, epsilon-greedy exploration, convolutional feature extraction from frames, reward clipping, frame stacking and preprocessing.

**Suggested Papers**:
- Mnih et al. "Playing Atari with Deep Reinforcement Learning" (2013)
- Mnih et al. "Human-Level Control through Deep Reinforcement Learning" (2015)
- Van Hasselt, Guez, Silver. "Deep Reinforcement Learning with Double Q-learning" (2016)
- Wang et al. "Dueling Network Architectures for Deep Reinforcement Learning" (2016)

**Dataset/Environment**: OpenAI Gymnasium (formerly Gym) Atari environments. Start with Pong (simplest) and move to Breakout or Space Invaders if time permits.

**Evaluation Metrics**: Average episode reward over 100 episodes (after training), reward learning curves during training, comparison of DQN vs. improved variant, training wall-clock time, number of frames to reach human-level performance (where applicable).

**What a Good Result Looks Like**: An agent that achieves positive average reward on Pong (i.e., beats the built-in opponent) within a reasonable training budget (1-5 million frames). A clear learning curve showing the agent improving over time. A measurable improvement from the chosen DQN enhancement. A visualization or recording of the trained agent playing.

---

### 13. Neural Architecture Search (Simple)

**Difficulty**: Advanced

**Description**: Implement a basic neural architecture search (NAS) system for CNN architectures on CIFAR-10. Define a search space of architectural choices (number of layers, filter sizes, whether to use skip connections, activation functions, etc.) and use random search or a simple evolutionary algorithm to find good architectures. Compare the searched architecture against a hand-designed baseline. The goal is understanding the NAS pipeline, not achieving state-of-the-art.

**Key Techniques**: Search space design, architecture encoding, random search or evolutionary algorithms, weight sharing (optional, for efficiency), early stopping, multi-fidelity evaluation (train for fewer epochs during search, full training for final evaluation).

**Suggested Papers**:
- Zoph and Le. "Neural Architecture Search with Reinforcement Learning" (2017)
- Real et al. "Regularized Evolution for Image Classifier Architecture Search" (2019)
- Li and Talwalkar. "Random Search and Reproducibility for Neural Architecture Search" (2020)

**Dataset**: CIFAR-10 for architecture evaluation. Use reduced training epochs (e.g., 10-20 epochs) during the search phase for efficiency, with full training (100+ epochs) for the final selected architecture.

**Evaluation Metrics**: Test accuracy of the best found architecture vs. hand-designed baseline, search cost (total GPU hours), accuracy vs. parameter count (efficiency frontier), distribution of accuracies across searched architectures.

**What a Good Result Looks Like**: A searched architecture that matches or slightly exceeds a hand-designed ResNet or VGG baseline on CIFAR-10. A clear visualization of the search process (accuracy distribution, convergence of the search). An analysis of which architectural choices matter most (e.g., skip connections help more than changing activation functions). An honest assessment of the computational cost of the search.

---

### 14. Knowledge Distillation

**Difficulty**: Intermediate

**Description**: Train a large "teacher" model and distill its knowledge into a smaller "student" model. Investigate what knowledge transfers during distillation: is it just the output probabilities, or does the student also learn internal representations? Compare distillation against training the student from scratch. Experiment with different distillation temperatures and loss weighting.

**Key Techniques**: Knowledge distillation (soft targets), temperature scaling, KL divergence loss, feature-level distillation (optional), model compression analysis, representation comparison (CKA or centered kernel alignment).

**Suggested Papers**:
- Hinton, Vinyals, Dean. "Distilling the Knowledge in a Neural Network" (2015)
- Romero et al. "FitNets: Hints for Thin Deep Nets" (2015)
- Tian et al. "Contrastive Representation Distillation" (2020)

**Dataset**: CIFAR-10 or CIFAR-100. CIFAR-100 is preferred because the larger number of classes makes the soft target distribution more informative.

**Evaluation Metrics**: Test accuracy of teacher, student trained from scratch, and distilled student. Plot accuracy vs. temperature, accuracy vs. loss weighting (alpha between hard and soft targets). Model size and inference speed comparison. Optionally, CKA similarity between teacher and student representations.

**What a Good Result Looks Like**: A distilled student that meaningfully outperforms training from scratch (e.g., 1-3% accuracy improvement on CIFAR-100). A clear demonstration of the effect of temperature on distillation quality. An analysis showing that distillation helps most when the student is sufficiently expressive to capture the teacher's knowledge. Visualization of what the soft targets look like (e.g., for a "cat" image, the teacher assigns non-trivial probability to "dog" and "tiger").

---

### 15. Model Interpretability Toolkit

**Difficulty**: Intermediate

**Description**: Build a unified interpretability toolkit for a trained image classification model (e.g., a ResNet on CIFAR-10 or ImageNet). Implement Grad-CAM, attention rollout (if using a ViT), feature visualization (optimizing an input to maximally activate a neuron), and saliency maps. Create an interactive dashboard (using Streamlit or Gradio) where a user can upload an image and see all interpretability visualizations side by side.

**Key Techniques**: Gradient-based attribution (saliency maps, Grad-CAM), feature visualization via activation maximization, attention visualization, interactive web application development (Streamlit or Gradio), hook-based feature extraction in PyTorch.

**Suggested Papers**:
- Selvaraju et al. "Grad-CAM: Visual Explanations from Deep Neural Networks" (2017)
- Olah et al. "Feature Visualization" (2017, Distill.pub)
- Abnar and Zuidema. "Quantifying Attention Flow in Transformers" (2020)

**Dataset**: ImageNet (use a pretrained model) or CIFAR-10 (train your own). For the dashboard, any images the user provides.

**Evaluation Metrics**: Qualitative evaluation of visualization quality, sanity checks (do attributions change when the model is randomized?), pointing game accuracy (do the attributions point to the correct object?), user feedback on the dashboard.

**What a Good Result Looks Like**: A polished dashboard where you can upload an image and instantly see Grad-CAM heatmaps, saliency maps, and the top predicted classes. A systematic comparison of different attribution methods on the same images, showing where they agree and disagree. Sanity checks demonstrating that the attributions are meaningful (not just edge detectors).

---

## Research-Level Projects

These projects are for apprentices who want to go deeper. They are more open-ended and require significant independent judgment. They also carry higher risk -- if you choose one of these, make sure your proposal has a solid backup plan.

### 16. Reproduce and Improve a Recent Paper

**Difficulty**: Research-level

**Description**: Select a paper published within the last 12 months, reproduce its core result, and then propose and test a concrete improvement. The improvement does not need to be groundbreaking -- a thoughtful modification that yields a measurable change (positive or negative) with a clear analysis of why is sufficient. The emphasis is on the process: reading the paper carefully, implementing from the description, debugging discrepancies with reported results, and thinking critically about what could be done better.

**Key Techniques**: Varies depending on the chosen paper. The meta-skill is reading and implementing from a paper, which draws on everything in Module 11.

**Suggested Papers**: Choose from recent publications at NeurIPS, ICML, ICLR, CVPR, or ACL. Pick a paper where:
- The method is described clearly enough to implement.
- The dataset is publicly available.
- The compute requirements are within your budget (avoid papers that require 100+ GPU hours to train).
- You find the problem genuinely interesting.

**Dataset**: As specified in the chosen paper.

**Evaluation Metrics**: As specified in the chosen paper. Additionally, your own metrics for the proposed improvement.

**What a Good Result Looks Like**: A faithful reproduction that achieves results within 5-10% of the paper's reported numbers (exact reproduction is often not possible due to undocumented details). A clear documentation of any discrepancies and hypotheses about their causes. A proposed improvement with a well-motivated rationale, even if the improvement does not yield better numbers. This is the project that is most like actual research.

---

### 17. Novel Data Augmentation Strategy

**Difficulty**: Research-level

**Description**: Design, implement, and rigorously evaluate a new data augmentation technique for image classification. It can be inspired by existing methods (Mixup, CutMix, AutoAugment, RandAugment) but must contain a novel element. Evaluate it on CIFAR-10, CIFAR-100, and ideally one additional dataset. Compare against a no-augmentation baseline and at least two existing augmentation methods. Analyze why your augmentation works (or does not) -- what invariances does it encode?

**Key Techniques**: Data augmentation, image transformations, training pipelines, ablation methodology, statistical testing, analysis of learned representations.

**Suggested Papers**:
- Zhang et al. "mixup: Beyond Empirical Risk Minimization" (2018)
- Yun et al. "CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features" (2019)
- Cubuk et al. "AutoAugment: Learning Augmentation Strategies from Data" (2019)
- Muller and Hutter. "TrivialAugment: Tuning-Free Yet State-of-the-Art Data Augmentation" (2021)

**Dataset**: CIFAR-10, CIFAR-100, and optionally STL-10 or Tiny ImageNet for additional validation.

**Evaluation Metrics**: Test accuracy with and without augmentation across datasets, accuracy improvement over baseline augmentation (standard random crop + horizontal flip), effect on training time, combination effects with existing augmentation methods, robustness to distribution shift (optional).

**What a Good Result Looks Like**: A clearly described novel augmentation method with a clear intuition for why it should help. A rigorous evaluation showing it provides consistent improvement across at least two datasets and two model architectures. An honest analysis of when it helps and when it does not. Even if the improvement is small, the evaluation methodology should be exemplary.

---

### 18. Efficient Fine-Tuning Comparison

**Difficulty**: Research-level

**Description**: Systematically compare parameter-efficient fine-tuning methods: LoRA, prefix tuning, adapter layers, and full fine-tuning. Evaluate on multiple downstream tasks (e.g., text classification, question answering, summarization) using a shared pretrained backbone (e.g., BERT-base or GPT-2 small). Analyze the performance-efficiency tradeoff: how many trainable parameters does each method use, and how does performance scale with the number of trainable parameters?

**Key Techniques**: Transfer learning, LoRA (low-rank adaptation), prefix tuning, adapter layers, fine-tuning, parameter-efficient methods, multi-task evaluation.

**Suggested Papers**:
- Hu et al. "LoRA: Low-Rank Adaptation of Large Language Models" (2022)
- Li and Liang. "Prefix-Tuning: Optimizing Continuous Prompts for Generation" (2021)
- Houlsby et al. "Parameter-Efficient Transfer Learning for NLP" (2019)
- He et al. "Towards a Unified View of Parameter-Efficient Transfer Learning" (2022)

**Dataset**: GLUE benchmark subset (SST-2, MRPC, QNLI at minimum), and optionally SQuAD for question answering. Use the same pretrained model (BERT-base or GPT-2 small) for all experiments.

**Evaluation Metrics**: Task accuracy/F1 for each method on each task, number of trainable parameters, training time, GPU memory usage, convergence speed (performance vs. training steps). Create efficiency frontiers plotting performance against parameter count.

**What a Good Result Looks Like**: A comprehensive comparison table showing all methods across all tasks. Clear efficiency frontier plots. An analysis of which method works best in which regime (very few parameters, moderate parameters, many parameters). Insights into when full fine-tuning is worth the cost and when parameter-efficient methods are sufficient. This project's value is in the thoroughness and fairness of the comparison, not in any single result.

---

## Choosing Your Project

Consider the following when selecting your capstone:

1. **Interest**: You will spend 2-3 weeks on this. Pick something you genuinely want to understand, not something that looks impressive on paper.

2. **Feasibility**: Can you get data? Do you have enough compute? Is the core experiment achievable in a week, leaving time for analysis and writing?

3. **Stretch**: The project should push you beyond your comfort zone in at least one dimension -- a technique you have not implemented, a domain you are less familiar with, or a scale you have not worked at before.

4. **Backup plan**: Every project should have a simpler fallback. If the diffusion model does not work, can you still submit a strong project with a simpler generative model? If the NAS search finds nothing useful, is the analysis of the search space itself interesting?

5. **Originality**: You do not need to invent something new. But you should bring your own thinking to the project. A reproduction with insightful analysis is better than a novel idea executed carelessly.

If you have your own project idea that is not on this list, discuss it with your mentor. The best capstone projects are often ones the apprentice is personally motivated to pursue.
