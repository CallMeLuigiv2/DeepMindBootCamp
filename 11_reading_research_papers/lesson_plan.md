# Module 11: Reading Research Papers — Lesson Plan

## Weeks 15-16 | The Skill That Separates Hobbyists from Researchers

> "At DeepMind, every Monday starts with a paper reading group. If you cannot read a paper on
> Friday and discuss it intelligently on Monday, you are not contributing. This module teaches
> you that skill."

This module overlaps with Module 10 (Performance Engineering). While you optimize systems during
the day, you read papers in the evening. This mirrors the actual rhythm of a research lab: you
always have a paper in progress.

---

## Prerequisites

- Completed Modules 1-9 (you need the technical vocabulary)
- Strong PyTorch fundamentals (Module 3-4)
- Familiarity with CNNs, RNNs, Transformers, and Generative Models (Modules 5-8)
- Mathematical maturity from Module 2

## Module Objectives

By the end of this module, you will be able to:

1. Read any ML paper and extract the key contributions within one hour
2. Critically evaluate experimental methodology and author claims
3. Translate a paper's method section into working code
4. Write a conference-quality paper review
5. Identify the gap between what a paper claims and what it demonstrates
6. Develop the beginning of research taste — knowing which problems matter

---

## Session 1: How to Read a Paper

**Duration:** 3 hours (1 hour instruction + 2 hours guided practice)

### Objectives

By the end of this session, the apprentice will:
- Apply the three-pass method (Keshav) to any ML paper
- Extract the five essential elements from a paper in under 10 minutes
- Know where to find papers and how to manage a reading queue
- Begin building a paper reading habit (target: 2 papers per week)

### Time Allocation

| Block | Duration | Activity |
|-------|----------|----------|
| 1 | 15 min | The three-pass method — lecture and demonstration |
| 2 | 15 min | What to extract from every paper — the five elements |
| 3 | 30 min | Live demo: first-pass reading of 3 papers in 15 minutes |
| 4 | 45 min | Guided practice: apprentice does first-pass on 5 papers |
| 5 | 30 min | Second-pass deep read of one selected paper |
| 6 | 15 min | Building your reading pipeline — tools and habits |
| 7 | 30 min | Discussion and debrief |

### Content

#### The Three-Pass Method (S. Keshav, 2007)

**First Pass — 5 to 10 minutes.**
The goal is triage. You are deciding whether this paper deserves more of your time.

Read:
1. Title, abstract, and keywords
2. Section headings only — get the structure
3. All figures, tables, and their captions (this is where the real results live)
4. Conclusions

After the first pass, you should be able to answer:
- What is the paper about? (one sentence)
- What type of paper is it? (new method, empirical study, theoretical analysis, survey)
- Is the paper relevant to what you are working on?
- Do the results look significant?

If the answer to the last two questions is "no," stop reading. This is not laziness — it is
time management. There are thousands of papers published each month. You cannot read them all.

**Second Pass — 1 hour.**
The goal is understanding. You are mapping the paper's argument structure.

Read the full paper, but:
- Focus on the main ideas, not the proofs
- Study every figure and table carefully — can you explain each one?
- Mark key equations but do not try to derive them yet
- Note references you should follow up on
- Write marginal notes: "I don't understand this," "This contradicts X," "Key insight"

After the second pass, you should be able to:
- Summarize the paper to someone else with supporting evidence
- Identify the paper's main contribution and how it differs from prior work
- List the strengths and weaknesses of the experimental evaluation
- Point to 2-3 things you do not fully understand

**Third Pass — 4+ hours.**
The goal is mastery. You are virtually re-creating the paper.

This is where you:
- Re-derive every key equation from first principles
- Challenge every assumption — "Why did they choose this loss function?"
- Mentally re-implement the method — could you code this?
- Identify what the paper does NOT say (missing ablations, unaddressed failure modes)
- Compare the paper's claims to the evidence presented

After the third pass, you should be able to:
- Reconstruct the entire paper from memory
- Identify implicit assumptions and potential failure modes
- Propose concrete improvements or extensions
- Implement the method from scratch

#### The Five Essential Elements

For every paper you read, extract these five things:

1. **Problem Statement** — What specific problem does this paper address? Why does it matter?
2. **Key Insight** — What is the one core idea that makes this work? (If you can't state this
   in one sentence, you don't understand the paper.)
3. **Method** — How does the approach work? What are the key technical components?
4. **Results** — What did they achieve? How does it compare to prior work?
5. **Limitations** — What doesn't work? What are the assumptions? Where might it fail?

#### Where to Find Papers

**Preprint servers:**
- arXiv (arxiv.org) — the primary source. Check cs.LG, cs.CV, cs.CL, stat.ML daily.
- Papers With Code (paperswithcode.com) — papers linked to implementations and benchmarks.

**Top conferences (in rough order of prestige for ML):**
- NeurIPS (Neural Information Processing Systems) — December
- ICML (International Conference on Machine Learning) — July
- ICLR (International Conference on Learning Representations) — May
- CVPR (Computer Vision and Pattern Recognition) — June
- ACL (Association for Computational Linguistics) — varies
- AAAI (Association for the Advancement of Artificial Intelligence) — February

**Curated sources:**
- Twitter/X ML community — follow key researchers
- Yannic Kilcher, Two Minute Papers (video explanations)
- The Gradient, Distill.pub (long-form explanations)
- Lab blogs: Google AI Blog, DeepMind Blog, OpenAI Blog, Meta AI

**Managing your queue:**
- Use a reference manager (Zotero, Mendeley, or even a simple spreadsheet)
- Maintain three lists: To Read, In Progress, Completed
- Tag papers by topic and relevance
- Keep the paper reading template (provided in this module) for every paper you finish

### Paper Reading Exercises

**Exercise 1.1:** Perform a first-pass reading of the following five papers. For each, write
the one-sentence summary and decide if you would do a second pass (justify your decision):
1. "Attention Is All You Need" (Vaswani et al., 2017)
2. "Deep Residual Learning for Image Recognition" (He et al., 2015)
3. "Generative Adversarial Nets" (Goodfellow et al., 2014)
4. "BERT: Pre-training of Deep Bidirectional Transformers" (Devlin et al., 2018)
5. "An Image is Worth 16x16 Words" (Dosovitskiy et al., 2020)

Time limit: 10 minutes per paper. No more.

**Exercise 1.2:** Choose one of the five papers above. Do a full second-pass reading. Fill out
the paper reading template (provided separately).

**Exercise 1.3:** Set up your paper reading pipeline. Create a reference manager or spreadsheet.
Add all five papers above. Add five more papers from arXiv that interest you. Commit to reading
two papers per week for the remainder of the course.

### Discussion Prompts

1. Which paper from Exercise 1.1 had the most informative figures? Why does figure quality
   correlate with paper quality?
2. During your second-pass reading, what was the most confusing part of the paper? What
   strategy would you use to resolve that confusion?
3. How would you explain the three-pass method to someone who says "I just read papers
   start to finish"? What are they missing?
4. Look at the latest arXiv submissions in cs.LG. How many papers were posted today? How does
   this reinforce the need for efficient reading strategies?

---

## Session 2: Anatomy of an ML Paper

**Duration:** 3 hours (1.5 hours instruction + 1.5 hours guided practice)

### Objectives

By the end of this session, the apprentice will:
- Understand the purpose and hidden information in each section of an ML paper
- Read a method section and map equations to architecture diagrams
- Critically evaluate an experiments section for rigor
- Use the related work section as a navigation tool for the field

### Time Allocation

| Block | Duration | Activity |
|-------|----------|----------|
| 1 | 20 min | Standard paper structure — what each section really tells you |
| 2 | 25 min | Reading the method section — following the math |
| 3 | 25 min | Reading the experiments section — evaluating evidence |
| 4 | 20 min | The related work section as a field map |
| 5 | 45 min | Guided dissection: "Attention Is All You Need" method section |
| 6 | 30 min | Practice: evaluate the experiments in a chosen paper |
| 7 | 15 min | Discussion and debrief |

### Content

#### Standard Structure and What Each Section REALLY Tells You

**Abstract (150-300 words)**
- What it says: Summary of the paper
- What it really tells you: The authors' best pitch. If the abstract is vague, the
  contribution is probably incremental. Strong papers have specific, quantitative abstracts.
- Red flag: "We propose a novel framework..." with no concrete results mentioned.

**Introduction (1-2 pages)**
- What it says: Background, motivation, contribution summary
- What it really tells you: How the authors position their work. Look for the "gap" paragraph
  — "Previous methods do X, but fail at Y. We address Y by doing Z." This reveals the key
  contribution claim.
- What to extract: The specific claims being made. Write them down. You will check each one
  against the experiments.

**Related Work (1-2 pages)**
- What it says: Survey of prior methods
- What it really tells you: The intellectual lineage of this paper. Which line of work are they
  building on? Which are they competing with? Who are they citing (and not citing)?
- How to use it: Treat this as a reading list. If you find a paper confusing, read the related
  work it cites. Build a mental graph of how papers connect.

**Method (2-4 pages)**
- What it says: Technical description of the approach
- What it really tells you: This is the core of the paper. Every equation, every design choice
  should be justified. If a choice is unjustified ("we set lambda to 0.1"), that is a
  potential weakness — it might be a sensitive hyperparameter.
- How to read it: (1) Read the text first, ignoring equations. Get the high-level flow.
  (2) Now read each equation. Map each symbol to what it represents. (3) Connect equations to
  any architecture diagrams. (4) Ask: "Could I implement this from what's written here?"

**Experiments (2-4 pages)**
- What it says: Empirical evaluation
- What it really tells you: Whether the method actually works, and under what conditions.
- What to check:
  - **Baselines:** Are they comparing against the right methods? Are baselines recent?
  - **Ablations:** Do they test each component of their method independently?
  - **Datasets:** Are datasets standard? Are they appropriate for the claims?
  - **Metrics:** Are metrics standard for the task? Do they report multiple metrics?
  - **Error bars/significance:** Do they report variance? Multiple runs?
  - **Cherry-picking:** Are they showing best results or typical results?
  - **Compute:** How much compute did this require? Could you reproduce it?

**Conclusion (0.5-1 page)**
- What it says: Summary and future work
- What it really tells you: The authors' honest assessment of limitations (sometimes). The
  "future work" section often reveals what they tried and couldn't get to work.

**Appendix**
- What it says: Supplementary material
- What it really tells you: The details they couldn't fit in the main paper. THIS IS WHERE
  THE IMPLEMENTATION DETAILS LIVE. Always read the appendix if you plan to implement.

#### Reading the Method Section

The method section is where most readers get lost. Here is a systematic approach:

1. **First, read only the text.** Skip all equations on first read. Understand the high-level
   approach. What are the inputs? What are the outputs? What are the main components?

2. **Draw the pipeline.** Before reading equations, sketch the data flow. What goes in, what
   transformations happen, what comes out.

3. **Now read each equation.** For each one:
   - What are the inputs? (left side or arguments)
   - What does it compute? (describe in words)
   - What are all the symbols? (build a notation table)
   - Why this particular form? (what would happen if you changed it?)

4. **Map equations to your pipeline diagram.** Each equation should correspond to a step.

5. **Look for the loss function.** The loss function tells you what the model is optimizing.
   Understanding the loss function is often the fastest way to understand the whole method.

6. **Identify the key equation.** Every paper has one equation that captures the core insight.
   Which one is it?

#### Evaluating Experiments

A checklist for evaluating experimental rigor:

- [ ] Are the baselines appropriate and recent?
- [ ] Are the baselines implemented fairly (same compute budget, same tuning effort)?
- [ ] Are there ablation studies testing each component?
- [ ] Are the datasets standard benchmarks?
- [ ] Are the evaluation metrics standard for this task?
- [ ] Are error bars or confidence intervals reported?
- [ ] Are results averaged over multiple random seeds?
- [ ] Is the compute budget stated?
- [ ] Could someone reproduce these results?
- [ ] Are failure cases discussed?
- [ ] Do the conclusions match the evidence?

### Paper Reading Exercises

**Exercise 2.1:** Take the "Attention Is All You Need" paper. For the method section (Section 3):
- List every equation
- For each equation, write: (a) what it computes in plain English, (b) all symbols defined,
  (c) the corresponding PyTorch operation
- Draw the full architecture by hand from the equations alone (do not look at Figure 1)
- Compare your drawing to Figure 1 — what did you miss?

**Exercise 2.2:** Take any paper from the landmark reading list in the notes. Read only the
experiments section. Answer:
- What are the baselines? Are they fair?
- Is there an ablation study? What does it reveal?
- Are error bars reported? If not, how does that affect your confidence?
- Do the authors' conclusions match the evidence in the tables?
- What experiment is missing that you would want to see?

**Exercise 2.3:** Take the related work section of "BERT" (Devlin et al., 2018). Create a
graph showing how the cited papers relate to each other. Identify the two main lines of work
that BERT builds on.

### Discussion Prompts

1. You read a paper that claims "state-of-the-art results" but only compares against methods
   from three years ago. How do you evaluate this claim?
2. A paper's ablation study shows that removing one component drops performance by 0.3%.
   Is this component important? How do you decide?
3. You notice a paper does not cite a highly relevant recent work. What might this mean?
   (Multiple interpretations are valid.)
4. The method section uses notation you have never seen. What is your strategy for decoding it?

---

## Session 3: From Paper to Code

**Duration:** 4 hours (1 hour instruction + 3 hours implementation practice)

### Objectives

By the end of this session, the apprentice will:
- Follow a systematic process for translating a paper into code
- Identify the gaps between paper descriptions and implementation requirements
- Handle common paper-to-code challenges (missing details, notation issues)
- Debug an implementation by comparing outputs to expected behavior from the paper

### Time Allocation

| Block | Duration | Activity |
|-------|----------|----------|
| 1 | 20 min | The translation process — four steps |
| 2 | 20 min | Common challenges and how to handle them |
| 3 | 20 min | Live demo: translating a simple paper section to PyTorch |
| 4 | 60 min | Guided implementation: multi-head attention from the paper |
| 5 | 60 min | Independent implementation: positional encoding from the paper |
| 6 | 30 min | Testing and debugging strategies |
| 7 | 30 min | Review and discussion |

### Content

#### The Translation Process

**Step 1: Identify the core algorithm.**
Strip away the narrative. What is the actual algorithm? Can you write it as numbered steps?
Many papers describe complex systems, but the core contribution is usually one specific
mechanism. Find it.

**Step 2: Write pseudocode from the paper.**
Before touching PyTorch, write pseudocode. This forces you to confront every ambiguity.
Your pseudocode should specify:
- Input shapes and types
- Every transformation in order
- Output shapes and types
- The loss function and how gradients flow

**Step 3: Handle the details papers skip.**
Papers are written for an audience of experts. They skip things they consider "obvious."
These are the things that will cost you days of debugging:
- **Weight initialization:** Papers rarely specify this, but it matters enormously. Check the
  appendix, check the code repository if available, or use standard initializations (Xavier,
  He, etc.) and document your choice.
- **Hyperparameters:** "We used a learning rate of 3e-4" might mean they searched over 50
  values and picked the best one. Your exact setup might need a different value.
- **Data preprocessing:** Normalization, tokenization, augmentation — these are often described
  in one sentence but can make or break reproduction.
- **Training tricks:** Gradient clipping, learning rate warmup, label smoothing — often
  mentioned in passing but critical for training stability.
- **Numerical stability:** Log-sum-exp tricks, epsilon values, clamping — never mentioned in
  papers but essential in code.

**Step 4: Implement incrementally and test components.**
Do NOT try to implement the entire system at once. Build bottom-up:
1. Implement the smallest component. Test it with known inputs and expected outputs.
2. Combine components. Test the combination.
3. Build up to the full system.
4. Train on a tiny dataset first (overfit a single batch).
5. Scale up gradually.

#### Common Paper-to-Code Challenges

**Notation inconsistencies.** Authors sometimes reuse symbols or change notation between
sections. Build a notation table as you read and update it when inconsistencies appear.

**Missing implementation details.** "We apply standard preprocessing" — standard to whom?
Strategy: check the appendix, check any linked code, check follow-up papers that cite this one,
or email the authors (they usually respond).

**"Trivially follows" that isn't trivial.** When a paper says "it is easy to show that..."
or "trivially, we have...", that is your cue to spend an hour working it out. These are the
steps where bugs hide.

**Differences between paper claims and reproducible results.** This is more common than the
field likes to admit. Your reproduction might not match the paper's numbers exactly. Document
the gap, investigate possible causes, but do not assume you are wrong — the paper might have
unreported tricks or favorable random seeds.

**Ambiguous dimensions.** Papers often describe operations without specifying exact tensor
shapes. Work out the shapes on paper before coding. Use assertions in your code to verify
shapes at every step.

### Paper Reading Exercises

**Exercise 3.1:** Take Section 3.2.2 of "Attention Is All You Need" (Multi-Head Attention).
- Write pseudocode for the multi-head attention mechanism
- List every detail you need to implement that is NOT in the paper
- Implement it in PyTorch
- Test it: create random input tensors of the correct shape, verify output shapes

**Exercise 3.2:** Implement the positional encoding from Section 3.5.
- Write the equations from the paper
- Implement in PyTorch
- Visualize the positional encodings (plot as a heatmap)
- Verify: do nearby positions have similar encodings? Does the encoding allow the model
  to attend to relative positions? (These are claims from the paper.)

**Exercise 3.3:** Find a discrepancy between a paper and its official implementation.
- Pick any paper from the landmark reading list that has a public codebase
- Compare the paper's method section to the actual code
- Document at least 3 differences (there are always differences)
- For each difference, hypothesize why the code differs from the paper

### Discussion Prompts

1. You are implementing a paper and your results are 5% worse than reported. What systematic
   debugging process would you follow?
2. A paper says "we use standard data augmentation." You find three different "standard"
   augmentation pipelines in different codebases. How do you decide which to use?
3. Is it ethical to use the author's code as a reference when reproducing a paper? Where is
   the line between "reference" and "copying"?
4. You find a bug in a famous paper's official code that would change the results. What do
   you do?

---

## Session 4: Critical Analysis and Research Taste

**Duration:** 3 hours (1.5 hours instruction + 1.5 hours practice)

### Objectives

By the end of this session, the apprentice will:
- Evaluate a paper's quality using a structured framework
- Identify red flags in methodology and presentation
- Write a conference-style paper review
- Begin developing research taste — understanding what makes a problem important

### Time Allocation

| Block | Duration | Activity |
|-------|----------|----------|
| 1 | 25 min | What makes a paper good — the four dimensions |
| 2 | 20 min | Red flags and how to spot them |
| 3 | 25 min | How to write a paper review |
| 4 | 20 min | Research taste — what makes a problem important |
| 5 | 45 min | Practice: write a review of a provided paper |
| 6 | 30 min | Peer review: exchange reviews and discuss |
| 7 | 15 min | Wrap-up: your research reading plan |

### Content

#### The Four Dimensions of Paper Quality

**1. Novelty** — Is this genuinely new?
- Does it introduce a new idea, or is it a minor variation of existing work?
- Is it novel in method, application, or both?
- Would the field be different without this paper?
- Novelty is not the same as complexity. A simple, clean idea can be highly novel.

**2. Rigor** — Is the work technically sound?
- Are the theoretical claims proven correctly?
- Are the experiments well-designed and fairly conducted?
- Are the comparisons appropriate?
- Are error bars reported? Are claims statistically significant?
- Are limitations acknowledged?

**3. Reproducibility** — Could someone else replicate this?
- Are all details provided (architecture, hyperparameters, data processing)?
- Is the compute requirement stated?
- Is code available?
- Are datasets publicly accessible?
- Would a competent graduate student be able to reproduce the main results?

**4. Significance** — Does this matter?
- Does it solve an important problem?
- Will other researchers build on this?
- Does it change how we think about something?
- Is the improvement meaningful or marginal?
- Significance is the hardest dimension to evaluate and the most important.

#### Red Flags

Learn to spot these warning signs:

- **No ablation study.** If a method has three components and the authors do not test each one
  independently, you cannot know which component actually helps.
- **Missing baselines.** Not comparing to the most obvious competing method is suspicious.
- **Unfair comparisons.** Comparing a heavily tuned model against an untuned baseline. Using
  more data or compute than the baseline. Comparing against old versions of competing methods.
- **Overclaimed results.** "Our method achieves state-of-the-art" when the improvement is
  within noise. Claims of generality based on one or two datasets.
- **Moving the goalposts.** Introducing a new metric that happens to favor the proposed method.
- **No error analysis.** No discussion of when or why the method fails.
- **Suspiciously clean results.** Real experiments are messy. If every result is perfect, be
  skeptical.
- **Vague descriptions.** "We found that X works better" without specifying by how much, on
  what data, or under what conditions.
- **Missing details in method section.** If you cannot implement the method from the paper
  alone, crucial information is missing.
- **Inconsistent notation.** Sloppy notation often correlates with sloppy thinking.

#### How to Write a Paper Review

Conference reviews follow a standard format. Learning to write reviews makes you a better reader.

**Structure of a review:**

1. **Summary (1 paragraph).** Demonstrate that you read and understood the paper. State the
   problem, method, and main result in your own words.

2. **Strengths (3-5 bullet points).** What is good about this paper? Be specific and generous.
   Good reviewing acknowledges genuine contributions.

3. **Weaknesses (3-5 bullet points).** What are the problems? Be specific and constructive.
   "The experiments are weak" is useless. "The experiments lack comparison to Method X, which
   addresses the same problem and was published at ICML 2023" is useful.

4. **Questions for the Authors (2-5 questions).** Things that are unclear or that you would
   like the authors to address. These should be genuine questions, not disguised criticisms.

5. **Minor Issues.** Typos, formatting issues, unclear figures. These should not affect the
   overall assessment.

6. **Overall Assessment.** A score (conferences use various scales) and a 2-3 sentence
   justification. Would you accept this paper? Why or why not?

**Review ethics:**
- Be respectful. A real person spent months on this paper.
- Be specific. Vague criticism is useless.
- Be constructive. Suggest how weaknesses could be addressed.
- Be honest. Do not inflate or deflate your assessment.
- Declare conflicts of interest.

#### Developing Research Taste

Research taste is the ability to identify which problems are worth working on. It is developed
over years of reading and thinking, but the process can be accelerated.

**What makes a problem important?**
- It is widely relevant (many people or systems are affected)
- It is currently unsolved or poorly solved
- A solution would enable new capabilities
- The problem is well-defined enough to make progress on

**Incremental vs. Transformative work:**
- Incremental: "We improve accuracy on ImageNet by 0.3% using a new regularization trick."
  This has value but rarely changes the field.
- Transformative: "We show that a simple attention mechanism can replace recurrence entirely."
  This changes how everyone thinks about the problem.
- Most good research falls between these extremes. The key is to recognize the difference.

**How to develop taste:**
1. Read broadly — not just your subfield
2. Read the classics — understand why certain papers became foundational
3. Ask "So what?" — if a method is 2% better, does anyone's life change?
4. Follow the best researchers — notice which problems they choose to work on
5. Track which papers from 5 years ago are still cited — that is a signal of lasting impact
6. Discuss papers with others — your taste calibrates against the community's

### Paper Reading Exercises

**Exercise 4.1:** Read the following paper and write a full conference-style review:
"Layer Normalization" (Ba et al., 2016). Your review should include all six sections described
above. Target length: 1-2 pages.

**Exercise 4.2:** Take three papers from the landmark reading list. For each, classify it
as incremental or transformative. Justify your classification in 2-3 sentences.

**Exercise 4.3:** Look at the accepted papers from the most recent NeurIPS or ICML. Read the
titles and abstracts of 20 papers. Rank them by anticipated significance. Return to your
ranking in six months and see how your predictions match citation counts.

**Exercise 4.4:** Write a one-page "research taste statement" — what problems in ML do you
think are the most important right now, and why? This is not graded; it is a snapshot of your
current thinking that you will revise as you develop.

### Discussion Prompts

1. You are reviewing a paper that has a brilliant idea but poor experiments. Score: accept or
   reject? Where does the balance between novelty and rigor lie?
2. A paper has been cited 10,000 times but you think the method is fundamentally flawed.
   How do you reconcile citation count with your own assessment?
3. What is the most important unsolved problem in machine learning right now? What would
   a solution look like? (There is no right answer. The quality of your reasoning matters.)
4. You notice that your favorite research lab publishes mostly incremental work. Does this
   change your opinion of them? Why or why not?

---

## Weekly Schedule

### Week 15

| Day | Activity | Time |
|-----|----------|------|
| Monday | Session 1: How to Read a Paper | 3 hours |
| Tuesday | Exercise 1.1-1.3 (first-pass practice, reading pipeline setup) | 2 hours |
| Wednesday | Session 2: Anatomy of an ML Paper | 3 hours |
| Thursday | Exercise 2.1 (Transformer method section dissection) | 3 hours |
| Friday | Exercise 2.2-2.3 (experiments evaluation, related work graph) | 2 hours |
| Weekend | Begin Assignment 1: Paper Dissection | 4 hours |

### Week 16

| Day | Activity | Time |
|-----|----------|------|
| Monday | Session 3: From Paper to Code | 4 hours |
| Tuesday | Exercise 3.1-3.2 (multi-head attention, positional encoding) | 3 hours |
| Wednesday | Session 4: Critical Analysis and Research Taste | 3 hours |
| Thursday | Exercise 4.1 (write a paper review) | 2 hours |
| Friday | Exercise 3.3, Exercise 4.2-4.4 | 3 hours |
| Weekend | Continue Assignment 1, Begin Assignment 2 | 6 hours |

**Ongoing (Weeks 15-16 and beyond):**
- Read 2 papers per week using the three-pass method
- Fill out a paper reading template for each paper
- Assignment 2 (paper reproduction) extends into the capstone period
- Assignment 3 (paper improvement) is the bridge to the capstone project

---

## Assessment

| Component | Weight | Description |
|-----------|--------|-------------|
| Paper Reading Templates | 20% | Quality and depth of 6+ completed templates |
| Assignment 1: Paper Dissection | 25% | Thoroughness of the Transformer paper analysis |
| Assignment 2: Paper Reproduction | 35% | Quality of reproduction and documentation |
| Assignment 3: Paper Improvement | 20% | Creativity and rigor of the extension |

## Resources

**Essential reading about reading:**
- S. Keshav, "How to Read a Paper" (2007) — the original three-pass method paper
- Michael Mitzenmacher, "How to Read a Research Paper" — alternative perspective
- William Griswold, "How to Read an Engineering Research Paper"

**Tools:**
- Zotero (free reference manager)
- Semantic Scholar (AI-powered paper search)
- Connected Papers (visualize paper citation graphs)
- ar5iv (arXiv papers rendered as HTML for easier reading)

**Community:**
- ML Reproducibility Challenge (annual event for paper reproduction)
- Papers We Love (community for reading classic CS papers)
- r/MachineLearning paper discussions
