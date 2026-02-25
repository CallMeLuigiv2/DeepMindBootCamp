# Learning Philosophy

How you learn matters as much as what you learn. This document lays out the pedagogical principles behind this apprenticeship. These are not arbitrary rules — they are distilled from how effective researchers actually develop expertise, and from the hard-won lessons of people who have spent years training others.

Read this document carefully. Return to it when you feel frustrated or lost. The principles here will tell you whether you are struggling productively or spinning your wheels.

---

## 1. Build From Scratch First

The central principle of this course is: **you must build it before you use it.**

Before you call `torch.nn.Linear`, you will implement a linear layer from scratch in NumPy. Before you use `torch.optim.Adam`, you will implement Adam yourself. Before you use a pre-trained Transformer, you will build a Transformer from the ground up.

This is not busywork. It is the difference between someone who uses deep learning and someone who understands it.

When you implement something from scratch, you discover things that no tutorial will teach you:

- Why certain numerical instabilities arise and how to handle them.
- What assumptions are baked into an API that the documentation does not mention.
- Where the computational cost actually comes from.
- What the gradient *really* looks like when you compute it by hand.

Once you have built something from scratch and understand it thoroughly, you earn the right to use the library version. Not before.

**The progression for every concept is the same:**

1. Understand the math on paper.
2. Implement it from scratch (NumPy or raw PyTorch tensors).
3. Verify your implementation against the library version.
4. Use the library version going forward, now with genuine understanding.

You will be tempted to skip steps 1 and 2. Do not. The understanding they provide is not optional — it is the entire point.

---

## 2. The Feynman Technique

Richard Feynman had a rule: if you cannot explain something in simple terms, you do not understand it. You only think you do.

This is the most reliable test of understanding available, and it costs nothing. Here is how to apply it:

### The Process

1. **Choose a concept** you have been studying (e.g., "batch normalization").
2. **Explain it as if you were teaching someone who knows basic math but nothing about deep learning.** Write it out or say it aloud. No jargon unless you define it.
3. **Identify the gaps.** Where did you wave your hands? Where did you say "it just works" or "it's complicated"? Those are the places where your understanding is shallow.
4. **Go back to the source material** and fill in exactly those gaps.
5. **Simplify your explanation** until it is clean, precise, and complete.

### What "Simple" Does Not Mean

Simple does not mean dumbed down. A good simple explanation is *precise*. It captures the essential mechanism without unnecessary complexity, but it does not sacrifice correctness.

"Batch normalization normalizes the activations" is not a simple explanation — it is a vague one. A good explanation would address *what* is normalized, *why* normalizing helps, *how* the learnable parameters restore representational power, and *when* it does and does not work well.

### Apply This Constantly

After every major concept, pause and try to explain it from scratch. If you keep a learning journal (recommended), write explanations in it. You will find that the act of writing forces a clarity that passive reading does not.

---

## 3. Spaced Repetition

You will forget things. This is not a character flaw — it is how human memory works. The solution is not to fight it but to design your learning around it.

### The Principle

Information decays from memory on a predictable curve (the Ebbinghaus forgetting curve). Each time you revisit a concept at the right interval, the curve flattens — you retain it longer. The optimal strategy is to revisit material at increasing intervals: one day later, then three days, then a week, then a month.

### How This Course Implements It

This course is deliberately structured so that earlier concepts reappear in later assignments:

- When you implement an LSTM in Week 7, you will need the backpropagation skills from Week 3.
- When you build a Transformer in Week 8, you will use the optimization and regularization knowledge from Weeks 4 and 5.
- When you debug a training pipeline in Week 13, everything from every prior week is fair game.

This is intentional. If you find yourself needing to revisit earlier material, that is the system working as designed.

### What You Should Do

- **Keep a running list of core concepts.** Review it weekly.
- **When you encounter something from an earlier module that feels fuzzy, do not just re-read your notes.** Re-derive the math. Re-implement it. Active recall beats passive review every time.
- **Consider using a spaced repetition tool** (Anki is a good one) for key formulas, definitions, and derivations. But only use it for things you already understand — spaced repetition reinforces memory, it does not create understanding.

---

## 4. The Three-Pass Approach to Learning

Every significant concept in this course should be learned in three passes. Each pass serves a different purpose, and skipping passes leads to shallow understanding.

### Pass 1: Intuition

**Goal:** Develop a rough, correct mental model.

Ask yourself: What is this thing *for*? What problem does it solve? What would the world look like without it?

At this stage, do not worry about mathematical precision. Use analogies, diagrams, and examples. Watch a video explanation. Read a blog post. Get the shape of the idea into your head.

**You know you are done with Pass 1 when:** You can explain, at a high level, what the concept does and why it exists, to someone who has never heard of it.

### Pass 2: Formalism

**Goal:** Understand the mathematics precisely.

Now go to the textbook or the paper. Read the definitions, the theorems, the proofs. Work through the derivations with pen and paper. Do not skip steps. Do not say "I see where this is going" and jump ahead — you are probably wrong about where it is going.

Key activities:
- Derive the key equations by hand.
- Understand every symbol in every equation.
- Identify the assumptions being made and what happens when they are violated.
- Work through the proofs of key results, even if they are marked as optional.

**You know you are done with Pass 2 when:** You can re-derive the key results without looking at your notes, and you can answer "why?" for every step in the derivation.

### Pass 3: Implementation

**Goal:** Translate the formalism into working code.

This is where understanding is truly tested. The gap between "I understand the math" and "I can make it work" is often enormous, and that gap is where the real learning happens.

Key activities:
- Implement the concept from scratch, starting with the simplest possible version.
- Test against known results (e.g., compare your gradient computation to PyTorch's autograd).
- Experiment: what happens when you change the hyperparameters? When you remove a component? When you give it adversarial inputs?

**You know you are done with Pass 3 when:** Your implementation works, you can explain every line of code, and you can predict how changes to the code will affect behavior.

### The Passes Are Not Linear

You will often loop back. Implementation (Pass 3) frequently reveals gaps in your formal understanding (Pass 2). That is expected. The three passes are a spiral, not a straight line.

---

## 5. Why We Implement Papers

Starting in Week 6, you will read and implement research papers. By Week 14, you will undertake a full paper reproduction. This is one of the most important skills you will develop.

### The Reasons

**Papers are the primary source.** Blog posts, tutorials, and videos are interpretations of papers. They are often excellent, but they are also often incomplete, outdated, or subtly wrong. The paper is the ground truth of what the authors actually did. Learning to read papers is learning to go to the source.

**Implementation reveals hidden complexity.** A paper might describe a method in a few paragraphs. The implementation might take hundreds of lines and a dozen decisions the paper does not mention. What initialization did they use? How did they handle edge cases? What hyperparameters were actually critical? You only discover these questions when you try to make it work.

**Reproduction builds research taste.** When you reproduce a paper, you develop an intuition for what is a genuinely novel contribution versus what is standard engineering. You learn which details matter and which are incidental. This is the beginning of research taste, and it cannot be taught through lectures.

**It is how researchers actually learn.** Walk into any serious research lab and ask how people learn new methods. The answer is always the same: they read the paper and implement it. This is the skill. Everything else is preparation for it.

### How to Approach a Paper

1. **First pass (10 minutes):** Read the title, abstract, introduction, figure captions, and conclusion. Understand what problem they are solving and what they claim to achieve.
2. **Second pass (1--2 hours):** Read the whole paper, but do not get stuck on the math. Understand the architecture, the experimental setup, and the results. Look at every figure and table carefully.
3. **Third pass (as long as it takes):** Go through the paper in detail. Re-derive the math. Understand every equation. Compare their approach to alternatives. Identify the key insight that makes their method work.
4. **Implementation (days to weeks):** Build it. Start simple. Compare your results to theirs. Diagnose discrepancies.

---

## 6. How to Struggle Productively vs. Spinning Your Wheels

Struggle is essential to learning. If you are never confused, you are not learning anything new. But not all struggle is productive. The difference between productive struggle and spinning your wheels is critical to recognize.

### Productive Struggle

Productive struggle has these characteristics:

- **You have a specific question**, even if you do not yet have the answer. "Why does my loss plateau after epoch 20?" is productive. "Nothing works" is not.
- **You are forming and testing hypotheses.** "I think the learning rate is too high because the loss is oscillating — let me try reducing it by a factor of 10." This is how debugging works. This is how science works.
- **You are learning something from each failed attempt.** Each experiment, even a failed one, narrows the space of possibilities. If you run the same experiment twice and expect different results, you are not struggling productively.
- **You are making progress, even if it is slow.** Progress does not mean the code works. Progress means your understanding of *why* it does not work is improving.
- **The difficulty is at the right level.** You are working at the edge of your current ability, not far beyond it. The task feels hard but not impossible.

### Spinning Your Wheels

You are spinning your wheels when:

- **You have been staring at the same error for more than an hour without a new idea.** Take a break. Do something else. Come back with fresh eyes, or ask for help.
- **You are randomly changing things** hoping something will work. Commenting out lines, changing numbers, adding print statements with no hypothesis — this is not debugging, it is gambling.
- **You are working on a problem that requires prerequisite knowledge you do not have.** If you do not understand the chain rule, you cannot debug backpropagation. Go back and learn the prerequisite.
- **You are spending all your time on setup, tooling, or infrastructure** instead of the actual learning objective. If you have spent three hours fighting a CUDA installation and have not written a single line of code, step back and use CPU (or Colab) temporarily.
- **You feel anxious rather than curious.** Productive struggle is characterized by a kind of engaged frustration — you are annoyed, but you are interested. If you feel only dread, something is wrong.

### Practical Heuristics

- **The 30-minute rule:** If you have been completely stuck for 30 minutes with no new ideas, change your approach. Try a simpler version of the problem. Read a different explanation. Draw a diagram. Implement just one piece.
- **The rubber duck rule:** Explain the problem aloud (to a person, a rubber duck, or an empty room). The act of articulating the problem often reveals the solution.
- **The "make it work, make it right, make it fast" rule:** Do not try to write elegant, efficient, correct code all at once. First, get something — anything — working. Then clean it up. Then optimize if necessary.
- **The sleep rule:** If you are stuck at the end of the day, stop. Your brain will continue working on the problem while you sleep. This is not a metaphor — it is well-documented cognitive science. Many of your best insights will come in the shower the next morning.

---

## 7. When to Ask for Help

There is no shame in asking for help. There is, however, a right way and a wrong way to do it.

### Before You Ask

Do these things first:

1. **Read the error message.** Actually read it, word by word. Many error messages tell you exactly what is wrong. Python tracebacks, in particular, are remarkably informative if you read them from the bottom up.
2. **Search for the error.** Copy the error message (without your specific file paths) and search for it. Stack Overflow, GitHub issues, and PyTorch forums contain solutions to the vast majority of common problems.
3. **Isolate the problem.** Create the smallest possible example that reproduces the issue. This process alone solves the problem more than half the time.
4. **Form a hypothesis.** What do you *think* is going wrong? Even if your hypothesis is wrong, having one demonstrates that you are engaging with the problem, not just reporting it.
5. **Try at least two different approaches.** If one approach is not working, try another. Read a different tutorial. Implement the concept in a different way.

### How to Ask

When you do ask for help, provide:

1. **What you are trying to do** (the goal, not just the immediate task).
2. **What you have tried** (specific actions, not "everything").
3. **What happened** (exact error messages, unexpected behavior, specific outputs).
4. **What you expected to happen** and why.
5. **Your hypothesis** about what might be wrong.

A well-formed request for help is itself a learning exercise. The discipline of formulating a clear question often leads you to the answer before you finish writing it.

### Bad Help Requests

- "It doesn't work." (What doesn't? What did you expect? What happened instead?)
- "I'm stuck." (On what? What have you tried?)
- "Can you just give me the code?" (No.)

### Good Help Requests

"I'm implementing batch normalization from scratch. During training, my implementation matches PyTorch's `nn.BatchNorm1d` output exactly. But during evaluation, my outputs diverge significantly. I think the issue is with how I'm tracking the running mean and variance — I'm using an exponential moving average with momentum 0.1, which matches the PyTorch default. Here's my code [code]. The discrepancy appears after about 100 training steps. Am I updating the running statistics at the wrong time?"

This request demonstrates understanding, shows specific work, identifies a precise discrepancy, and offers a hypothesis. It makes the helper's job dramatically easier and, more importantly, it shows that you are learning.

---

## The Meta-Lesson

All of these principles share a common thread: **active engagement over passive consumption.**

Reading a textbook is not learning. Watching a lecture is not learning. Running someone else's code is not learning. These are inputs to learning, but the learning itself happens when you do something difficult with the information: derive it, implement it, explain it, apply it to a new problem, debug it when it breaks.

This is harder and slower than passive consumption. That is the point. The discomfort of active engagement is the feeling of your understanding deepening. Learn to recognize it. Learn to seek it out.

There is no shortcut. But if you follow these principles consistently, you will develop a quality of understanding that shortcuts can never provide — the kind that lets you walk into a new problem, a new paper, a new domain, and figure out what is going on. That is what this apprenticeship is building toward.
