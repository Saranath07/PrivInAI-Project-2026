# How to Write This Proposal — Guide

> **Constraint**: 2 pages (NeurIPS format), excluding references.
> **Graded on**: time & effort evidence, not results. Negative results = fine.
> **LLM Policy**: You CANNOT use LLMs to draft/write/rewrite text. You CAN use them for coding, finding papers, brainstorming ideas. All written text must be your own.

---

## What the Proposal Must Answer (from course doc, Section 0.2)

1. **What exactly do you want to do?**
2. **Why is this problem important?**
3. **Identify relevant papers** for the literature survey

That's it. No results. No proofs. No code. Just: problem → why it matters → what you'll do → what you'll read.

---

## Suggested Page Budget

| Section | Space | What Goes Here |
|---|---|---|
| **Title + Authors** | ~header | Already set up in template |
| **1. Introduction** | ~0.5 page | Problem statement, motivation, thesis, contributions |
| **2. Proposed Approach** | ~1.0 page | Benchmark design, image track, language track, analysis plan |
| **3. Relevant Literature** | ~0.5 page | Key papers organized by theme |
| **References** | doesn't count | Already pre-loaded in `references.bib` |

Total: 2 pages of content + references on a separate page.

---

## Section 1: Introduction — What to Write

**Structure** (5-6 sentences, roughly):

1. **Setup sentence**: "Training with DP-SGD on private data achieves strong privacy guarantees, but typically at significant accuracy cost. Recent work shows pretraining on public data dramatically reduces this cost [cite Li, Ganesh, De]."

2. **The gap**: "However, in practice the public pretrain data rarely comes from the same distribution as the private finetune data. The effect of this distribution shift on the DP privacy-utility tradeoff is not well understood."

3. **Why it matters**: "If shift amplifies the privacy cost multiplicatively, practitioners need much larger privacy budgets when public and private data differ — a common real-world scenario (e.g., pretraining on web images, finetuning on medical scans)."

4. **What you propose**: "We propose DP-ShiftBench, a controlled benchmark with two tracks — images (SVHN→MNIST) and language (Markov chains) — where distribution shift is a continuous, tunable parameter."

5. **Contributions** (3 bullet points):
   - A benchmark suite where shift is controllable
   - Analysis of which divergence metric best predicts DP degradation under shift
   - Empirical test of whether shift and privacy interact additively, multiplicatively, or with a threshold (Goldilocks zone)

### What NOT to put here
- ❌ No math/equations (save space)
- ❌ No implementation details (batch size, learning rate, etc.)
- ❌ Don't explain what DP-SGD is (the reader is your instructor — he knows)
- ❌ Don't say "to the best of our knowledge" — waste of space

---

## Section 2: Proposed Approach — What to Write

Start with a **1-2 sentence overview**:
"We design a two-track benchmark where each experiment pretrains on public data, then finetunes with DP-SGD on private data. We vary the distribution shift systematically and measure its effect across ε ∈ {0.5, 1, 2, 4, 8, ∞} with δ = 10⁻⁵."

### 2.1 Image Track (YOUR part, ~1/3 page)

**What to cover (4-5 sentences + table)**:
- We pretrain on SVHN [Netzer et al.] and finetune with DP-SGD on MNIST
- Preprocessing: resize MNIST to 32×32, convert to 3-channel to match SVHN format
- Model: SmallCNN (~500K params) with GroupNorm (required for Opacus compatibility)
- Shift control mechanisms (briefly):
  - Class-subset matching: pretrain on digits {0-4}, finetune on digits {0-4} (matched) vs pretrain on {5-9} (mismatched)
  - Pretraining objective: supervised classification vs autoencoder (as suggested in [course doc 1.6])
- For each pair, compute FID and proxy A-distance to quantify the shift magnitude
- **Include the experiment table** (uncomment it in the LaTeX) — it's compact and conveys a lot in little space

**What NOT to put**:
- ❌ Don't explain what SVHN/MNIST are in detail (just "32×32 color digit images" and cite)
- ❌ Don't list hyperparameters (clipping norm, LR, batch size) — that's for the mid-review
- ❌ Don't discuss Opacus API details
- ❌ Don't say P0/P1/P2

### 2.2 Language Track (teammate, ~1/3 page)

They should cover:
- Markov chains with vocab=100, order k=3
- Shift parameter α: P₂(α) = (1-α)P₁ + αP_uniform
- Exact KL divergence (key advantage over images — no estimation)
- Small LSTM model, same ε grid

### 2.3 Analysis (joint, ~1/3 page)

This is the most important subsection — it's where the novelty is:
- Compute divergence metrics for each (pretrain, finetune) pair
- Correlate with DP accuracy → find best predictor
- **The three hypotheses** (use equations — they look good and convey precision):
  - H_additive: Error = C₁/ε + C₂·d + C₃ (Bassily et al.)
  - H_multiplicative: Error = C₁·d/ε + C₂
  - H_threshold: piecewise with critical d* (Setlur et al.)
- "We fit all three models to the 2D grid (ε × shift) and compare via AIC/BIC"

---

## Section 3: Relevant Literature — What to Write

**Keep this short and organized by theme** (~4-5 sentences covering groups of papers):

**(a) Pretraining for DP** (2 sentences):
"Li et al. [cite] showed that pretraining dramatically improves DP learning for LLMs. Ganesh et al. [cite] provided theoretical justification via a two-phase analysis. De et al. [cite] demonstrated that scale + pretraining unlocks high DP accuracy on images. Tramèr & Boneh [cite] showed that the bottleneck is feature quality, not optimization."

**(b) Distribution shift + DP** (2 sentences):
"Bassily et al. [cite] derived additive error bounds for DP domain adaptation. Setlur et al. [cite] proved lower bounds showing a 'Goldilocks zone' where public data helps only if shift is below a threshold. Ben-David et al. [cite] provide the foundational theory of domain adaptation we build on."

**(c) Gap measurement** (1 sentence):
"We measure distribution gap using the divergence frontier approach of Pillutla et al. [cite MAUVE], FID [cite], and proxy A-distance [cite Ben-David]."

### What NOT to put
- ❌ Don't summarize every paper in detail — just 1 phrase per paper saying what it does and why you cite it
- ❌ Don't include papers you haven't read at all — the instructor says "coming up with a good list requires initial reading"
- ❌ Don't copy-paste abstracts

---

## General Tips

### DO ✅
- **Be specific**: "ε ∈ {0.5, 1, 2, 4, 8, ∞}" is better than "various privacy budgets"
- **Use the table**: The image experiment table is compact and communicates your full design in 5 rows
- **Use equations for the hypotheses**: They show precision and take less space than words
- **Cite densely**: Every claim should have a citation. The instructor is the author of MAUVE — definitely cite [10, 11]
- **Name your benchmark**: "DP-ShiftBench" — it signals ambition and reusability

### DON'T ❌
- **Don't write filler**: "In recent years, differential privacy has gained significant attention..." — your reader wrote some of these papers
- **Don't over-promise**: Don't claim you'll derive theoretical bounds (that's P2/stretch)
- **Don't include results**: You have none yet. The proposal is about the plan.
- **Don't explain DP-SGD basics**: The reader (Krishna Pillutla) teaches this course
- **Don't use the word "novel"**: Let the work speak for itself

### Space-Saving Tricks
- Use `\vspace{-2mm}` between sections if tight
- Use `\small` for the experiment table
- Inline citations: "as shown by De et al.~\cite{de2022unlocking}" saves a line vs a separate sentence
- Use `\paragraph{Image Track.}` instead of `\subsection` if you need to save the subsection header space

---

## Checklist Before Submission

- [ ] Author names filled in (replace "Member 1/2/3" in `main.tex` line 19-33)
- [ ] All TODO comments removed or addressed
- [ ] `\nocite{...}` replaced with actual `\cite{}` in the text (then remove the `\nocite` line)
- [ ] Experiment table uncommented
- [ ] Spell check (there's no built-in one — read it aloud)
- [ ] PDF is exactly 2 pages (content) + references
- [ ] Every citation can be verified on Google Scholar / DBLP / Semantic Scholar (LLM policy!)
- [ ] Rebuild: `pdflatex main && bibtex main && pdflatex main && pdflatex main`
