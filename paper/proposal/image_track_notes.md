# Image Track — Proposal Notes (Your Part)

> **Context**: 2-page proposal (NeurIPS format). Team of 3. You own the image track.
> The proposal must answer: (1) What do we do? (2) Why does it matter? + list relevant papers.
> **Important**: Section 1.6 from the course doc explicitly suggests SVHN→MNIST with subsets, measuring gap with MAUVE [11], and adding autoencoder pretraining as complexity.

---

## What Section 1.6 Actually Asks For (Image Part)

From the course project document (page 11), the instructor says:

> **Images**: You could consider pretraining on various subsets of the SVHN dataset [23] and fine-tune (with differential privacy) on MNIST. You can measure the gap between the pretraining and finetuning dataset with the approach presented in [11]. You can add a layer of complexity by using SVHN for unsupervised pretraining (e.g., using an auto-encoder) rather than supervised pretraining for classification.

So the instructor **explicitly** wants:
1. SVHN pretrain → MNIST DP-finetune
2. Various subsets of SVHN (shift control)
3. Measure the gap (divergence metrics)
4. Supervised vs unsupervised (autoencoder) pretraining comparison

---

## Your Image Experiments (for the proposal)

You only need to describe the **design** in the proposal, not results. Keep it tight — maybe 1/3 of a page for your part within the overall 2-page proposal.

### Core Experiments to Mention

| ID | Setup | What It Tests |
|---|---|---|
| IMG-00 | No pretrain → MNIST (DP) | Baseline: DP-SGD from scratch |
| IMG-01 | SVHN full → MNIST full (DP) | Full domain gap: does SVHN pretraining help under DP? |
| IMG-02 | SVHN {0–4} → MNIST {0–4} (DP) | Matched classes: pure domain gap (same labels, different style) |
| IMG-03 | SVHN {5–9} → MNIST {0–4} (DP) | Mismatched classes: domain gap + label shift |
| IMG-04 | SVHN autoencoder → MNIST (DP) | Unsupervised pretrain: does task-agnostic pretraining help? |

### DP Settings

- ε ∈ {0.5, 1.0, 2.0, 4.0, 8.0, ∞}
- δ = 10⁻⁵
- DP-SGD via Opacus
- Model: SmallCNN (~500K params) or ResNet-18 (with GroupNorm replacing BatchNorm)

### Divergence Metrics (Your Part — Images)

For each (pretrain, finetune) pair, compute:
- **FID** (Fréchet Inception Distance) via `clean-fid` — standard image distribution metric
- **Proxy A-distance** — train domain classifier between pretrain/finetune sets, PAD = 2(1 − 2·error)
- **MMD** (Maximum Mean Discrepancy) with RBF kernel on CNN features

Then correlate these with DP finetuning accuracy.

### Preprocessing Note (Important Detail for Proposal)

SVHN is 32×32 RGB, MNIST is 28×28 grayscale. Standardize to: resize MNIST → 32×32, replicate to 3 channels. Both normalized to [0,1]. This keeps the model architecture consistent across pretrain and finetune phases.

---

## Key Points to Make in the Proposal (Image-Specific Angles)

### Why SVHN → MNIST?

1. **Canonical DA benchmark** — well-studied in domain adaptation literature (Ganin & Lempitsky 2015, Tzeng et al. 2017), so baselines exist
2. **Same label space** (digits 0–9) enables clean class-subset experiments — you can disentangle domain shift from label shift
3. **Computationally cheap** — small images, enables full ε-grid sweeps with 3 seeds each
4. **Meaningful domain gap** — color vs grayscale, real-world photos vs handwritten, cluttered backgrounds vs clean — but not so large that transfer is hopeless
5. **Instructor explicitly suggested it** (ref [23] in course doc)

### Why the Subset Experiments Matter

- **IMG-02 vs IMG-03** is the key comparison: same model architecture, same DP budget, same target data — only the pretrain class overlap changes
- This isolates **label shift** from **domain shift**
- Hypothesis: at low ε, the mismatched pretrain (IMG-03) may actually *hurt* vs no pretrain (IMG-00) because DP can't afford to "unlearn" bad features
- This connects to the Goldilocks zone idea (Setlur et al. 2025)

### Why Autoencoder Pretraining (IMG-04)

- Tests whether **task-agnostic** features (learned without labels) transfer better under DP than **task-specific** features
- Motivation: unsupervised pretraining doesn't impose a decision boundary that might conflict with the private task
- Reference: Tramèr & Boneh (2021) showed that pretraining quality matters more than pretraining task for DP

---

## References You Should Cite (Image Track Relevant)

### Must-cite (from course document)

- **[21] Li et al. (2022)** — "Large language models can be strong differentially private learners" — established that pretraining dramatically helps DP learning
- **[22] Ganesh et al. (ICML 2023)** — "Why is public pretraining necessary for private model training?" — the two-phase hypothesis; theoretical foundation for why pretrain helps
- **[23] Netzer et al. (2011)** — SVHN dataset paper
- **[11] Pillutla et al. (JMLR 2023)** — MAUVE scores — the instructor's own work, suggested for measuring distribution gap
- **[30] De et al. (2022)** — "Unlocking high-accuracy differentially private image classification through scale" — key paper showing pretrain + scale enables high DP accuracy on images

### Additional references to include

- **Tramèr & Boneh (ICLR 2021)** — "Differentially Private Learning Needs Better Features (or Much More Data)" — foundational insight that DP struggles with feature learning, not optimization
- **Setlur, Thaker, Ullman (2025)** — "Lower Bounds for Public-Private Learning under Distribution Shift" — Goldilocks zone: theoretical result showing there's a critical shift threshold
- **Bassily, Cortes, Mao, Mohri (2023)** — "Differentially Private Domain Adaptation with Theoretical Guarantees" — additive error bound (your experiments test whether this holds)
- **Ben-David et al. (2010)** — "A theory of learning from different domains" — domain adaptation theory, defines H-divergence/proxy A-distance

---

## How Your Part Fits in the 2-Page Proposal

The overall proposal structure (across all 3 team members) is probably:

```
Section 1: Introduction + Problem Statement (~0.5 page)
  - What is the problem? (distribution shift + DP)
  - Why does it matter?
  - One-paragraph thesis

Section 2: Proposed Approach (~1 page)
  - 2.1 Benchmark Design (overview)
  - 2.2 Image Track (YOUR PART — ~1/3 page)
  - 2.3 Language Track (teammate's part)
  - 2.4 Analysis Plan (divergence metrics + hypothesis testing)

Section 3: Relevant Literature (~0.5 page)
  - Key papers organized by theme

References (not counted in 2 pages)
```

### Your ~1/3 page should cover:

**In ~4-5 sentences:**
1. We pretrain on SVHN and finetune with DP-SGD on MNIST across ε ∈ {0.5, 1, 2, 4, 8, ∞}
2. We control distribution shift via: (a) class-subset matching (pretrain on digits 0–4 vs 5–9), (b) data scarcity (10/25/50% of SVHN), (c) pretraining objective (supervised classification vs autoencoder)
3. For each pair, we compute FID and proxy A-distance to quantify the shift
4. We compare these against DP finetuning accuracy to find which shift metric best predicts degradation
5. Model: SmallCNN with GroupNorm, trained with Opacus DP-SGD

**Then a small table:**

| Experiment | Pretrain | Finetune (DP) | Shift Type |
|---|---|---|---|
| Baseline | None | MNIST | No pretrain |
| Full transfer | SVHN full | MNIST | Domain gap |
| Matched classes | SVHN {0–4} | MNIST {0–4} | Domain only |
| Mismatched classes | SVHN {5–9} | MNIST {0–4} | Domain + label |
| Unsupervised | SVHN (autoenc.) | MNIST | Task gap |

---

## What NOT to Put in the Proposal

- No results (you don't have any yet)
- No implementation details (Opacus API, batch sizes, learning rates)
- No code
- Don't over-explain SVHN/MNIST — just cite [23] and say "32×32 color digit images"
- Don't mention P0/P1/P2 priority system — that's internal planning

---

## Quick Sanity Check: Does This Align with 1.6?

| Course doc asks for | ✓/✗ | Where in our plan |
|---|---|---|
| Pretrain on various subsets of SVHN | ✓ | IMG-02, IMG-03, data scarcity exps |
| Fine-tune with DP on MNIST | ✓ | All IMG experiments |
| Measure gap with approach in [11] (MAUVE) | ✓ | Divergence metrics (also FID, PAD) |
| Autoencoder pretraining | ✓ | IMG-04 |
| Empirically quantify effect of shift | ✓ | Core contribution: shift vs DP accuracy |

Everything aligns. You're doing exactly what's asked, plus going further with the systematic hypothesis testing (additive vs multiplicative vs threshold).
