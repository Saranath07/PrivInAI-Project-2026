# DP-ShiftBench: How Distribution Shift Governs the Value of Pretraining for Private Learning

## Project Overview

**Course:** PrivAI 2026 (IIT Madras) — Krishna Pillutla
**Problem Statement:** 1.6 — The Effect of Distribution Shifts on Pretraining for Private Learning
**Project Type:** Original empirical research (with theoretical validation)
**Target:** Workshop paper at NeurIPS 2026 (Privacy in ML / Distribution Shift workshops)

---

## Key Dates

| Milestone | Date | Status |
|---|---|---|
| Project Proposal (2 pages, NeurIPS format) | March 27, 2026 | `TODO` |
| Mid-Review Report (4 pages) | April 17, 2026 | `TODO` |
| Project Presentation | April 29 - May 4, 2026 | `TODO` |
| Final Report (9 pages) | May 11, 2026 | `TODO` |

---

## Paper Thesis

> **When a model is pretrained on public data and finetuned with DP on private data, how does the distribution shift between these two datasets affect the privacy-utility tradeoff? We build a controlled benchmark (DP-ShiftBench), discover which divergence metrics predict DP degradation, and empirically test whether the interaction between shift and privacy is additive or multiplicative.**

---

## Three Core Contributions

### Contribution 1: DP-ShiftBench — A Controlled Benchmark Suite

A reusable, open-source benchmark where distribution shift is a **continuous, tunable knob**.

#### 1A. Image Track: SVHN --> MNIST

| Experiment ID | Pretrain Data | Pretrain Method | Finetune Data (DP) | Shift Control Knob | Priority |
|---|---|---|---|---|---|
| `IMG-00` | None (random init) | — | MNIST full | Baseline (no pretrain) | P0 |
| `IMG-01` | SVHN full | Supervised classification | MNIST full | Full domain gap | P0 |
| `IMG-02` | SVHN digits {0-4} | Supervised | MNIST digits {0-4} | Matched class subset | P0 |
| `IMG-03` | SVHN digits {5-9} | Supervised | MNIST digits {0-4} | Mismatched class subset | P0 |
| `IMG-04` | SVHN full | Autoencoder (unsupervised) | MNIST full | Task gap (supervised vs unsupervised pretrain) | P1 |
| `IMG-05` | SVHN 10% subset | Supervised | MNIST full | Data scarcity | P1 |
| `IMG-06` | SVHN 25% subset | Supervised | MNIST full | Data scarcity | P1 |
| `IMG-07` | SVHN 50% subset | Supervised | MNIST full | Data scarcity | P1 |
| `IMG-08` | CIFAR-10 | Supervised | MNIST full | Larger domain gap | P1 |
| `IMG-09` | SVHN + augmentation (blur, color jitter) | Supervised | MNIST full | Controlled style shift | P2 |
| `IMG-10` | FashionMNIST | Supervised | MNIST full | Same format, different semantics | P2 |

**Model:** ResNet-18 (or small CNN with ~500K params for faster iteration)

**DP Settings for each experiment:**
- epsilon in {0.5, 1.0, 2.0, 4.0, 8.0, inf (non-private)}
- delta = 1e-5
- Clipping norm C in {0.1, 1.0} (pick best after initial sweep)
- Batch size: 256 (with gradient accumulation if needed)
- Epochs: 10-20 for finetuning

#### 1B. Language Track: Controlled Markov Chains

| Experiment ID | Pretrain Distribution | Finetune Distribution (DP) | Shift Control Knob | Priority |
|---|---|---|---|---|
| `LANG-00` | None | Markov(P1) | Baseline | P0 |
| `LANG-01` | Markov(P1) | Markov(P1) | Zero shift | P0 |
| `LANG-02` | Markov(P1) | Markov(0.9*P1 + 0.1*P_uniform) | alpha=0.1 | P0 |
| `LANG-03` | Markov(P1) | Markov(0.7*P1 + 0.3*P_uniform) | alpha=0.3 | P0 |
| `LANG-04` | Markov(P1) | Markov(0.5*P1 + 0.5*P_uniform) | alpha=0.5 | P0 |
| `LANG-05` | Markov(P1) | Markov(0.3*P1 + 0.7*P_uniform) | alpha=0.7 | P0 |
| `LANG-06` | Markov(P1) | Markov(P2), independent | alpha=1.0 (max shift) | P0 |
| `LANG-07` | Small Transformer on Corpus A | Small Transformer on Corpus B (DP) | Real distribution shift | P1 |
| `LANG-08` | GPT-2 small on WikiText | GPT-2 small on medical text (DP) | Real domain shift | P2 |

**Markov Chain Details:**
- Vocabulary size: 100 tokens
- Order: k=3 (start with), then k=4, k=5
- Transition probabilities P1: randomly generated, then fixed
- Shift control: P2(alpha) = (1 - alpha) * P1 + alpha * P_uniform
- This gives EXACT KL divergence computation (no estimation needed)
- Train a small LSTM or Transformer on sequences from these chains

**Model:** 2-layer LSTM (for Markov) or GPT-2 small (for real text)

**DP Settings:** Same epsilon grid as image track.

#### 1C. Benchmark Package Design

```
dp_shift_bench/
    __init__.py
    datasets/
        image_shifts.py      # SVHN/MNIST/CIFAR shift generators
        markov_chains.py     # Controllable Markov chain distributions
        text_shifts.py       # Real text domain shift loaders
    models/
        image_models.py      # ResNet-18, SmallCNN
        language_models.py   # LSTM, small Transformer
    training/
        dp_trainer.py        # Unified DP-SGD training loop (Opacus)
        standard_trainer.py  # Non-private pretraining loop
    metrics/
        divergences.py       # All divergence measures
        evaluation.py        # Accuracy, perplexity, etc.
    configs/
        image_experiments.yaml
        language_experiments.yaml
    runners/
        run_experiment.py    # Single experiment runner
        run_sweep.py         # Full grid sweep
```

---

### Contribution 2: Divergence-as-Predictor Analysis

For EVERY experiment, compute multiple divergence measures between pretrain and finetune data, then analyze which metric best predicts the DP finetuning performance.

#### Divergence Metrics to Compute

| Metric | Implementation | Works For | Priority |
|---|---|---|---|
| **KL Divergence** | Exact (Markov), `scipy` estimated (neural) | Language (exact), Images (estimated) | P0 |
| **Reverse KL** | Same | Both | P0 |
| **Jensen-Shannon Divergence** | `scipy` | Both | P0 |
| **Total Variation** | Exact (Markov), estimated (neural) | Both | P0 |
| **MAUVE Score** | `mauve-text` package (Pillutla et al.) | Language | P0 |
| **FID (Frechet Inception Distance)** | `torch-fidelity` or `clean-fid` | Images | P0 |
| **MMD (Maximum Mean Discrepancy)** | Custom with RBF kernel | Both | P1 |
| **Proxy A-distance (H-divergence)** | Train domain classifier, use 2(1-2*error) | Both | P1 |
| **Wasserstein Distance** | `POT` library | Both | P1 |
| **CKA (Centered Kernel Alignment)** | Custom (representation similarity) | Both | P2 |

#### Analysis Plan

1. **Correlation analysis:** For each metric, compute Pearson/Spearman correlation with DP test accuracy across all experiments
2. **Predictive modeling:** Can we fit: `accuracy(epsilon, divergence) = f(epsilon, d)` where d is the divergence?
3. **Rank the metrics:** Which one has highest predictive power?
4. **Cross-domain validation:** Does the best predictor on images also work for language?

#### Key Plots

- **Scatter plot:** Divergence metric (x) vs DP accuracy (y), colored by epsilon — one plot per metric
- **Correlation heatmap:** Metrics x Epsilons, showing R-squared values
- **Radar chart:** Comparing predictive power across all metrics

---

### Contribution 3: Additive vs. Multiplicative Interaction

#### The Competing Hypotheses

**H_additive (current theory — Bassily et al. 2023):**
```
Error(epsilon, d) = C1 * (1/epsilon) + C2 * d + C3
```
The DP cost (1/epsilon) and domain shift cost (d) contribute independently.

**H_multiplicative (open question):**
```
Error(epsilon, d) = C1 * d / epsilon + C2
```
Distribution shift AMPLIFIES the privacy cost — more shift means you need even more privacy budget.

**H_threshold (Setlur et al. 2025 "Goldilocks zone"):**
```
Error(epsilon, d) = C1/epsilon + C2*d    if d < d_threshold
Error(epsilon, d) = C1/epsilon + C3      if d >= d_threshold  (pretrain useless)
```
There's a critical shift threshold beyond which public data stops helping.

#### Experimental Design

Run a **full 2D grid:**
- epsilon in {0.5, 1.0, 2.0, 4.0, 8.0, inf}
- shift_level in {0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0} (for Markov: alpha parameter)

This gives a 6 x 7 = 42 cell grid. Each cell is repeated 3 times for error bars. Total: 126 runs for language, 126 for images.

#### Analysis Plan

1. **3D surface plot:** accuracy(epsilon, shift) — the "hero figure"
2. **Model fitting:** Fit H_additive, H_multiplicative, H_threshold to the data. Report AIC/BIC.
3. **Phase diagram / heatmap:** 2D plot showing "gain from pretraining" = accuracy(pretrained) - accuracy(random_init) as a function of (epsilon, shift). Identify the Goldilocks zone boundary.
4. **Interaction test:** Statistical test (e.g., two-way ANOVA) for whether there's a significant interaction effect between epsilon and shift.

---

## Bonus Enhancements (If Time Permits)

### Enhancement A: Membership Inference Attacks Under Shift

**Motivation:** Does distribution shift affect EMPIRICAL privacy leakage?
**Method:** Run LiRA or basic shadow-model MIA on DP-finetuned models across shift levels
**Key question:** Does shift amplify or reduce MIA success rate?
**Priority:** P2
**Effort:** ~3 days

### Enhancement B: Pretraining Data Selection

**Motivation:** Given a pool of public data, can we SELECT a subset that minimizes DP finetuning loss?
**Method:** Use divergence metrics to rank public data subsets, pretrain on top-k, evaluate
**Priority:** P2
**Effort:** ~2 days

### Enhancement C: LoRA vs Full Finetuning Under Shift

**Motivation:** Does parameter-efficient finetuning (LoRA) behave differently under shift?
**Method:** Repeat key experiments with LoRA finetuning instead of full finetuning
**Priority:** P2
**Effort:** ~2 days

### Enhancement D: Theoretical Bound Derivation (Stretch)

**Motivation:** Derive a simple theoretical bound connecting distribution shift to DP finetuning error for convex/linear models
**Approach:**
- For linear models, the DP-SGD error depends on gradient norm / clipping threshold
- Under distribution shift, the gradient norms change predictably
- Can derive: `excess_risk_DP(shift) = O(d * C * sqrt(T) * sigma / n) + O(shift_distance)` where the sigma term is the DP noise
- Check if the interaction is additive or multiplicative for this simple case
**Priority:** P2 (but very high impact if achieved)
**Effort:** ~5 days

---

## Technical Stack

| Component | Tool | Notes |
|---|---|---|
| **DP Training** | [Opacus](https://opacus.ai/) v1.4+ | PyTorch-based DP-SGD |
| **Deep Learning** | PyTorch 2.x | Core framework |
| **Image Models** | torchvision (ResNet-18, custom CNN) | Pretrained or from scratch |
| **Language Models** | HuggingFace Transformers (GPT-2) | For real text experiments |
| **MAUVE Score** | `mauve-text` | Pillutla et al. |
| **FID** | `clean-fid` or `torch-fidelity` | Image quality |
| **Optimal Transport** | `POT` (Python Optimal Transport) | Wasserstein distance |
| **Experiment Tracking** | Weights & Biases or TensorBoard | Logging all runs |
| **Config Management** | Hydra or YAML configs | Reproducible experiments |
| **Compute** | V100 (16GB) | Single GPU sufficient |
| **Plotting** | matplotlib + seaborn | Publication-quality figures |
| **Paper** | LaTeX (NeurIPS 2025 template) | As required by course |

---

## Project Structure

```
PrivInAI-Project-2026/
|
|-- PROJECT_PLAN.md              # This file
|-- README.md                    # Repo overview (for GitHub)
|
|-- dp_shift_bench/              # Main benchmark package
|   |-- __init__.py
|   |-- datasets/
|   |   |-- __init__.py
|   |   |-- image_shifts.py      # SVHN/MNIST/CIFAR loaders with shift control
|   |   |-- markov_chains.py     # Controllable Markov chain data generators
|   |   |-- text_shifts.py       # Real text domain shift loaders
|   |
|   |-- models/
|   |   |-- __init__.py
|   |   |-- image_models.py      # ResNet-18, SmallCNN
|   |   |-- language_models.py   # LSTM, small Transformer
|   |
|   |-- training/
|   |   |-- __init__.py
|   |   |-- dp_trainer.py        # DP-SGD training loop (Opacus wrapper)
|   |   |-- standard_trainer.py  # Non-private pretraining loop
|   |   |-- utils.py             # LR schedulers, early stopping, etc.
|   |
|   |-- metrics/
|   |   |-- __init__.py
|   |   |-- divergences.py       # KL, JSD, TV, MAUVE, FID, MMD, Wasserstein, H-div
|   |   |-- evaluation.py        # Accuracy, perplexity, loss curves
|   |
|   |-- configs/
|   |   |-- default.yaml
|   |   |-- image_experiments.yaml
|   |   |-- language_experiments.yaml
|   |
|   |-- runners/
|       |-- __init__.py
|       |-- run_single.py        # Run one experiment
|       |-- run_sweep.py         # Run full experiment grid
|       |-- run_divergences.py   # Compute all divergence metrics
|
|-- experiments/                 # Experiment scripts and notebooks
|   |-- 01_image_baselines.py
|   |-- 02_image_shift_grid.py
|   |-- 03_language_markov.py
|   |-- 04_language_transformer.py
|   |-- 05_divergence_analysis.py
|   |-- 06_interaction_test.py
|
|-- analysis/                    # Analysis notebooks
|   |-- divergence_predictor.ipynb
|   |-- interaction_analysis.ipynb
|   |-- phase_diagram.ipynb
|   |-- hero_figures.ipynb
|
|-- results/                     # Saved results (gitignored except summaries)
|   |-- .gitkeep
|   |-- image/
|   |-- language/
|   |-- figures/
|
|-- paper/                       # LaTeX source for reports
|   |-- proposal/
|   |   |-- main.tex
|   |   |-- references.bib
|   |-- midreview/
|   |   |-- main.tex
|   |   |-- references.bib
|   |-- final/
|       |-- main.tex
|       |-- references.bib
|
|-- tests/                       # Unit tests for benchmark code
|   |-- test_datasets.py
|   |-- test_divergences.py
|   |-- test_training.py
|
|-- requirements.txt
|-- setup.py                     # Make dp_shift_bench installable
|-- .gitignore
```

---

## Execution Timeline (Aggressive — targeting April 17 mid-review with near-complete results)

### Week 1: March 22-28 (Proposal Week)
**Deadline: Proposal due March 27**

| Day | Task | Parallelizable? | Owner |
|---|---|---|---|
| Mar 22-23 | Set up repo, project structure, dependencies | No | — |
| Mar 22-23 | Read core papers: [21] Li et al., [22] Ganesh et al. | No | Team |
| Mar 23-24 | Implement `markov_chains.py` — data generator with shift control | Yes (Agent 1) | — |
| Mar 23-24 | Implement `image_shifts.py` — SVHN/MNIST loaders with subset control | Yes (Agent 2) | — |
| Mar 23-24 | Implement `divergences.py` — KL, JSD, TV (exact for Markov) | Yes (Agent 3) | — |
| Mar 24-25 | Implement `dp_trainer.py` — Opacus-based training loop | No (core dependency) | — |
| Mar 25-26 | Run first baseline experiments (IMG-00, IMG-01, LANG-00, LANG-01) | Yes | — |
| Mar 26-27 | **Write 2-page proposal** | No | Team |
| Mar 27 | **SUBMIT PROPOSAL** | — | — |

### Week 2: March 29 - April 4 (Core Experiments)

| Day | Task | Parallelizable? | Est. Time |
|---|---|---|---|
| Mar 29-30 | Run full image grid (IMG-00 to IMG-08) across all epsilons | Yes (Agent 1) | ~8 hours GPU |
| Mar 29-30 | Run full Markov language grid (LANG-00 to LANG-06) across all epsilons | Yes (Agent 2) | ~2 hours GPU |
| Mar 29-30 | Implement remaining divergence metrics (MAUVE, FID, MMD, Wasserstein) | Yes (Agent 3) | — |
| Mar 31 - Apr 1 | Compute ALL divergence metrics for ALL dataset pairs | Yes | ~4 hours |
| Apr 1-2 | Run Contribution 3 full 2D grid (epsilon x shift) — language | Yes (Agent 1) | ~6 hours |
| Apr 1-2 | Run Contribution 3 full 2D grid (epsilon x shift) — images | Yes (Agent 2) | ~12 hours |
| Apr 2-3 | Implement proxy H-divergence (domain classifier) | Yes (Agent 3) | — |
| Apr 3-4 | Run P1 experiments (IMG-04 to IMG-07, LANG-07) | Yes | ~8 hours |

### Week 3: April 5-11 (Analysis & Extra Experiments)

| Day | Task | Parallelizable? |
|---|---|---|
| Apr 5-6 | Contribution 2 analysis: correlation analysis, predictive modeling | Yes (Agent 1) |
| Apr 5-6 | Contribution 3 analysis: model fitting (additive vs multiplicative), ANOVA | Yes (Agent 2) |
| Apr 5-6 | Generate ALL hero figures (phase diagram, 3D surface, scatter plots) | Yes (Agent 3) |
| Apr 7-8 | Run Enhancement A (MIA under shift) if time permits | Optional |
| Apr 7-8 | Run Enhancement C (LoRA comparison) if time permits | Optional |
| Apr 9-10 | Run any failed/missing experiments, collect final results | — |
| Apr 10-11 | Create results summary tables | — |

### Week 4: April 12-17 (Mid-Review Report)

| Day | Task |
|---|---|
| Apr 12-13 | Write mid-review Sections 1-2 (intro, literature survey with math) |
| Apr 13-14 | Write mid-review Section 3 (experimental setup, benchmark design) |
| Apr 14-15 | Write mid-review Section 4 (results so far) |
| Apr 15-16 | Create all figures and tables for mid-review |
| Apr 16-17 | Polish, proofread, finalize |
| Apr 17 | **SUBMIT MID-REVIEW** |

### Week 5-6: April 18 - May 4 (Presentation + Remaining Work)

| Task | Timeline |
|---|---|
| Enhancement experiments (MIA, LoRA, data selection) | Apr 18-22 |
| Package dp_shift_bench for release | Apr 22-25 |
| Prepare presentation slides | Apr 25-28 |
| **PRESENT** | Apr 29 - May 4 |

### Week 7: May 5-11 (Final Report)

| Task | Timeline |
|---|---|
| Write full 9-page report | May 5-9 |
| Final figures, proofreading | May 9-10 |
| **SUBMIT FINAL REPORT** | May 11 |

---

## Key References

### Core References (from course document)

| Ref# | Authors | Title | Year | Relevance |
|---|---|---|---|---|
| [21] | Li, Tramer, Liang, Hashimoto | Large language models can be strong differentially private learners | 2022 | Why pretraining helps DP |
| [22] | Ganesh, Haghifam, Nasr, Oh, Steinke, Thakkar, Thakurta, Wang | Why is public pretraining necessary for private model training? | 2023 | Two-phase hypothesis |
| [10] | Pillutla, Swayamdipta, Zellers, Thickstun, Welleck, Choi, Harchaoui | MAUVE: Measuring the Gap Between Neural Text and Human Text | 2021 | Distribution gap measurement |
| [11] | Pillutla, Liu, Thickstun, Welleck, Swayamdipta, Zellers, Oh, Choi, Harchaoui | MAUVE Scores for Generative Models: Theory and Practice | 2023 | Extended MAUVE theory |
| [23] | Netzer et al. | Reading digits in natural images with unsupervised feature learning (SVHN) | 2011 | SVHN dataset |
| [24] | Sakarvadia et al. | Mitigating memorization in language models | 2025 | Language memorization |

### Additional Key References (from literature search)

| Authors | Title | Year | Relevance |
|---|---|---|---|
| Setlur, Thaker, Ullman | Lower Bounds for Public-Private Learning under Distribution Shift | 2025 | Goldilocks zone theory |
| Bassily, Cortes, Mao, Mohri | Differentially Private Domain Adaptation with Theoretical Guarantees | 2023 | Additive bound (we test this) |
| Ben-David, Blitzer, Crammer, et al. | A theory of learning from different domains | 2010 | Domain adaptation theory |
| Auddy, Cai, Chakraborty | Minimax and Adaptive Transfer Learning under Distributed DP | 2025 | Minimax rates for private transfer |
| Dong, Roth, Su | Gaussian Differential Privacy | 2022 | GDP framework |
| Tobaben et al. | On the Efficacy of DP Few-shot Image Classification | 2023 | DP with pretrained features |
| De, Berrada, Hayes, Smith, Balle | Unlocking High-Accuracy DP Image Classification through Scale | 2022 | Scaling + pretraining for DP |
| Li, Guo, Li, Fan, Hu, Liu | PrivLM-Bench | 2024 | Existing DP benchmark (no shift) |
| Yichuan, Kotevska, Reshniak | Assessing MIA under Distribution Shifts | 2024 | MIA + shift |

---

## Risk Register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| GPU time insufficient for full grid | Low (V100 is enough) | High | Prioritize P0 experiments; reduce grid |
| Markov chain results too "toy" | Medium | Medium | Include real text experiments (LANG-07/08) |
| No clear winner among divergence metrics | Medium | Low | This is itself a valid negative result |
| Additive vs multiplicative is inconclusive | Medium | Medium | Report the fitted models and let data speak |
| Opacus bugs / compatibility issues | Low | High | Pin versions; test early |
| Proposal too ambitious for reviewers | Low | Low | Clearly mark P0/P1/P2 priorities |

---

## Success Criteria

### Minimum Viable Paper (Course Project)
- [ ] DP-ShiftBench with image track (P0 experiments) working
- [ ] DP-ShiftBench with language Markov track (P0 experiments) working
- [ ] At least 4 divergence metrics computed and correlated with DP accuracy
- [ ] 2D grid (epsilon x shift) for either images or language
- [ ] Phase diagram / hero figure
- [ ] Clear narrative about which metric predicts DP degradation

### Full Paper (Workshop Submission)
- [ ] All of the above PLUS:
- [ ] Both image AND language tracks complete
- [ ] All 6+ divergence metrics computed and compared
- [ ] Formal additive vs multiplicative hypothesis test
- [ ] Cross-domain validation (best predictor on images also works for language?)
- [ ] Open-source benchmark package released
- [ ] At least one enhancement (MIA, LoRA, or data selection)

### Stretch Goals (Full Conference Paper)
- [ ] Theoretical bound derivation for linear models
- [ ] Enhancement D complete
- [ ] Extended to more realistic datasets (CIFAR-100, real NLP)
- [ ] Temporal shift experiments
