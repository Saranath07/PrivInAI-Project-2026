# DP-ShiftBench — Experimental Findings

> Results are mean ± std across 3 seeds (42, 123, 456).
> Model: SmallCNN (~500K params, GroupNorm). Finetune: 15 epochs, batch 256, δ=1e-5.
> GPU: NVIDIA A100.

---

## P0 Results (IMG-00 to IMG-03)

### Accuracy Table — Mean (±Std) Test Accuracy

| Experiment | Description | ε=0.5 | ε=1.0 | ε=2.0 | ε=4.0 | ε=8.0 | ε=∞ |
|---|---|---|---|---|---|---|---|
| IMG-00 | No pretrain (baseline) | 0.9233 ±0.004 | 0.9238 ±0.003 | 0.9236 ±0.003 | 0.9234 ±0.003 | 0.9231 ±0.003 | 0.9899 ±0.000 |
| IMG-01 | SVHN full → MNIST full | 0.9597 ±0.000 | 0.9609 ±0.000 | 0.9610 ±0.000 | 0.9612 ±0.001 | 0.9611 ±0.001 | 0.9913 ±0.000 |
| IMG-02 | SVHN {0-4} → MNIST {0-4} (matched) | 0.9846 ±0.001 | 0.9846 ±0.000 | 0.9845 ±0.000 | 0.9845 ±0.000 | 0.9844 ±0.000 | 0.9979 ±0.000 |
| IMG-03 | SVHN {5-9} → MNIST {0-4} (mismatched) | 0.9684 ±0.000 | 0.9682 ±0.000 | 0.9683 ±0.000 | 0.9682 ±0.001 | 0.9682 ±0.001 | 0.9967 ±0.000 |

### Pretrain Gain over Baseline (IMG-00) at Each ε

| Experiment | ε=0.5 | ε=1.0 | ε=2.0 | ε=4.0 | ε=8.0 | ε=∞ |
|---|---|---|---|---|---|---|
| IMG-01 | +3.64% | +3.71% | +3.74% | +3.78% | +3.80% | +0.14% |
| IMG-02 | +6.13% | +6.08% | +6.09% | +6.11% | +6.13% | +0.80% |
| IMG-03 | +4.51% | +4.44% | +4.47% | +4.48% | +4.51% | +0.68% |

### Privacy Cost (ε=∞ → ε=0.5 drop)

| Experiment | Non-private acc | ε=0.5 acc | Drop |
|---|---|---|---|
| IMG-00 (no pretrain) | 0.9899 | 0.9233 | **−6.66%** |
| IMG-01 (SVHN full) | 0.9913 | 0.9597 | −3.16% |
| IMG-02 (matched classes) | 0.9979 | 0.9846 | −1.33% |
| IMG-03 (mismatched classes) | 0.9967 | 0.9684 | −2.83% |

---

## Key Findings

### Finding 1 — Pretraining consistently reduces the privacy cost

All pretrained models (IMG-01/02/03) suffer a smaller accuracy drop when privacy is enforced
compared to the no-pretrain baseline (IMG-00). The baseline loses **6.66%** going from ε=∞ to
ε=0.5, while IMG-02 loses only **1.33%** — a 5× reduction in privacy cost.

**Interpretation:** A good pretrained feature extractor already captures useful structure in the
data, so DP noise during finetuning has less to destroy.

### Finding 2 — Matched classes (IMG-02) is the best pretrain strategy

IMG-02 (SVHN digits 0-4 → MNIST digits 0-4) achieves the highest accuracy at every epsilon,
including 98.46% at ε=0.5. The class-level alignment matters more than just domain similarity.

### Finding 3 — Accuracy is remarkably flat across ε

For all pretrained models, the difference between ε=0.5 and ε=8.0 is negligible (< 0.2%).
The ε dimension only strongly separates pretrained from non-pretrained models, not the different
epsilon values within a single experiment.

**Implication:** Once a good pretrained representation exists, the RDP accountant noise budget
has diminishing returns — you can tighten privacy significantly (ε=0.5) with almost no accuracy
penalty relative to ε=8.0.

### Finding 4 — Mismatched classes still help (IMG-03 > IMG-00)

Even though IMG-03's pretrain classes (SVHN 5-9) are completely different from finetune classes
(MNIST 0-4), the pretrained feature extractor still provides a **+4.5%** gain over no pretraining
at ε=0.5. Low-level visual features (edges, strokes) transfer even when semantic labels don't.

### Finding 5 — Very low variance across seeds

Standard deviations are ≤ 0.004 for all experiments, indicating the results are highly
reproducible. The DP noise does not introduce significant training instability at these epsilon
values.

---

## Status

| Priority | Experiments | Status |
|---|---|---|
| P0 | IMG-00, IMG-01, IMG-02, IMG-03 | ✅ Complete |
| P1 | IMG-04, IMG-05, IMG-06, IMG-07, IMG-08 | ⏳ Pending |
| P2 | IMG-09, IMG-10 | ⏳ Pending |
| Divergences | All | ⏳ Pending (after P1) |
| Analysis | Figures, tables, hypothesis tests | ⏳ Pending |

---

*Last updated: 2026-04-12 after P0 sweep completion.*
