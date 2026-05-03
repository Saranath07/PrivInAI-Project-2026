# DP-ShiftBench Presentation — Design

**Date:** 2026-05-03
**Owner:** Saranath P
**Deadline driver:** 7-minute live presentation; 3 speakers

## 1. Goal

A light-themed, interactive React presentation covering the DP-ShiftBench project across three tracks (image, Markov chains, transformers placeholder). Deployed to GitHub Pages. Hard time budget: **7:00**.

## 2. Time Budget

| Segment | Speaker | Time |
|---|---|---|
| Intro: problem + setup | Speaker 1 | 0:30 |
| Image track | Saranath | 2:45 |
| Markov chains | Speaker 2 | 2:15 |
| Transformers (placeholder) | Speaker 3 | 1:00 |
| Close / takeaways | last speaker | 0:30 |

A visible segmented countdown helps speakers self-pace.

## 3. Stack

- **Build:** Vite + React 18 + TypeScript
- **Styling:** Tailwind CSS
- **Animation:** Framer Motion
- **Math:** KaTeX via `react-katex`
- **Charts:** custom SVG components (no chart library — full control)
- **Deploy:** GitHub Actions → GitHub Pages (`gh-pages` via `actions/deploy-pages`)
- **Target URL:** `https://saranath07.github.io/PrivInAI-Project-2026/`
- **Vite `base`:** `/PrivInAI-Project-2026/`

## 4. Directory

```
presentation/
├── index.html
├── vite.config.ts
├── tsconfig.json
├── package.json
├── tailwind.config.js
├── postcss.config.js
├── public/
│   ├── markov/image-{0..5}.png   # copied from /markov-chains-slides/
│   └── figures/fig6_hypotheses.png  # only fig6 is pasted (hypothesis diagram)
├── src/
│   ├── main.tsx
│   ├── App.tsx
│   ├── deck/
│   │   ├── Deck.tsx
│   │   ├── Slide.tsx
│   │   ├── Timer.tsx
│   │   └── nav.ts
│   ├── data/
│   │   ├── results.ts        # accuracy × ε × experiment
│   │   ├── divergences.ts    # per-experiment divergence metrics
│   │   ├── spearman.ts       # Spearman ρ per metric at each ε
│   │   └── linearProbe.ts    # (lp_acc, dp_acc) pairs
│   ├── slides/
│   │   ├── 00-title.tsx
│   │   ├── 01-intro.tsx
│   │   ├── 10-image-setup.tsx
│   │   ├── 11-image-epsilon.tsx
│   │   ├── 12-image-divergences.tsx
│   │   ├── 13-image-mauve-fail.tsx
│   │   ├── 14-image-linear-probe.tsx
│   │   ├── 15-image-mechanism.tsx
│   │   ├── 20-markov-{1..6}.tsx
│   │   ├── 30-transformers.tsx
│   │   └── 40-close.tsx
│   ├── components/
│   │   ├── Math.tsx
│   │   ├── AccuracyVsEpsilonChart.tsx
│   │   ├── SpearmanBars.tsx
│   │   ├── LinearProbeScatter.tsx
│   │   ├── EpsilonSlider.tsx
│   │   └── CountUp.tsx
│   └── styles/
│       └── index.css
└── .github/workflows/deploy.yml
```

## 5. Visual Language

- **BG:** `#FAFAF7` warm off-white
- **Ink:** `#1F2937`
- **Accent indigo:** `#6366F1`
- **Accent pink:** `#EC4899` (DP/privacy highlights)
- **Accent emerald:** `#10B981` (reveals)
- **Fonts:** Inter (UI), Instrument Serif (display), JetBrains Mono (numerics)
- **Motion:** 400–600ms cubic easing, no springs (academic tone)
- **Track pill:** top-right, colour-shifts per segment

## 6. Navigation

- **Keys:** `→`/`Space` advance, `←` back, `Home`/`End` jump, `T` start/pause timer, `R` reset timer, `F` fullscreen
- **Click:** click right half advances, left half goes back
- **URL hash:** `#/12-image-divergences` so refresh/share preserves position

## 7. Slide Plan

### Intro (2 slides, 0:30)

1. **Title** — project title, 3 authors, animated serif underline. IIT Madras DA5001.
2. **The question** — animated pipeline: Public data → Pretrain → Model → DP-SGD finetune → Private data. KaTeX overlay:
   $$\text{Excess Risk}_{\text{DP}} \approx O\!\left(\frac{\sqrt{d \log(1/\delta)}}{n\varepsilon}\right)$$
   Tagline: *Does the gap between public and private predict DP accuracy?*

### Image Track (6 slides, 2:45)

3. **Setup** — Sample SVHN/MNIST thumbnails. 10 experiments listed. KaTeX for ε sweep.
4. **Result 1: ε barely matters** — *interactive* `AccuracyVsEpsilonChart` rendered from `results.ts`. Hover a line = highlight + show value. An ε slider drives a "Δ accuracy from ε=∞" readout. ANOVA F=0.0005 fades in.
5. **Result 2: divergences fail** — *interactive* `SpearmanBars`. Bars animate upward from the per-metric ρ data. Wrong-sign Wasserstein bar flips negative and turns red. Click a bar to see scatter of that metric vs DP accuracy.
6. **MAUVE fails on CNN features** — `fig6_hypotheses.png` (kept as image — hypothesis diagram, not a chart). KaTeX for the MAUVE mixture. Text: MAUVE compressed into [0.9962, 0.9983].
7. **Reveal: linear probe predicts** — `LinearProbeScatter` (interactive scatter with hover tooltips, correlation line). `CountUp` animates ρ from 0 → **0.927**. Insight: no DP-SGD run needed.
8. **Mechanism** — short KaTeX + 2-line explanation: DP-SGD clipping effectively restricts learning to a linear head; feature discriminability is the bottleneck.

### Markov Track (6 slides, 2:15)

9–14. Each slide displays one of `image-0.png` … `image-5.png` from `/markov-chains-slides/` at full width, with a small caption and a KaTeX overlay on the relevant formula slide for
$$[T]_Q(w' \mid c) = (1-\lambda)[T]_P(w' \mid c) + \lambda \cdot \text{Uniform}(V)$$
(These are speaker-presented; we add only framing + motion on enter/exit.)

### Transformers (1 slide, 1:00)

15. **Transformers — placeholder.** Three bullets: GPT-2 XL pretrain proxy, Gaussian-noised variants (σ ∈ {0.05, 0.10, 0.20}), MAUVE scores 0.980 / 0.970 / 0.920. Light background. Easy to swap figures in later when speaker provides them.

### Close (1 slide, 0:30)

16. **Three takeaways** (staggered fade):
    1. Privacy budget matters less than pretrain *quality* once pretraining is in place.
    2. Standard divergences fail to predict DP accuracy.
    3. Linear probe accuracy (no DP needed) predicts DP finetuning with ρ = 0.927.

**Total: 16 slides.**

## 8. Data Modules

The image track uses real data from `/results/image/tables/results_table.csv`, `divergence_table.csv`, `fig7_spearman_table.csv`, and `linear_probe_results.csv`. These are inlined as TypeScript modules at build time (no network fetch at runtime — simpler for GH Pages).

- `results.ts`: `{ experiment, epsilon, accMean, accStd }[]`
- `divergences.ts`: `{ experiment, fid, kl, jsd, ..., linearProbeAcc }[]`
- `spearman.ts`: `{ metric, rhoAtEpsilon05: number }[]`
- `linearProbe.ts`: `{ id, label, lpAcc, dpAccEps05 }[]`

## 9. Interactive Components — Contracts

**AccuracyVsEpsilonChart**
- Input: `results.ts` data
- Renders: SVG with log-x axis (ε), lines per experiment, hover → tooltip + highlight
- Optional: `highlight?: string` prop to pre-highlight one experiment

**SpearmanBars**
- Input: `spearman.ts`
- Renders: horizontal bars, animates on mount (Framer Motion), negative bars red, linear-probe green
- Click bar → emits `onSelect(metric)` for optional scatter drill-down

**LinearProbeScatter**
- Input: `linearProbe.ts`
- Renders: SVG scatter with labeled points, regression line, hover tooltip showing experiment name

**EpsilonSlider**
- Controlled slider over {0.5, 1, 2, 4, 8, ∞}
- Emits current ε, used to drive a "Δ accuracy vs ε=∞" readout

## 10. Timer

Persistent pill bottom-left: `MM:SS / 7:00`. Below it a thin horizontal bar segmented by track (intro 0:30 | image 2:45 | markov 2:15 | transformers 1:00 | close 0:30); current segment highlights. Pill goes amber at 6:00, red at 6:45. Keys: `T` toggle, `R` reset.

## 11. Deployment

`.github/workflows/deploy.yml`:
- Trigger: push to `main` touching `presentation/**` or manual
- Steps: checkout → setup-node 20 → `npm ci` in `presentation/` → `npm run build` → `actions/upload-pages-artifact` → `actions/deploy-pages`
- Requires user to enable Pages → "GitHub Actions" source in repo settings (one-time; document in `presentation/README.md`)

`vite.config.ts` sets `base: '/PrivInAI-Project-2026/'` so asset paths resolve under the subpath.

## 12. Out of Scope

- Unit tests (presentation, not a product)
- Mobile layout (presented on a laptop/projector at 16:9)
- Presenter notes view (not needed for a 7-min deck)
- Speaker-specific builds

## 13. Verification

- `npm run build` succeeds with no TS errors
- `npm run dev` loads at `http://localhost:5173/PrivInAI-Project-2026/`
- Click through all 16 slides; KaTeX renders; all three interactive charts respond to input; timer counts down
- After GH Actions run, the live URL loads and is visually identical to local
