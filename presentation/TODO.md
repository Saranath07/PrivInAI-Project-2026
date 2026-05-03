# Presentation — TODO

Tracks what's done and what still needs work. Keep this updated as things change.

---

## ✅ Done

### Infrastructure
- [x] Vite + React 18 + TypeScript scaffold, base `/PrivInAI-Project-2026/`
- [x] Tailwind via `@tailwindcss/vite`
- [x] Framer Motion + AnimatePresence slide transitions
- [x] KaTeX via `react-katex` for math
- [x] 16:9 canvas letterboxed to viewport (fullscreen-ready)
- [x] Keyboard nav (arrows, space, F fullscreen, Home/End)
- [x] **Click-to-advance disabled** — arrows only
- [x] Deploy workflow → GitHub Pages (`.github/workflows/deploy.yml`)
- [x] Reusable `Modal` overlay component (`src/components/Modal.tsx`)
- [x] Metric definitions data (`src/data/metricDefinitions.ts`) — FID, KL, JSD, TV, MMD, Proxy-A, Wasserstein, MAUVE, Linear Probe

### Slides — Intro
- [x] **00-title** — DP-ShiftBench title, authors, 128pt display font
- [x] **01-question** — Public/Private distribution clouds, animated pipeline,
      DP-SGD arrow with noise blobs, overlapping Gaussian gap viz,
      explanatory paragraph on distribution gap

### Slides — Image Track (5 slides, was 6)
- [x] **10-image-setup** — 9 experiment cards in 3×3 grid,
      click → centered Modal with per-experiment SVG visual + description + result
- [x] **11-image-epsilon** — Interactive accuracy-vs-ε chart,
      ε slider, explicit legend, ANOVA card with F/p definitions
- [x] **12-image-mechanism** — DP-SGD gradient-lifecycle SVG animation,
      motivation banner, clipping formula, 500K→5K params visual
- [x] **13-image-divergences** — Spearman bars, each bar clickable → Modal with
      metric formula, definition, intuition, mini-scatter
- [x] **14-image-mauve** — **REMOVED** (user request)
- [x] **15-image-probe** — Linear Probe definition with formula ŷ=Wφ(x)+b,
      scatter plot, ρ=0.927 CountUp, practical takeaway

### Slides — Markov Track (still using PNG images, see below)
- [x] **20-markov-setup** — image-0.png (dataset synthesis)
- [x] **21-markov-renyi** — image-1.png (Rényi vs λ)
- [x] **22-markov-sgd** — image-2.png (SGD perplexity)
- [x] **23-markov-dpsgd** — image-3.png (DP-SGD cliff)
- [x] **24-markov-insight** — image-4.png (threshold governs DP utility)

### Slides — Closing
- [x] **30-transformers** — 3 info cards (GPT-2 XL, Gaussian noise, MAUVE)
- [x] **40-takeaways** — 3 numbered findings
- [x] **41-thankyou** — Closing slide

### Design
- [x] Light theme (Swiss Minimalism)
- [x] Fonts: Crimson Pro (display), Atkinson Hyperlegible (body), JetBrains Mono (data)
- [x] Color palette: #FAFAF7 bg, #171717 ink, #6366F1 indigo, #EC4899 pink, #10B981 emerald

---

## 🔧 In Progress / Needs Work

### Font sizes (ongoing)
- [x] Slide 01 — heading 40, subtitle 22, gap paragraph 22, DP-SGD label 14
- [x] Slide 10 — heading 38, cards 22
- [x] Slide 11 — legend 20, slider 20-22, ANOVA 20-24
- [x] Slide 13 — Spearman bar labels 20, values 17, axis 15, title 36
- [x] Slide 15 — most body copy bumped to 20
- [ ] **Titles on all slides → ≥38px** (some still at 28-30)
- [ ] **Body text enforced ≥20px minimum** across slides 12, 30, 40
- [ ] Chart axis tick labels still small on AccuracyVsEpsilonChart & LinearProbeScatter
- [ ] Animations (NN pulse, flow particles) could be bigger/more visible

### Layout issues (open)
- [ ] **Slide 15 vertical empty space** — scatter plot squished top-left,
      bottom of slide is empty. Make scatter fill available height or restructure.
- [ ] **Slide 01 "DP-SGD" label** — still overlapping with model box in some sizes
- [ ] Re-verify all slides at 1920×1080 fullscreen (not just 1280×720)

### Slide 12 (DP-SGD mechanism)
- [x] Motivation banner added
- [x] Animated gradient lifecycle SVG (raw → clipped → noisy)
- [ ] **Replace abstract boxes with actual conv block diagrams**
      (draw 3×3 kernels sliding over a feature map — user request)
- [ ] **Make gradient lifecycle bigger and more organized** — currently cramped
- [ ] Bump all body text to ≥20px

### Slide 13 (divergences)
- [x] Clickable metric bars → Modal with formula
- [x] Spearman ρ primer card
- [ ] **Fill vertical space** — bars compressed, bottom empty
  (partial fix applied: rowH=40, barH=28; may need to go higher)

---

## 📋 Blocked / Waiting on Data

### Markov track — native React rewrites (user: "do not stick images")
User wants PNGs replaced with proper React components. Needs data:

- [ ] **20-markov-setup** — animated transition-matrix heatmap with λ slider.
      Have: 33K vocab, 2M tokens, orders {3,4,5}, 5000×50 samples. ✅ Can build.
- [ ] **21-markov-renyi** — interactive line chart, D_α(P‖Q) vs λ.
      **Need:** CSV/JSON with columns `alpha, order, lambda, D_alpha` (~5×3×10=150 rows)
- [ ] **22-markov-sgd** — scatter `(D_α, perplexity_SGD, order)`.
      **Need:** corresponding data rows.
- [ ] **23-markov-dpsgd** — scatter `(D_α, perplexity_DPSGD, epsilon, order)` with cliff.
      **Need:** corresponding data rows.
- [ ] **24-markov-insight** — side-by-side SGD vs DP-SGD curves with threshold annotation.
      Derivable from 22 + 23 data.

**Unblock options:**
1. Point at CSV/JSON in `/results/markov/` or experiment output dir
2. Paste numbers directly from analysis notebooks
3. If only PNGs exist, keep PNGs as interim and mark as "midterm figures"

---

## 🧪 Not Yet Addressed

### Interactivity / UX
- [ ] "Previous slide" hint or progress indicator (user may prefer none — confirmed: no timer)
- [ ] Mobile / smaller screen fallback (probably low priority for in-person presentation)

### Content
- [ ] Verify 7-minute total runtime with new pacing
- [ ] Image track was 6 slides, now 5 — adjust track labels (`trackTotal`) ✅ done
- [ ] Transformers track still placeholder — waiting on speaker 3's content

### Deploy
- [ ] Final push + verify GitHub Pages live URL works
- [ ] Test all modals / interactions in deployed build (not just dev)

---

## 🐛 Known issues

- Build bundle 645KB (warn threshold 500KB) — could code-split by route, low priority
- Zsh `compdef` warning is from user's shell init, not a real build issue
