export interface MetricDefinition {
  key: string
  name: string
  short: string
  formula: string
  definition: string
  intuition: string
  range: string
}

export const METRIC_DEFINITIONS: Record<string, MetricDefinition> = {
  fid: {
    key: 'fid', name: 'Fréchet Inception Distance', short: 'FID',
    formula: 'FID(P,Q) = \\|\\mu_P - \\mu_Q\\|_2^2 + \\text{tr}(\\Sigma_P + \\Sigma_Q - 2(\\Sigma_P \\Sigma_Q)^{1/2})',
    definition: 'Distance between two Gaussian fits in Inception-v3 feature space. Treats each distribution as N(μ, Σ) and measures how far their means and covariances differ.',
    intuition: 'Standard image-generation quality metric. Lower = more similar images.',
    range: '[0, ∞) · lower is better',
  },
  kl_estimated: {
    key: 'kl_estimated', name: 'Kullback-Leibler Divergence', short: 'KL div',
    formula: 'D_{KL}(P \\| Q) = \\sum_x P(x) \\log \\frac{P(x)}{Q(x)}',
    definition: 'Expected log-ratio of probabilities under P. Asymmetric - penalizes Q missing mass that P has.',
    intuition: '"Bits wasted" when coding samples from P using a code optimized for Q.',
    range: '[0, ∞) · lower = more similar',
  },
  jsd: {
    key: 'jsd', name: 'Jensen-Shannon Divergence', short: 'JSD',
    formula: 'JSD(P,Q) = \\tfrac{1}{2} D_{KL}(P \\| M) + \\tfrac{1}{2} D_{KL}(Q \\| M), \\; M = \\tfrac{P+Q}{2}',
    definition: 'Symmetric, bounded version of KL. Averages both directions via the midpoint distribution M.',
    intuition: 'Bounded in [0, log 2]. Well-defined when KL isn\'t.',
    range: '[0, log 2] · lower = more similar',
  },
  total_variation: {
    key: 'total_variation', name: 'Total Variation', short: 'TV',
    formula: 'TV(P,Q) = \\tfrac{1}{2} \\sum_x |P(x) - Q(x)|',
    definition: 'Half the L1 distance between probability mass functions. Exactly the largest difference P and Q can assign to any event.',
    intuition: 'Bounded in [0, 1]. Simple, interpretable.',
    range: '[0, 1] · lower = more similar',
  },
  mmd_rbf: {
    key: 'mmd_rbf', name: 'Maximum Mean Discrepancy (RBF)', short: 'MMD',
    formula: 'MMD^2(P,Q) = \\|\\mathbb{E}_P[\\phi(x)] - \\mathbb{E}_Q[\\phi(y)]\\|_{\\mathcal{H}}^2',
    definition: 'Kernel two-sample test. Measures the distance between mean embeddings of P and Q in a reproducing kernel Hilbert space (RBF kernel here).',
    intuition: 'Distribution-free. Zero iff P = Q (for characteristic kernels).',
    range: '[0, ∞) · lower = more similar',
  },
  proxy_a: {
    key: 'proxy_a', name: 'Proxy-A Distance', short: 'Proxy-A',
    formula: '\\hat{d}_A = 2(1 - 2\\varepsilon),\\; \\varepsilon = \\text{error of domain classifier}',
    definition: 'Train a classifier to distinguish P samples from Q samples. The lower its error, the more separable - and the higher the distance.',
    intuition: 'Domain adaptation classic. ε = 0.5 (can\'t tell apart) ⇒ Proxy-A = 0.',
    range: '[0, 2] · lower = more similar',
  },
  wasserstein: {
    key: 'wasserstein', name: 'Wasserstein-1 (Earth Mover)', short: 'Wasserstein',
    formula: 'W_1(P,Q) = \\inf_{\\gamma \\in \\Pi(P,Q)} \\mathbb{E}_{(x,y) \\sim \\gamma}[\\|x - y\\|]',
    definition: 'Minimum "work" to transform P into Q by moving probability mass. Respects the geometry of the underlying space.',
    intuition: 'Unlike KL/JSD, it gives meaningful gradients even when P and Q don\'t overlap.',
    range: '[0, ∞) · lower = more similar',
  },
  mauve: {
    key: 'mauve', name: 'MAUVE', short: 'MAUVE',
    formula: 'MAUVE(P,Q) = \\text{AUC of divergence curve} (P, Q) \\text{ under KL via mixtures}',
    definition: 'Computes KL divergences along a family of mixtures between P and Q, then integrates. Designed to capture both quality and coverage of generative text/image models.',
    intuition: 'Bounded in [0, 1]. 1 = identical distributions.',
    range: '[0, 1] · higher = more similar',
  },
  linear_probe: {
    key: 'linear_probe', name: 'Linear Probe Accuracy', short: 'Linear Probe',
    formula: 'LP(P,Q) = \\max_{W,b} \\mathrm{acc}(W\\phi_P(x) + b,\\; y \\mid (x,y) \\sim Q)',
    definition: 'Freeze the pretrained encoder φ_P (trained on P). Train ONLY a linear head on labeled Q. The resulting accuracy measures how class-discriminative P\'s features are for Q.',
    intuition: 'Directly tests "are these features useful for the downstream task?" - which is exactly what DP-SGD effectively does.',
    range: '[0, 1] · higher = better transfer',
  },
}
