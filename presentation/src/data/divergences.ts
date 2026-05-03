export interface DivRow {
  experiment: string
  shortLabel: string
  fid: number
  kl: number
  jsd: number
  tv: number
  mmd: number
  proxyA: number
  wasserstein: number
  mauve: number
  linearProbeAcc: number
}

export const divergences: DivRow[] = [
  { experiment: 'SVHN full',        shortLabel: 'SVHN full',   fid: 196.68, kl: 52.67,  jsd: 0.263, tv: 0.139, mmd: 0.061, proxyA: 1.974, wasserstein: 10.96, mauve: 0.9981, linearProbeAcc: 0.9698 },
  { experiment: 'SVHN {0-4}',       shortLabel: 'SVHN {0-4}',  fid: 204.84, kl: 63.76,  jsd: 0.523, tv: 0.166, mmd: 0.078, proxyA: 1.982, wasserstein:  9.02, mauve: 0.9977, linearProbeAcc: 0.9864 },
  { experiment: 'SVHN {5-9}',       shortLabel: 'SVHN {5-9}',  fid: 190.06, kl: 89.16,  jsd: 0.504, tv: 0.181, mmd: 0.137, proxyA: 1.984, wasserstein:  9.72, mauve: 0.9973, linearProbeAcc: 0.9815 },
  { experiment: 'SVHN AE',          shortLabel: 'SVHN AE',     fid: 194.57, kl: 53.45,  jsd: 0.490, tv: 0.139, mmd: 0.060, proxyA: 1.974, wasserstein: 11.02, mauve: 0.9962, linearProbeAcc: 0.8985 },
  { experiment: 'SVHN 10%',         shortLabel: 'SVHN 10%',    fid: 196.14, kl: 67.52,  jsd: 0.490, tv: 0.178, mmd: 0.103, proxyA: 1.990, wasserstein: 10.97, mauve: 0.9980, linearProbeAcc: 0.9620 },
  { experiment: 'SVHN 25%',         shortLabel: 'SVHN 25%',    fid: 194.04, kl: 56.59,  jsd: 0.430, tv: 0.143, mmd: 0.063, proxyA: 1.977, wasserstein: 10.13, mauve: 0.9983, linearProbeAcc: 0.9663 },
  { experiment: 'SVHN 50%',         shortLabel: 'SVHN 50%',    fid: 194.74, kl: 51.07,  jsd: 0.377, tv: 0.139, mmd: 0.059, proxyA: 1.972, wasserstein: 10.51, mauve: 0.9982, linearProbeAcc: 0.9703 },
  { experiment: 'CIFAR-10',         shortLabel: 'CIFAR-10',    fid: 254.79, kl: 65.84,  jsd: 0.623, tv: 0.257, mmd: 0.169, proxyA: 1.996, wasserstein: 12.75, mauve: 0.9966, linearProbeAcc: 0.9628 },
  { experiment: 'SVHN+Aug',         shortLabel: 'SVHN+Aug',    fid: 191.39, kl: 58.69,  jsd: 0.493, tv: 0.143, mmd: 0.062, proxyA: 1.982, wasserstein: 11.12, mauve: 0.9983, linearProbeAcc: 0.9713 },
  { experiment: 'FashionMNIST',     shortLabel: 'Fashion',     fid: 239.43, kl: 185.92, jsd: 0.641, tv: 0.289, mmd: 0.139, proxyA: 1.991, wasserstein: 13.41, mauve: 0.9967, linearProbeAcc: 0.9434 },
]
