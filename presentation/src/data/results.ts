export interface ResultRow {
  experiment: string
  shortLabel: string
  eps05: number; eps05std: number
  eps1: number; eps1std: number
  eps2: number; eps2std: number
  eps4: number; eps4std: number
  eps8: number; eps8std: number
  epsInf: number; epsInfStd: number
}

export const EPSILONS = [0.5, 1, 2, 4, 8, Infinity] as const
export type Epsilon = typeof EPSILONS[number]

export const results: ResultRow[] = [
  { experiment: 'No pretrain',          shortLabel: 'No pretrain',     eps05: 0.923, eps05std: 0.004, eps1: 0.924, eps1std: 0.004, eps2: 0.924, eps2std: 0.004, eps4: 0.923, eps4std: 0.003, eps8: 0.923, eps8std: 0.004, epsInf: 0.990, epsInfStd: 0.001 },
  { experiment: 'SVHN full',            shortLabel: 'SVHN full',       eps05: 0.960, eps05std: 0.000, eps1: 0.961, eps1std: 0.000, eps2: 0.961, eps2std: 0.000, eps4: 0.961, eps4std: 0.001, eps8: 0.961, eps8std: 0.001, epsInf: 0.991, epsInfStd: 0.001 },
  { experiment: 'SVHN {0-4}→MNIST',     shortLabel: 'SVHN {0-4}',     eps05: 0.985, eps05std: 0.001, eps1: 0.985, eps1std: 0.001, eps2: 0.984, eps2std: 0.000, eps4: 0.984, eps4std: 0.000, eps8: 0.984, eps8std: 0.000, epsInf: 0.998, epsInfStd: 0.000 },
  { experiment: 'SVHN {5-9}→MNIST',     shortLabel: 'SVHN {5-9}',     eps05: 0.968, eps05std: 0.000, eps1: 0.968, eps1std: 0.000, eps2: 0.968, eps2std: 0.001, eps4: 0.968, eps4std: 0.001, eps8: 0.968, eps8std: 0.001, epsInf: 0.997, epsInfStd: 0.000 },
  { experiment: 'SVHN autoencoder',     shortLabel: 'SVHN AE',        eps05: 0.877, eps05std: 0.006, eps1: 0.877, eps1std: 0.003, eps2: 0.877, eps2std: 0.001, eps4: 0.877, eps4std: 0.002, eps8: 0.877, eps8std: 0.002, epsInf: 0.988, epsInfStd: 0.000 },
  { experiment: 'SVHN 10%',            shortLabel: 'SVHN 10%',       eps05: 0.953, eps05std: 0.000, eps1: 0.953, eps1std: 0.001, eps2: 0.954, eps2std: 0.001, eps4: 0.954, eps4std: 0.000, eps8: 0.953, eps8std: 0.000, epsInf: 0.990, epsInfStd: 0.000 },
  { experiment: 'SVHN 25%',            shortLabel: 'SVHN 25%',       eps05: 0.953, eps05std: 0.001, eps1: 0.953, eps1std: 0.000, eps2: 0.953, eps2std: 0.000, eps4: 0.953, eps4std: 0.000, eps8: 0.953, eps8std: 0.000, epsInf: 0.989, epsInfStd: 0.001 },
  { experiment: 'SVHN 50%',            shortLabel: 'SVHN 50%',       eps05: 0.955, eps05std: 0.000, eps1: 0.956, eps1std: 0.000, eps2: 0.956, eps2std: 0.000, eps4: 0.956, eps4std: 0.000, eps8: 0.956, eps8std: 0.000, epsInf: 0.990, epsInfStd: 0.000 },
  { experiment: 'CIFAR-10',            shortLabel: 'CIFAR-10',       eps05: 0.942, eps05std: 0.000, eps1: 0.943, eps1std: 0.000, eps2: 0.943, eps2std: 0.000, eps4: 0.943, eps4std: 0.000, eps8: 0.943, eps8std: 0.000, epsInf: 0.991, epsInfStd: 0.000 },
  { experiment: 'SVHN+Aug',            shortLabel: 'SVHN+Aug',       eps05: 0.958, eps05std: 0.001, eps1: 0.959, eps1std: 0.000, eps2: 0.959, eps2std: 0.000, eps4: 0.960, eps4std: 0.000, eps8: 0.960, eps8std: 0.000, epsInf: 0.990, epsInfStd: 0.000 },
  { experiment: 'FashionMNIST',        shortLabel: 'FashionMNIST',   eps05: 0.936, eps05std: 0.000, eps1: 0.936, eps1std: 0.000, eps2: 0.936, eps2std: 0.000, eps4: 0.936, eps4std: 0.000, eps8: 0.936, eps8std: 0.000, epsInf: 0.990, epsInfStd: 0.001 },
]

export function getAccForEpsilon(row: ResultRow, eps: Epsilon): { mean: number; std: number } {
  switch (eps) {
    case 0.5:      return { mean: row.eps05,  std: row.eps05std }
    case 1:        return { mean: row.eps1,   std: row.eps1std }
    case 2:        return { mean: row.eps2,   std: row.eps2std }
    case 4:        return { mean: row.eps4,   std: row.eps4std }
    case 8:        return { mean: row.eps8,   std: row.eps8std }
    case Infinity: return { mean: row.epsInf, std: row.epsInfStd }
    default: return { mean: 0, std: 0 }
  }
}

export const PALETTE = [
  '#6366F1','#EC4899','#10B981','#F59E0B','#3B82F6',
  '#8B5CF6','#EF4444','#14B8A6','#F97316','#84CC16','#64748B',
]
