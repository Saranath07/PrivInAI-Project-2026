export interface SpearmanRow {
  metric: string
  label: string
  rho: number
}

export const spearmanData: SpearmanRow[] = [
  { metric: 'fid',             label: 'FID',              rho: -0.236 },
  { metric: 'kl_estimated',    label: 'KL div',           rho: -0.055 },
  { metric: 'jsd',             label: 'JSD',              rho: -0.115 },
  { metric: 'total_variation', label: 'Total Variation',  rho: -0.115 },
  { metric: 'mmd_rbf',         label: 'MMD',              rho: -0.091 },
  { metric: 'proxy_a',         label: 'Proxy-A',          rho: -0.182 },
  { metric: 'wasserstein',     label: 'Wasserstein-1',    rho: -0.709 },
  { metric: 'mauve',           label: 'MAUVE',            rho:  0.406 },
  { metric: 'linear_probe',    label: 'Linear Probe',     rho:  0.927 },
]
