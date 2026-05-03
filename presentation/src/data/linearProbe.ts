export interface LinearProbeRow {
  id: string
  label: string
  lpAcc: number
  dpAccEps05: number
}

export const linearProbeData: LinearProbeRow[] = [
  { id: 'IMG-01', label: 'SVHN full',   lpAcc: 0.9698, dpAccEps05: 0.9597 },
  { id: 'IMG-02', label: 'SVHN {0-4}',  lpAcc: 0.9864, dpAccEps05: 0.9846 },
  { id: 'IMG-03', label: 'SVHN {5-9}',  lpAcc: 0.9815, dpAccEps05: 0.9684 },
  { id: 'IMG-04', label: 'SVHN AE',     lpAcc: 0.8985, dpAccEps05: 0.8771 },
  { id: 'IMG-05', label: 'SVHN 10%',    lpAcc: 0.9620, dpAccEps05: 0.9534 },
  { id: 'IMG-06', label: 'SVHN 25%',    lpAcc: 0.9663, dpAccEps05: 0.9529 },
  { id: 'IMG-07', label: 'SVHN 50%',    lpAcc: 0.9703, dpAccEps05: 0.9552 },
  { id: 'IMG-08', label: 'CIFAR-10',    lpAcc: 0.9628, dpAccEps05: 0.9419 },
  { id: 'IMG-09', label: 'SVHN+Aug',    lpAcc: 0.9713, dpAccEps05: 0.9585 },
  { id: 'IMG-10', label: 'FashionMNIST',lpAcc: 0.9434, dpAccEps05: 0.9355 },
]
