import { motion } from 'framer-motion'
import { Slide } from '../deck/Slide'

const RANKING = [
  { rank: 1,  id: 'IMG-02', desc: 'SVHN {0-4} → MNIST {0-4}',  acc: '98.46%', std: '±0.001', cost: '−1.33%', highlight: true,  accent: '#10B981' },
  { rank: 2,  id: 'IMG-03', desc: 'SVHN {5-9} → MNIST {0-4}',  acc: '96.84%', std: '±0.000', cost: '−2.83%', highlight: false, accent: '#F59E0B' },
  { rank: 3,  id: 'IMG-01', desc: 'SVHN full → MNIST',          acc: '95.97%', std: '±0.000', cost: '−3.16%', highlight: false, accent: '#6366F1' },
  { rank: 4,  id: 'IMG-09', desc: 'SVHN + aug → MNIST',         acc: '95.85%', std: '±0.001', cost: '−3.19%', highlight: false, accent: '#14B8A6' },
  { rank: 5,  id: 'IMG-07', desc: 'SVHN 50% → MNIST',           acc: '95.52%', std: '±0.001', cost: '−3.52%', highlight: false, accent: '#8B5CF6' },
  { rank: 6,  id: 'IMG-05', desc: 'SVHN 10% → MNIST',           acc: '95.34%', std: '±0.000', cost: '−3.62%', highlight: false, accent: '#8B5CF6' },
  { rank: 7,  id: 'IMG-06', desc: 'SVHN 25% → MNIST',           acc: '95.29%', std: '±0.001', cost: '−3.64%', highlight: false, accent: '#8B5CF6' },
  { rank: 8,  id: 'IMG-08', desc: 'CIFAR-10 → MNIST',           acc: '94.19%', std: '±0.000', cost: '−4.87%', highlight: false, accent: '#EF4444' },
  { rank: 9,  id: 'IMG-10', desc: 'FashionMNIST → MNIST',       acc: '93.55%', std: '±0.000', cost: '−5.45%', highlight: false, accent: '#F97316' },
  { rank: 10, id: 'IMG-00', desc: 'No pretrain (baseline)',      acc: '92.33%', std: '±0.004', cost: '−6.66%', highlight: false, accent: '#6B7280' },
  { rank: 11, id: 'IMG-04', desc: 'SVHN autoencoder → MNIST',   acc: '87.71%', std: '±0.006', cost: '−11.05%', highlight: true,  accent: '#EC4899' },
]

export function Slide11b() {
  return (
    <Slide track="image" trackIndex={3} trackTotal={6}>
      <motion.h2
        initial={{ opacity: 0 }} animate={{ opacity: 1 }}
        className="font-serif-display"
        style={{ fontSize: 38, fontWeight: 600, color: '#171717', marginBottom: 4 }}>
        Full Ranking at ε = 0.5 (Tightest Privacy)
      </motion.h2>
      <motion.p
        initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.08 }}
        style={{ fontFamily: 'Atkinson Hyperlegible', fontSize: 20, color: '#6B7280', marginBottom: 16 }}>
        Mean ± std across <strong style={{ color: '#171717' }}>3 seeds</strong> (42, 123, 456). Privacy cost = accuracy lost from ε=∞ → ε=0.5.
      </motion.p>

      <div className="flex gap-6 flex-1 min-h-0">
        {/* Table */}
        <motion.div
          initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.15 }}
          style={{ flex: 1, minWidth: 0, overflowY: 'auto', borderRadius: 10, border: '1px solid #E5E5E5' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontFamily: 'Atkinson Hyperlegible' }}>
            <thead>
              <tr style={{ background: '#F9FAFB', borderBottom: '2px solid #E5E5E5' }}>
                {['Rank', 'Exp', 'Description', 'ε=0.5 Acc (mean ± std)', 'Privacy Cost'].map(h => (
                  <th key={h} style={{ padding: '10px 14px', textAlign: 'left', fontSize: 14, color: '#6B7280', fontWeight: 700, letterSpacing: 0.3 }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {RANKING.map((row, i) => (
                <motion.tr
                  key={row.id}
                  initial={{ opacity: 0, x: -8 }} animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: 0.2 + i * 0.04 }}
                  style={{
                    background: row.highlight ? `${row.accent}10` : (i % 2 === 0 ? '#FFFFFF' : '#FAFAF7'),
                    borderBottom: '1px solid #F3F4F6',
                    borderLeft: row.highlight ? `4px solid ${row.accent}` : '4px solid transparent',
                  }}>
                  <td style={{ padding: '9px 14px', fontSize: 16, color: '#9CA3AF', fontWeight: 700 }}>#{row.rank}</td>
                  <td style={{ padding: '9px 14px', fontSize: 15, color: row.accent, fontWeight: 700, fontFamily: 'JetBrains Mono, monospace' }}>{row.id}</td>
                  <td style={{ padding: '9px 14px', fontSize: 17, color: '#171717' }}>{row.desc}</td>
                  <td style={{ padding: '9px 14px', fontFamily: 'JetBrains Mono, monospace' }}>
                    <span style={{ fontSize: 17, color: row.highlight ? row.accent : '#171717', fontWeight: row.highlight ? 700 : 400 }}>{row.acc}</span>
                    <span style={{ fontSize: 13, color: '#9CA3AF', marginLeft: 4 }}>{row.std}</span>
                  </td>
                  <td style={{ padding: '9px 14px', fontSize: 17, fontFamily: 'JetBrains Mono, monospace', color: row.rank === 11 ? '#EF4444' : row.rank === 1 ? '#10B981' : '#6B7280' }}>{row.cost}</td>
                </motion.tr>
              ))}
            </tbody>
          </table>
        </motion.div>

        {/* Right: insight cards */}
        <motion.div
          initial={{ opacity: 0, x: 16 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.4 }}
          style={{ width: 340, flexShrink: 0, display: 'flex', flexDirection: 'column', gap: 14, justifyContent: 'center' }}>

          <div style={{ background: '#ECFDF5', padding: '14px 18px', border: '1px solid #A7F3D0', borderLeft: '4px solid #10B981', borderRadius: 10 }}>
            <p style={{ fontFamily: 'Atkinson Hyperlegible', fontSize: 15, color: '#065F46', fontWeight: 700, letterSpacing: 0.3, marginBottom: 6 }}>BEST CASE · IMG-02</p>
            <p style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 28, color: '#10B981', fontWeight: 700, marginBottom: 4 }}>−1.33%</p>
            <p style={{ fontFamily: 'Atkinson Hyperlegible', fontSize: 15, color: '#065F46', lineHeight: 1.5 }}>Only 1.33% accuracy loss at ε=0.5 - <strong>5× less</strong> than the no-pretrain baseline.</p>
          </div>

          <div style={{ background: '#FDF2F8', padding: '14px 18px', border: '1px solid #FBCFE8', borderLeft: '4px solid #EC4899', borderRadius: 10 }}>
            <p style={{ fontFamily: 'Atkinson Hyperlegible', fontSize: 15, color: '#9F1239', fontWeight: 700, letterSpacing: 0.3, marginBottom: 6 }}>WORST CASE · IMG-04</p>
            <p style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 28, color: '#EC4899', fontWeight: 700, marginBottom: 4 }}>−11.05%</p>
            <p style={{ fontFamily: 'Atkinson Hyperlegible', fontSize: 15, color: '#9F1239', lineHeight: 1.5 }}>Autoencoder pretrain is <strong>worse than no pretrain</strong> - bad features amplify DP noise.</p>
          </div>

          <div style={{ background: '#EEF2FF', padding: '14px 18px', border: '1px solid #C7D2FE', borderLeft: '4px solid #6366F1', borderRadius: 10 }}>
            <p style={{ fontFamily: 'Atkinson Hyperlegible', fontSize: 16, color: '#171717', lineHeight: 1.5 }}>
              <strong style={{ color: '#4F46E5' }}>Key pattern:</strong> class-level alignment &gt; domain similarity &gt; format similarity &gt; unsupervised.
            </p>
          </div>
        </motion.div>
      </div>
    </Slide>
  )
}
