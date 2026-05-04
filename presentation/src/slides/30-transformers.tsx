import { motion } from 'framer-motion'
import { Slide } from '../deck/Slide'

const cards = [
  {
    title: 'Pretrain Proxy',
    color: '#6366F1',
    bg: '#EEF2FF',
    border: '#C7D2FE',
    body: 'GPT-2 XL (1.5B) generates 3,000 OpenWebText-prefixed completions (≤512 tokens, top-p=0.95, T=0.8).',
    detail: '3,000 seqs · 2,400 / 400 / 200 split',
  },
  {
    title: 'Controlled Shift',
    color: '#EC4899',
    bg: '#FDF2F8',
    border: '#FBCFE8',
    body: 'Add Gaussian noise s · σ(W) to every weight of GPT-2 XL before generation. Higher s ⇒ more drift from clean output.',
    detail: 's ∈ {0.00, 0.05, 0.10, 0.20, 0.25–0.28, 0.30, 0.35, 0.40, 0.60}',
  },
  {
    title: 'Private Fine-Tune',
    color: '#10B981',
    bg: '#ECFDF5',
    border: '#A7F3D0',
    body: 'GPT-2 Medium (355M) + LoRA (r=4, α=32) per Yu et al. 2022. Trained with and without DP-SGD (Opacus).',
    detail: 'σ ∈ {0.60, 0.64, 0.78, 0.94} → ε ∈ {8.28, 6.52, 3.66, 2.30}, δ ≈ 1.9e-4',
  },
]

export function Slide30() {
  return (
    <Slide track="transformers" trackIndex={1} trackTotal={5}>
      <motion.h2
        initial={{ opacity: 0 }} animate={{ opacity: 1 }}
        className="font-serif-display"
        style={{ fontSize: 36, fontWeight: 600, color: '#171717', marginBottom: 4 }}>
        Transformer Track: Controlled Shift via GPT-2 XL
      </motion.h2>
      <motion.p
        initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.1 }}
        style={{ fontFamily: 'Atkinson Hyperlegible', fontSize: 18, color: '#6B7280', marginBottom: 18 }}>
        Perturbing the teacher's weights lets us dial in distribution shift continuously, then measure it with MAUVE.
      </motion.p>

      <div className="flex gap-5 flex-1 min-h-0">
        {cards.map((c, i) => (
          <motion.div
            key={c.title}
            initial={{ opacity: 0, scale: 0.94 }} animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 0.2 + i * 0.1, duration: 0.35, ease: [0, 0, 0.2, 1] }}
            style={{
              flex: 1, padding: '18px 20px', background: c.bg,
              borderRadius: 12, border: `1px solid ${c.border}`,
              borderLeft: `4px solid ${c.color}`,
              display: 'flex', flexDirection: 'column',
            }}>
            <div style={{
              fontFamily: 'Atkinson Hyperlegible', fontWeight: 700, fontSize: 14,
              color: c.color, marginBottom: 10, letterSpacing: 0.4,
            }}>
              {String(i + 1).padStart(2, '0')} · {c.title.toUpperCase()}
            </div>
            <p style={{ fontFamily: 'Atkinson Hyperlegible', fontSize: 15, color: '#171717', lineHeight: 1.5, marginBottom: 10, flex: 1 }}>
              {c.body}
            </p>
            <p style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 11, color: c.color, lineHeight: 1.5 }}>
              {c.detail}
            </p>
          </motion.div>
        ))}
      </div>

      <motion.div
        initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.6 }}
        style={{
          marginTop: 14, padding: '10px 16px',
          background: '#F9FAFB', border: '1px solid #E5E7EB',
          borderRadius: 8, display: 'flex', alignItems: 'center', gap: 16,
        }}>
        <span style={{ fontFamily: 'Atkinson Hyperlegible', fontSize: 12, color: '#6B7280', fontWeight: 700, letterSpacing: 0.4 }}>
          DISTRIBUTION DISTANCE
        </span>
        <span style={{ fontFamily: 'Atkinson Hyperlegible', fontSize: 13, color: '#374151', lineHeight: 1.4 }}>
          MAUVE between generations at scale 0.00 (clean) and scale s - large gap means the fine-tune dataset is far from the pretraining distribution.
        </span>
      </motion.div>
    </Slide>
  )
}
