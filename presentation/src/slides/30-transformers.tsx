import { motion } from 'framer-motion'
import { Slide } from '../deck/Slide'

const cards = [
  {
    title: 'Pretrain Proxy',
    color: '#6366F1',
    bg: '#EEF2FF',
    border: '#C7D2FE',
    body: 'GPT-2 XL generates sequences on OpenWebText prefixes',
    detail: '3,000 sequences · 1,024 tokens each',
  },
  {
    title: 'Gaussian Perturbation',
    color: '#EC4899',
    bg: '#FDF2F8',
    border: '#FBCFE8',
    body: 'Noise added to GPT-2 XL weights (σ % of weight std dev)',
    detail: 'σ ∈ {5%, 10%, 20%}',
  },
  {
    title: 'MAUVE Scores',
    color: '#10B981',
    bg: '#ECFDF5',
    border: '#A7F3D0',
    body: '',
    scores: [
      { label: '5% noise', val: '0.9801' },
      { label: '10% noise', val: '0.9701' },
      { label: '20% noise', val: '0.9200' },
    ],
  },
]

export function Slide30() {
  return (
    <Slide track="transformers">
      <motion.h2
        initial={{ opacity: 0 }} animate={{ opacity: 1 }}
        className="font-serif-display"
        style={{ fontSize: 28, fontWeight: 600, color: '#171717', marginBottom: 20 }}>
        Transformers Track: Distribution Shift via GPT-2 XL
      </motion.h2>

      <div className="flex gap-5 flex-1">
        {cards.map((c, i) => (
          <motion.div
            key={c.title}
            initial={{ opacity: 0, scale: 0.92 }} animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 0.15 + i * 0.1, duration: 0.35, ease: [0, 0, 0.2, 1] }}
            style={{
              flex: 1, padding: '20px 22px', background: c.bg,
              borderRadius: 12, border: `1px solid ${c.border}`,
            }}>
            <div style={{
              fontFamily: 'Atkinson Hyperlegible', fontWeight: 700, fontSize: 17,
              color: c.color, marginBottom: 10,
            }}>
              {c.title}
            </div>
            {c.body && (
              <p style={{ fontFamily: 'Atkinson Hyperlegible', fontSize: 14, color: '#374151', lineHeight: 1.5, marginBottom: 6 }}>
                {c.body}
              </p>
            )}
            {c.detail && (
              <p style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 12, color: c.color }}>{c.detail}</p>
            )}
            {c.scores && c.scores.map(s => (
              <div key={s.label} className="flex justify-between items-center" style={{ marginBottom: 6 }}>
                <span style={{ fontFamily: 'Atkinson Hyperlegible', fontSize: 13, color: '#374151' }}>{s.label}</span>
                <span style={{ fontFamily: 'JetBrains Mono', fontSize: 18, fontWeight: 700, color: c.color }}>{s.val}</span>
              </div>
            ))}
          </motion.div>
        ))}
      </div>

      <motion.p
        initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.6 }}
        style={{
          marginTop: 16, fontFamily: 'Atkinson Hyperlegible', fontStyle: 'italic',
          fontSize: 13, color: '#9CA3AF', textAlign: 'center',
        }}>
        Next: finetune GPT-2 Medium with/without DP → perplexity vs privacy budget — results in final report
      </motion.p>
    </Slide>
  )
}
