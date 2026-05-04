import { motion } from 'framer-motion'
import { Slide } from '../deck/Slide'

const rows = [
  { s: '0.00', mauve: '1.000', zone: 'clean' },
  { s: '0.05', mauve: '0.980', zone: 'clean' },
  { s: '0.10', mauve: '0.970', zone: 'clean' },
  { s: '0.20', mauve: '0.928', zone: 'clean' },
  { s: '0.25', mauve: '0.811', zone: 'transition' },
  { s: '0.26', mauve: '0.745', zone: 'transition' },
  { s: '0.27', mauve: '0.677', zone: 'transition' },
  { s: '0.28', mauve: '0.558', zone: 'transition' },
  { s: '0.30', mauve: '0.344', zone: 'transition' },
  { s: '0.35', mauve: '0.067', zone: 'collapse' },
  { s: '0.40', mauve: '0.031', zone: 'collapse' },
  { s: '0.60', mauve: '0.006', zone: 'collapse' },
]

const zoneColor: Record<string, { bg: string; text: string; border: string }> = {
  clean:      { bg: '#DBEAFE', text: '#1E3A8A', border: '#93C5FD' },
  transition: { bg: '#FEF3C7', text: '#92400E', border: '#FCD34D' },
  collapse:   { bg: '#FEE2E2', text: '#991B1B', border: '#FCA5A5' },
}

export function Slide31() {
  const base = import.meta.env.BASE_URL
  return (
    <Slide track="transformers" trackIndex={2} trackTotal={4}>
      <motion.h2
        initial={{ opacity: 0 }} animate={{ opacity: 1 }}
        className="font-serif-display"
        style={{ fontSize: 34, fontWeight: 600, color: '#171717', marginBottom: 4 }}>
        Distribution Shift vs Perturbation Scale
      </motion.h2>
      <motion.p
        initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.1 }}
        style={{ fontFamily: 'Atkinson Hyperlegible', fontSize: 17, color: '#6B7280', marginBottom: 14 }}>
        MAUVE score between clean GPT-2 XL output and perturbed output. A sharp transition band between s = 0.25 and s = 0.35.
      </motion.p>

      <div className="flex gap-5 flex-1 min-h-0">
        <motion.div
          initial={{ opacity: 0, scale: 0.98 }} animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.2, duration: 0.4 }}
          style={{
            flex: 1, background: '#FFFFFF', border: '1px solid #E5E5E5',
            borderRadius: 10, padding: 10,
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            minWidth: 0, minHeight: 0,
          }}>
          <img
            src={`${base}transformer/mauve-shift.png`}
            alt="MAUVE vs perturbation scale"
            style={{ maxWidth: '100%', maxHeight: '100%', objectFit: 'contain' }}
          />
        </motion.div>

        <motion.div
          initial={{ opacity: 0, x: 16 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.3 }}
          style={{ width: 300, flexShrink: 0, display: 'flex', flexDirection: 'column', gap: 8, minHeight: 0 }}>
          <div style={{
            fontFamily: 'Atkinson Hyperlegible', fontSize: 12, color: '#6B7280',
            fontWeight: 700, letterSpacing: 0.5, marginBottom: 2,
          }}>
            MAUVE BY SCALE
          </div>
          <div style={{
            display: 'grid', gridTemplateColumns: '1fr 1fr',
            padding: '6px 10px',
            background: '#F9FAFB', border: '1px solid #E5E7EB', borderRadius: 6,
            fontFamily: 'Atkinson Hyperlegible', fontSize: 11, fontWeight: 700,
            color: '#6B7280', letterSpacing: 0.3,
          }}>
            <span>s</span>
            <span style={{ textAlign: 'right' }}>MAUVE</span>
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 4, overflow: 'auto', minHeight: 0 }}>
            {rows.map((r, i) => {
              const c = zoneColor[r.zone]
              return (
                <motion.div
                  key={r.s}
                  initial={{ opacity: 0, x: 8 }} animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: 0.4 + i * 0.03 }}
                  style={{
                    display: 'grid', gridTemplateColumns: '1fr 1fr',
                    padding: '6px 10px',
                    background: c.bg, border: `1px solid ${c.border}`,
                    borderLeft: `3px solid ${c.text}`,
                    borderRadius: 6,
                    fontFamily: 'JetBrains Mono, monospace', fontSize: 13,
                    color: c.text,
                  }}>
                  <span style={{ fontWeight: 700 }}>{r.s}</span>
                  <span style={{ textAlign: 'right' }}>{r.mauve}</span>
                </motion.div>
              )
            })}
          </div>
        </motion.div>
      </div>
    </Slide>
  )
}
