import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Slide } from '../deck/Slide'

type View = 'bars' | 'dpsgd' | 'traj'

const VIEWS: { key: View; label: string; file: string; sub: string }[] = [
  {
    key: 'bars',
    label: 'Privacy levels',
    file: 'math-model-test-acc-provacy-levels.png',
    sub: 'Test accuracy across ε ∈ {1.87, 3.72, 5.53, 7.17} and no-DP · mean ± std, n=5 seeds',
  },
  {
    key: 'dpsgd',
    label: 'DP-SGD vs epsilon',
    file: 'math-model-dpsgd-test.png',
    sub: 'Test accuracy as a function of privacy budget (log-scaled ε)',
  },
  {
    key: 'traj',
    label: 'Trajectories',
    file: 'math-model-training-trajectories.png',
    sub: 'Eval accuracy per epoch: DP (ε = 1.87) vs no-DP · mean ± std over 5 seeds',
  },
]

export function Slide36() {
  const base = import.meta.env.BASE_URL
  const [view, setView] = useState<View>('bars')
  const current = VIEWS.find(v => v.key === view)!

  return (
    <Slide track="transformers" trackIndex={7} trackTotal={7}>
      <motion.h2
        initial={{ opacity: 0 }} animate={{ opacity: 1 }}
        className="font-serif-display"
        style={{ fontSize: 34, fontWeight: 600, color: '#171717', marginBottom: 4 }}>
        Math Model · Results
      </motion.h2>
      <motion.p
        initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.1 }}
        style={{ fontFamily: 'Atkinson Hyperlegible', fontSize: 16, color: '#6B7280', marginBottom: 10 }}>
        Test accuracy on the four target sequences under AdamW (no-DP) vs DP-SGD at four privacy budgets. Click a view.
      </motion.p>

      <div style={{ display: 'flex', gap: 6, marginBottom: 10, flexWrap: 'wrap' }}>
        {VIEWS.map(v => {
          const active = v.key === view
          return (
            <button
              key={v.key}
              onClick={() => setView(v.key)}
              style={{
                padding: '6px 14px',
                background: active ? '#0D9488' : '#F3F4F6',
                color: active ? '#FFFFFF' : '#374151',
                border: `1px solid ${active ? '#0D9488' : '#E5E7EB'}`,
                borderRadius: 6,
                fontFamily: 'Atkinson Hyperlegible',
                fontSize: 12, fontWeight: 700, letterSpacing: 0.3,
                cursor: 'pointer',
                transition: 'all 0.15s',
              }}>
              {v.label.toUpperCase()}
            </button>
          )
        })}
      </div>

      <motion.div
        key={`hdr-${view}`}
        initial={{ opacity: 0, y: 4 }} animate={{ opacity: 1, y: 0 }}
        style={{
          padding: '8px 12px', marginBottom: 10,
          background: '#F0FDFA', border: '1px solid #99F6E4',
          borderLeft: '4px solid #0D9488', borderRadius: 6,
          fontFamily: 'Atkinson Hyperlegible', fontSize: 13, color: '#115E59',
        }}>
        <strong>{current.label}.</strong> {current.sub}
      </motion.div>

      <div className="flex gap-4 flex-1 min-h-0">
        {/* Image viewer */}
        <div style={{
          flex: 1, minWidth: 0, minHeight: 0,
          background: '#FFFFFF', border: '1px solid #E5E7EB', borderRadius: 10,
          padding: 10, display: 'flex', alignItems: 'center', justifyContent: 'center',
          position: 'relative', overflow: 'hidden',
        }}>
          <AnimatePresence mode="wait">
            <motion.img
              key={current.file}
              src={`${base}math/${current.file}`}
              alt={current.label}
              initial={{ opacity: 0, scale: 0.98 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.98 }}
              transition={{ duration: 0.25 }}
              style={{ maxWidth: '100%', maxHeight: '100%', objectFit: 'contain' }}
            />
          </AnimatePresence>
        </div>

        {/* Takeaways */}
        <motion.div
          initial={{ opacity: 0, x: 8 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.2 }}
          style={{ width: 280, flexShrink: 0, display: 'flex', flexDirection: 'column', gap: 10 }}>

          <div style={{
            background: '#EEF2FF', border: '1px solid #C7D2FE',
            borderLeft: '4px solid #6366F1', borderRadius: 10,
            padding: '12px 14px',
          }}>
            <p style={{ fontFamily: 'Atkinson Hyperlegible', fontSize: 11, color: '#4F46E5', fontWeight: 700, letterSpacing: 0.4, marginBottom: 6 }}>
              SIMILAR TARGETS
            </p>
            <p style={{ fontFamily: 'Atkinson Hyperlegible', fontSize: 13, color: '#171717', lineHeight: 1.5 }}>
              <strong>11+x</strong> and <strong>x-7</strong> stay near ceiling (~0.95) even at ε = 1.87 - the pretrain prior transfers well when the shift is small.
            </p>
          </div>

          <div style={{
            background: '#FDF2F8', border: '1px solid #FBCFE8',
            borderLeft: '4px solid #EC4899', borderRadius: 10,
            padding: '12px 14px',
          }}>
            <p style={{ fontFamily: 'Atkinson Hyperlegible', fontSize: 11, color: '#BE185D', fontWeight: 700, letterSpacing: 0.4, marginBottom: 6 }}>
              DISTANT TARGETS
            </p>
            <p style={{ fontFamily: 'Atkinson Hyperlegible', fontSize: 13, color: '#171717', lineHeight: 1.5 }}>
              <strong>2·x mod</strong> drops from 0.54 to 0.40 as ε tightens; <strong>x² mod</strong> flatlines near 0.31 - polynomial shift is out of reach for this capacity.
            </p>
          </div>

          <div style={{
            background: '#FFFBEB', border: '1px solid #FDE68A',
            borderLeft: '4px solid #F59E0B', borderRadius: 10,
            padding: '12px 14px',
          }}>
            <p style={{ fontFamily: 'Atkinson Hyperlegible', fontSize: 11, color: '#B45309', fontWeight: 700, letterSpacing: 0.4, marginBottom: 6 }}>
              PRIVACY COST
            </p>
            <p style={{ fontFamily: 'Atkinson Hyperlegible', fontSize: 13, color: '#171717', lineHeight: 1.5 }}>
              Gap between no-DP and ε = 1.87 grows with distributional distance - same signature seen on Markov chains.
            </p>
          </div>
        </motion.div>
      </div>
    </Slide>
  )
}
