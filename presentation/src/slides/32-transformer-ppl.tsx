import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Slide } from '../deck/Slide'

type Key = 'nondp' | '828' | '652' | '366' | '230'

const epsLevels: { key: Key; eps: string; label: string; sigma: string; img: string; note: string }[] = [
  { key: 'nondp', eps: '∞',    label: 'no-DP',   sigma: '-',      img: 'ppl-nondp.png', note: 'Baseline. PPL floor ≈ 5.4; 0p35 dips to 4.6.' },
  { key: '828',   eps: '8.28', label: 'ε = 8.28', sigma: '0.5975', img: 'ppl-828.png',   note: 'Weak privacy. Still tracks non-DP shape closely.' },
  { key: '652',   eps: '6.52', label: 'ε = 6.52', sigma: '0.6440', img: 'ppl-652.png',   note: 'Moderate privacy. Cliff at 0p40 becomes sharper.' },
  { key: '366',   eps: '3.66', label: 'ε = 3.66', sigma: '0.7823', img: 'ppl-366.png',   note: 'Tight privacy. 0p40 PPL ≈ 7.3 - large penalty.' },
  { key: '230',   eps: '2.30', label: 'ε = 2.30', sigma: '0.9411', img: 'ppl-230.png',   note: 'Tightest privacy tested. Divergence amplified at every s.' },
]

export function Slide32() {
  const base = import.meta.env.BASE_URL
  const [idx, setIdx] = useState(0)
  const curr = epsLevels[idx]

  return (
    <Slide track="transformers" trackIndex={3} trackTotal={4}>
      <motion.h2
        initial={{ opacity: 0 }} animate={{ opacity: 1 }}
        className="font-serif-display"
        style={{ fontSize: 34, fontWeight: 600, color: '#171717', marginBottom: 4 }}>
        Test Perplexity Across Privacy Levels
      </motion.h2>
      <motion.p
        initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.1 }}
        style={{ fontFamily: 'Atkinson Hyperlegible', fontSize: 17, color: '#6B7280', marginBottom: 12 }}>
        GPT-2 Medium + LoRA fine-tuned on each perturbed dataset. Scroll ε to watch the privacy tax grow.
      </motion.p>

      <div className="flex gap-5 flex-1 min-h-0">
        <motion.div
          initial={{ opacity: 0, scale: 0.98 }} animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.2, duration: 0.4 }}
          style={{
            flex: 1, background: '#FFFFFF', border: '1px solid #E5E5E5',
            borderRadius: 10, padding: 10, position: 'relative',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            minWidth: 0, minHeight: 0, overflow: 'hidden',
          }}>
          <AnimatePresence mode="wait">
            <motion.img
              key={curr.key}
              src={`${base}transformer/${curr.img}`}
              alt={`Test PPL at ${curr.label}`}
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.25 }}
              style={{ maxWidth: '100%', maxHeight: '100%', objectFit: 'contain' }}
            />
          </AnimatePresence>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, x: 16 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.3 }}
          style={{ width: 320, flexShrink: 0, display: 'flex', flexDirection: 'column', gap: 14 }}>

          <div style={{
            fontFamily: 'Atkinson Hyperlegible', fontSize: 12, color: '#6B7280',
            fontWeight: 700, letterSpacing: 0.5,
          }}>
            PRIVACY BUDGET
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
            {epsLevels.map((e, i) => {
              const active = i === idx
              return (
                <button
                  key={e.key}
                  onClick={() => setIdx(i)}
                  style={{
                    textAlign: 'left',
                    padding: '10px 14px',
                    background: active ? '#0D9488' : '#F9FAFB',
                    color: active ? '#FFFFFF' : '#374151',
                    border: `1px solid ${active ? '#0D9488' : '#E5E7EB'}`,
                    borderRadius: 8,
                    fontFamily: 'JetBrains Mono, monospace',
                    fontSize: 14, fontWeight: active ? 700 : 500,
                    cursor: 'pointer',
                    transition: 'all 0.15s',
                    display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                  }}>
                  <span>{e.label}</span>
                  <span style={{ fontSize: 11, opacity: 0.8 }}>σ = {e.sigma}</span>
                </button>
              )
            })}
          </div>

          <motion.div
            key={curr.key + '-note'}
            initial={{ opacity: 0, y: 6 }}
            animate={{ opacity: 1, y: 0 }}
            style={{
              padding: '12px 14px',
              background: '#F0FDFA',
              border: '1px solid #99F6E4',
              borderLeft: '4px solid #0D9488',
              borderRadius: 8,
            }}>
            <p style={{ fontFamily: 'Atkinson Hyperlegible', fontSize: 14, color: '#115E59', lineHeight: 1.5 }}>
              {curr.note}
            </p>
          </motion.div>

        </motion.div>
      </div>
    </Slide>
  )
}
