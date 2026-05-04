import { useState } from 'react'
import { motion } from 'framer-motion'
import { Slide } from '../deck/Slide'

interface Row {
  scale: string
  mauve: number
  ttr: number
  g2: number
  g3: number
  g4: number
  g5: number
  rep: number
  top: string
  zone: 'clean' | 'transition' | 'collapse'
}

const DATA: Row[] = [
  { scale: '0p00', mauve: 1.000, ttr: 15.4, g2: 58.5, g3: 31.3, g4: 20.4, g5: 16.5, rep: 31.3, top: '"- - -" 315×',         zone: 'clean' },
  { scale: '0p05', mauve: 0.980, ttr: 15.5, g2: 57.8, g3: 30.0, g4: 19.0, g5: 15.0, rep: 32.7, top: '"a lot of" 65×',       zone: 'clean' },
  { scale: '0p10', mauve: 0.970, ttr: 15.4, g2: 58.6, g3: 31.5, g4: 20.3, g5: 16.2, rep: 37.0, top: '"* * *" 435×',         zone: 'clean' },
  { scale: '0p20', mauve: 0.928, ttr: 13.1, g2: 65.4, g3: 40.9, g4: 29.5, g5: 25.0, rep: 49.0, top: '"very very very" 256×', zone: 'clean' },
  { scale: '0p25', mauve: 0.811, ttr: 12.1, g2: 71.2, g3: 50.4, g4: 39.5, g5: 34.0, rep: 69.7, top: '"a lot of" 127×',      zone: 'transition' },
  { scale: '0p26', mauve: 0.745, ttr: 11.1, g2: 72.9, g3: 52.8, g4: 41.8, g5: 36.3, rep: 68.0, top: '"very, very, very," 233×', zone: 'transition' },
  { scale: '0p27', mauve: 0.677, ttr: 11.0, g2: 73.9, g3: 54.9, g4: 44.5, g5: 39.0, rep: 72.3, top: '"of the same" 141×',   zone: 'transition' },
  { scale: '0p28', mauve: 0.558, ttr: 10.5, g2: 75.7, g3: 58.1, g4: 48.1, g5: 42.9, rep: 77.3, top: '"This is the" 157×',   zone: 'transition' },
  { scale: '0p30', mauve: 0.344, ttr:  9.4, g2: 79.1, g3: 64.1, g4: 54.9, g5: 49.3, rep: 82.7, top: '"a lot of" 179×',      zone: 'transition' },
  { scale: '0p35', mauve: 0.067, ttr:  7.5, g2: 83.6, g3: 71.9, g4: 64.1, g5: 58.9, rep: 89.3, top: '"The The The" 511×',   zone: 'collapse' },
  { scale: '0p40', mauve: 0.031, ttr:  9.8, g2: 76.3, g3: 57.7, g4: 44.6, g5: 37.3, rep: 74.0, top: '"the the the" 3891×',  zone: 'collapse' },
  { scale: '0p60', mauve: 0.006, ttr: 18.5, g2: 53.4, g3: 24.6, g4: 10.7, g5:  5.5, rep: 43.7, top: '"the the the" 2702×',  zone: 'collapse' },
]

const ZONE_STYLES = {
  clean:      { bg: '#EFF6FF', border: '#93C5FD', text: '#1E3A8A' },
  transition: { bg: '#FEF3C7', border: '#FCD34D', text: '#92400E' },
  collapse:   { bg: '#FEE2E2', border: '#FCA5A5', text: '#991B1B' },
}

type Col = 'mauve' | 'ttr' | 'g2' | 'g3' | 'g4' | 'g5' | 'rep'
const COLS: { key: Col; label: string; unit: string; help: string }[] = [
  { key: 'mauve', label: 'MAUVE', unit: '',  help: 'Distribution similarity to clean GPT-2 XL output (1.0 = identical).' },
  { key: 'ttr',   label: 'TTR',   unit: '%', help: 'Type-token ratio - lexical diversity. Drops as text becomes repetitive.' },
  { key: 'g2',    label: '2-gram', unit: '%', help: 'Share of repeated bigrams. Rises with degenerate text.' },
  { key: 'g3',    label: '3-gram', unit: '%', help: 'Repeated trigrams.' },
  { key: 'g4',    label: '4-gram', unit: '%', help: 'Repeated 4-grams.' },
  { key: 'g5',    label: '5-gram', unit: '%', help: 'Repeated 5-grams.' },
  { key: 'rep',   label: 'Repeated docs', unit: '%', help: 'Share of generated documents that are near-duplicates.' },
]

export function Slide33() {
  const [col, setCol] = useState<Col>('mauve')
  const [hover, setHover] = useState<string | null>(null)
  const current = COLS.find(c => c.key === col)!

  const val = (r: Row) => (r[col] as number)
  const fmt = (r: Row) => col === 'mauve' ? val(r).toFixed(3) : val(r).toFixed(1) + '%'

  const values = DATA.map(val)
  const min = Math.min(...values)
  const max = Math.max(...values)
  const bar = (v: number) => ((v - min) / (max - min || 1)) * 100

  return (
    <Slide track="transformers" trackIndex={4} trackTotal={7}>
      <motion.h2
        initial={{ opacity: 0 }} animate={{ opacity: 1 }}
        className="font-serif-display"
        style={{ fontSize: 32, fontWeight: 600, color: '#171717', marginBottom: 4 }}>
        Analytics of the Generated Data
      </motion.h2>
      <motion.p
        initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.1 }}
        style={{ fontFamily: 'Atkinson Hyperlegible', fontSize: 16, color: '#6B7280', marginBottom: 12 }}>
        Perturbed GPT-2 XL collapses into degenerate repetition long before MAUVE bottoms out. Click a column to re-rank.
      </motion.p>

      <div style={{ display: 'flex', gap: 6, marginBottom: 10, flexWrap: 'wrap' }}>
        {COLS.map(c => {
          const active = c.key === col
          return (
            <button
              key={c.key}
              onClick={() => setCol(c.key)}
              style={{
                padding: '6px 12px',
                background: active ? '#0D9488' : '#F3F4F6',
                color: active ? '#FFFFFF' : '#374151',
                border: `1px solid ${active ? '#0D9488' : '#E5E7EB'}`,
                borderRadius: 6,
                fontFamily: 'Atkinson Hyperlegible',
                fontSize: 12, fontWeight: 700, letterSpacing: 0.3,
                cursor: 'pointer',
                transition: 'all 0.15s',
              }}>
              {c.label.toUpperCase()}
            </button>
          )
        })}
      </div>

      <motion.div
        key={col}
        initial={{ opacity: 0, y: 4 }} animate={{ opacity: 1, y: 0 }}
        style={{
          padding: '8px 12px', marginBottom: 10,
          background: '#F0FDFA', border: '1px solid #99F6E4',
          borderLeft: '4px solid #0D9488', borderRadius: 6,
          fontFamily: 'Atkinson Hyperlegible', fontSize: 13, color: '#115E59',
        }}>
        <strong>{current.label}{current.unit}.</strong> {current.help}
      </motion.div>

      <div className="flex gap-4 flex-1 min-h-0">
        <div style={{
          flex: 1, display: 'flex', flexDirection: 'column', gap: 4,
          overflow: 'auto', minHeight: 0,
        }}>
          <div style={{
            display: 'grid',
            gridTemplateColumns: '60px 90px 1fr 200px',
            alignItems: 'center', gap: 10,
            padding: '6px 12px',
            background: '#F9FAFB', border: '1px solid #E5E7EB', borderRadius: 6,
            fontFamily: 'Atkinson Hyperlegible', fontSize: 11,
            color: '#6B7280', fontWeight: 700, letterSpacing: 0.4,
          }}>
            <span>SCALE</span>
            <span>{current.label.toUpperCase()}</span>
            <span>DISTRIBUTION</span>
            <span>TOP TRIGRAM</span>
          </div>
          {DATA.map((r, i) => {
            const z = ZONE_STYLES[r.zone]
            const isHover = hover === r.scale
            return (
              <motion.div
                key={r.scale}
                initial={{ opacity: 0, x: -6 }} animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.15 + i * 0.025 }}
                onMouseEnter={() => setHover(r.scale)}
                onMouseLeave={() => setHover(null)}
                style={{
                  display: 'grid',
                  gridTemplateColumns: '60px 90px 1fr 200px',
                  alignItems: 'center', gap: 10,
                  padding: '6px 12px',
                  background: isHover ? z.border : z.bg,
                  border: `1px solid ${z.border}`,
                  borderLeft: `4px solid ${z.text}`,
                  borderRadius: 6,
                  transition: 'background 0.15s',
                  cursor: 'default',
                }}>
                <span style={{
                  fontFamily: 'JetBrains Mono, monospace', fontSize: 13,
                  fontWeight: 700, color: z.text,
                }}>{r.scale}</span>
                <span style={{
                  fontFamily: 'JetBrains Mono, monospace', fontSize: 13,
                  color: z.text,
                }}>{fmt(r)}</span>
                <div style={{
                  height: 10, background: '#FFFFFF66', borderRadius: 3, overflow: 'hidden',
                  border: `1px solid ${z.border}`,
                }}>
                  <motion.div
                    layout
                    animate={{ width: `${bar(val(r))}%` }}
                    transition={{ duration: 0.35, ease: [0, 0, 0.2, 1] }}
                    style={{
                      height: '100%', background: z.text, borderRadius: 2,
                    }}
                  />
                </div>
                <span style={{
                  fontFamily: 'Atkinson Hyperlegible', fontSize: 12,
                  color: z.text, fontStyle: 'italic',
                }}>{r.top}</span>
              </motion.div>
            )
          })}
        </div>

        <div style={{ width: 220, flexShrink: 0, display: 'flex', flexDirection: 'column', gap: 8 }}>
          <div style={{ fontFamily: 'Atkinson Hyperlegible', fontSize: 11, color: '#6B7280', fontWeight: 700, letterSpacing: 0.4 }}>
            LEGEND
          </div>
          {([
            { z: 'clean',      label: 'Clean',      desc: 's ≤ 0.20 · MAUVE ≥ 0.93' },
            { z: 'transition', label: 'Transition', desc: '0.25 ≤ s ≤ 0.30' },
            { z: 'collapse',   label: 'Collapse',   desc: 's ≥ 0.35 · degenerate text' },
          ] as const).map(item => {
            const s = ZONE_STYLES[item.z]
            return (
              <div key={item.z} style={{
                padding: '8px 10px',
                background: s.bg, border: `1px solid ${s.border}`,
                borderLeft: `4px solid ${s.text}`,
                borderRadius: 6,
              }}>
                <p style={{ fontFamily: 'Atkinson Hyperlegible', fontSize: 12, fontWeight: 700, color: s.text, marginBottom: 2 }}>
                  {item.label}
                </p>
                <p style={{ fontFamily: 'Atkinson Hyperlegible', fontSize: 11, color: s.text, opacity: 0.85 }}>
                  {item.desc}
                </p>
              </div>
            )
          })}
          <div style={{
            padding: '8px 10px', background: '#F9FAFB',
            border: '1px solid #E5E7EB', borderRadius: 6,
          }}>
            <p style={{ fontFamily: 'Atkinson Hyperlegible', fontSize: 11, color: '#6B7280', lineHeight: 1.5 }}>
              <strong>Key signal.</strong> 5-gram repetition at s = 0.35 reaches 58.9% - classic GPT-2 degenerate loop, matching the MAUVE cliff.
            </p>
          </div>
        </div>
      </div>
    </Slide>
  )
}
