import { useState } from 'react'
import { motion } from 'framer-motion'
import { Slide } from '../deck/Slide'
import { AccuracyVsEpsilonChart } from '../components/AccuracyVsEpsilonChart'
import type { Epsilon } from '../data/results'
import { results, PALETTE, getAccForEpsilon } from '../data/results'

const EPS_OPTIONS: Epsilon[] = [0.5, 1, 2, 4, 8, Infinity]
const EPS_LABELS = ['0.5', '1', '2', '4', '8', '∞']
const DASH_PATTERNS = ['', '6,3', '3,3', '8,2,2,2', '4,2', '2,2']

export function Slide11() {
  const [epsIdx, setEpsIdx] = useState(0)
  const currentEps = EPS_OPTIONS[epsIdx]

  const avgDrop = results.reduce((sum, row) => {
    const dpAcc = getAccForEpsilon(row, currentEps).mean
    const infAcc = getAccForEpsilon(row, Infinity).mean
    return sum + (infAcc - dpAcc)
  }, 0) / results.length

  return (
    <Slide track="image" trackIndex={2} trackTotal={5}>
      <motion.h2
        initial={{ opacity: 0 }} animate={{ opacity: 1 }}
        className="font-serif-display"
        style={{ fontSize: 38, fontWeight: 600, color: '#171717', marginBottom: 6 }}>
        Privacy Budget Has Almost No Effect
      </motion.h2>
      <motion.p
        initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.1 }}
        style={{ fontSize: 22, color: '#6B7280', fontFamily: 'Atkinson Hyperlegible', marginBottom: 10 }}>
        Once good features are pretrained, <strong style={{ color: '#171717' }}>ε = 0.5 ≈ ε = 8</strong> in accuracy.
        Experiment identity matters ~100× more than ε.
      </motion.p>

      <div className="flex gap-5 flex-1 min-h-0">
        {/* Left: chart + legend + slider */}
        <div className="flex flex-col" style={{ flex: 1, minWidth: 0 }}>
          <AccuracyVsEpsilonChart highlightEps={currentEps} width={720} height={260} hideLegend />

          {/* Explicit legend */}
          <div className="flex flex-wrap gap-x-4 gap-y-1 mt-2 px-1"
            style={{ fontFamily: 'Atkinson Hyperlegible', fontSize: 20, color: '#374151' }}>
            {results.map((row, ri) => (
              <div key={row.experiment} className="flex items-center gap-1.5">
                <svg width={22} height={8}>
                  <line x1={0} x2={22} y1={4} y2={4}
                    stroke={PALETTE[ri % PALETTE.length]} strokeWidth={2.5}
                    strokeDasharray={DASH_PATTERNS[ri % DASH_PATTERNS.length]} />
                </svg>
                {row.shortLabel}
              </div>
            ))}
          </div>

          {/* Slider — below legend, clearly separated */}
          <motion.div
            initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.7 }}
            style={{
              marginTop: 14,
              padding: '10px 14px',
              background: '#FFFFFF',
              border: '1px solid #E5E5E5',
              borderRadius: 10,
              display: 'flex',
              alignItems: 'center',
              gap: 14,
            }}>
            <span style={{ fontSize: 20, color: '#171717', fontFamily: 'Atkinson Hyperlegible', fontWeight: 700, whiteSpace: 'nowrap' }}>
              Drag ε to highlight:
            </span>
            <input
              type="range" min={0} max={5} step={1} value={epsIdx}
              onChange={e => setEpsIdx(Number(e.target.value))}
              style={{ flex: 1, accentColor: '#6366F1' }}
            />
            <span style={{
              fontFamily: 'JetBrains Mono, monospace', fontSize: 22, fontWeight: 700,
              color: '#6366F1', minWidth: 50, textAlign: 'center',
              background: '#EEF2FF', padding: '2px 8px', borderRadius: 6,
            }}>
              ε={EPS_LABELS[epsIdx]}
            </span>
          </motion.div>
        </div>

        {/* Right: ANOVA card with explanations */}
        <motion.div
          initial={{ opacity: 0, x: 16 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.8 }}
          style={{
            width: 250, flexShrink: 0, alignSelf: 'flex-start',
            display: 'flex', flexDirection: 'column', gap: 10,
          }}>
          <div style={{
            padding: '14px 16px', background: '#EEF2FF',
            borderRadius: 10, border: '1px solid #C7D2FE',
            borderLeft: '4px solid #6366F1',
          }}>
            <p style={{ fontFamily: 'Atkinson Hyperlegible', fontSize: 20, color: '#4F46E5', marginBottom: 8, fontWeight: 700, letterSpacing: 0.3 }}>
              ANOVA ACROSS ε VALUES
            </p>
            <div className="flex gap-3 items-baseline" style={{ marginBottom: 6 }}>
              <span style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 24, color: '#4F46E5', fontWeight: 700 }}>F</span>
              <span style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 22, color: '#171717' }}>= 0.0005</span>
            </div>
            <p style={{ fontFamily: 'Atkinson Hyperlegible', fontSize: 20, color: '#6B7280', lineHeight: 1.4, marginBottom: 8 }}>
              between-group variance ÷ within-group variance. F ≈ 0 means ε barely moves the needle.
            </p>
            <div className="flex gap-3 items-baseline" style={{ marginBottom: 6 }}>
              <span style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 24, color: '#4F46E5', fontWeight: 700 }}>p</span>
              <span style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 22, color: '#171717' }}>= 1.0</span>
            </div>
            <p style={{ fontFamily: 'Atkinson Hyperlegible', fontSize: 20, color: '#6B7280', lineHeight: 1.4 }}>
              probability the differences are due to chance. p = 1 ⇒ no significant effect.
            </p>
          </div>

          <div style={{
            padding: '12px 16px', background: '#FEF2F2',
            borderRadius: 10, border: '1px solid #FECACA',
            borderLeft: '4px solid #EF4444',
          }}>
            <p style={{ fontFamily: 'Atkinson Hyperlegible', fontSize: 20, color: '#9F1239', marginBottom: 4, fontWeight: 700, letterSpacing: 0.3 }}>
              AVG DROP FROM ε=∞
            </p>
            <p style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 22, color: '#DC2626', fontWeight: 700 }}>
              {avgDrop >= 0 ? '−' : '+'}{Math.abs(avgDrop).toFixed(3)}
            </p>
            <p style={{ fontFamily: 'Atkinson Hyperlegible', fontSize: 20, color: '#7F1D1D', marginTop: 4, lineHeight: 1.4 }}>
              At ε={EPS_LABELS[epsIdx]} — practically zero.
            </p>
          </div>
        </motion.div>
      </div>
    </Slide>
  )
}
