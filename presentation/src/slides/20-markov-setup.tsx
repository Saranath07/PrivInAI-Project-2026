import { useEffect, useState } from 'react'
import { motion } from 'framer-motion'
import { Slide } from '../deck/Slide'
import { Math as TeX } from '../components/Math'

// Five-state Markov chain laid out on a circle with a random-walking token.
// λ-slider interpolates toward the uniform distribution, visualised as
// edges fading to a neutral grey & the walker losing its "pull" to structure.
const STATES = ['the', 'cat', 'sat', 'on', 'mat']
const CX = 280
const CY = 230
const R = 160

const positions = STATES.map((_, i) => {
  const theta = (-Math.PI / 2) + (i * 2 * Math.PI) / STATES.length
  return { x: CX + R * Math.cos(theta), y: CY + R * Math.sin(theta) }
})

// Baseline structured transition matrix P (rows sum to 1)
const P: number[][] = [
  [0.05, 0.55, 0.10, 0.20, 0.10],
  [0.15, 0.05, 0.55, 0.15, 0.10],
  [0.10, 0.10, 0.05, 0.60, 0.15],
  [0.20, 0.10, 0.10, 0.05, 0.55],
  [0.50, 0.15, 0.15, 0.15, 0.05],
]

const UNIFORM = 1 / STATES.length

function sampleNext(from: number, lambda: number): number {
  const row = P[from].map(p => (1 - lambda) * p + lambda * UNIFORM)
  const r = Math.random()
  let acc = 0
  for (let j = 0; j < row.length; j++) {
    acc += row[j]
    if (r < acc) return j
  }
  return row.length - 1
}

function MarkovChain({ lambda }: { lambda: number }) {
  const [cur, setCur] = useState(0)
  const [prev, setPrev] = useState(0)

  useEffect(() => {
    const id = setInterval(() => {
      setPrev(cur)
      setCur(c => sampleNext(c, lambda))
    }, 900)
    return () => clearInterval(id)
  }, [cur, lambda])

  return (
    <svg width={560} height={460} viewBox="0 0 560 460" style={{ overflow: 'visible' }}>
      {/* edges: one per ordered pair (skip self-loops to reduce clutter) */}
      {positions.map((a, i) =>
        positions.map((b, j) => {
          if (i === j) return null
          const p = (1 - lambda) * P[i][j] + lambda * UNIFORM
          const mx = (a.x + b.x) / 2
          const my = (a.y + b.y) / 2
          // offset the curve so the two directions don't overlap
          const dx = b.x - a.x
          const dy = b.y - a.y
          const len = Math.sqrt(dx * dx + dy * dy)
          const ox = (-dy / len) * 14
          const oy = (dx / len) * 14
          const cx = mx + ox
          const cy = my + oy
          const opacity = 0.15 + p * 1.3
          const active = prev === i && cur === j
          return (
            <path
              key={`${i}-${j}`}
              d={`M ${a.x} ${a.y} Q ${cx} ${cy} ${b.x} ${b.y}`}
              stroke={active ? '#EC4899' : '#6366F1'}
              strokeWidth={active ? 3 : 1 + p * 4}
              opacity={active ? 0.9 : Math.min(0.7, opacity)}
              fill="none"
            />
          )
        })
      )}

      {/* nodes */}
      {positions.map((pos, i) => {
        const isCur = i === cur
        return (
          <g key={i}>
            <motion.circle
              cx={pos.x} cy={pos.y}
              r={42}
              fill={isCur ? '#EC4899' : '#FFFFFF'}
              stroke={isCur ? '#BE185D' : '#6366F1'}
              strokeWidth={isCur ? 3 : 2}
              animate={{ scale: isCur ? 1.08 : 1 }}
              transition={{ duration: 0.25 }}
            />
            <text
              x={pos.x} y={pos.y + 6}
              textAnchor="middle"
              fontSize={20}
              fontFamily="Atkinson Hyperlegible"
              fontWeight={700}
              fill={isCur ? '#FFFFFF' : '#171717'}
            >
              {STATES[i]}
            </text>
          </g>
        )
      })}
    </svg>
  )
}

export function Slide20() {
  const [lambda, setLambda] = useState(0)
  return (
    <Slide track="markov" trackIndex={1} trackTotal={5}>
      <motion.h2
        initial={{ opacity: 0 }} animate={{ opacity: 1 }}
        className="font-serif-display"
        style={{ fontSize: 38, fontWeight: 600, color: '#171717', marginBottom: 6 }}>
        Markov Language Model: Controlled Distribution Shift
      </motion.h2>
      <motion.p
        initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.1 }}
        style={{ fontFamily: 'Atkinson Hyperlegible', fontSize: 22, color: '#6B7280', marginBottom: 14 }}>
        Order-k chains on WikiText-2 vocabulary. Slide λ to mix the pretrain distribution with uniform noise.
      </motion.p>

      {/* Big equation banner above everything */}
      <motion.div
        initial={{ opacity: 0, y: -6 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.15 }}
        style={{
          background: '#FFFFFF', padding: '18px 28px',
          border: '1px solid #E5E5E5', borderRadius: 12,
          marginBottom: 18,
          display: 'flex', alignItems: 'center', gap: 28,
        }}>
        <span style={{
          fontFamily: 'Atkinson Hyperlegible', fontSize: 15,
          color: '#6B7280', fontWeight: 700, letterSpacing: 0.3,
          flexShrink: 0,
        }}>
          PERTURBED<br />TRANSITION
        </span>
        <div style={{ flex: 1, overflow: 'hidden' }}>
          <TeX
            latex="[T]_Q(w' \mid c) \;=\; (1-\lambda)\,[T]_P(w' \mid c) \;+\; \lambda \cdot U(V)"
            block size={1.9}
          />
        </div>
      </motion.div>

      <div className="flex gap-8 flex-1 min-h-0">
        {/* Left: animated chain */}
        <motion.div
          initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.2 }}
          style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center' }}>
          <MarkovChain lambda={lambda} />
        </motion.div>

        {/* Right: distribution shift explanation + slider + setup */}
        <motion.div
          initial={{ opacity: 0, x: 16 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.3 }}
          style={{ width: 440, flexShrink: 0, display: 'flex', flexDirection: 'column', gap: 16, justifyContent: 'center' }}>

          <div style={{
            background: '#EEF2FF', padding: '18px 22px',
            border: '1px solid #C7D2FE', borderLeft: '4px solid #6366F1',
            borderRadius: 10,
          }}>
            <p style={{ fontFamily: 'Atkinson Hyperlegible', fontSize: 19, color: '#171717', lineHeight: 1.5, marginBottom: 10 }}>
              <strong style={{ color: '#4F46E5' }}>Distribution shift</strong> = the gap between what the model
              was pretrained on (<em>P</em>) and what it is fine-tuned on (<em>Q</em>).
            </p>
            <p style={{ fontFamily: 'Atkinson Hyperlegible', fontSize: 18, color: '#4B5563', lineHeight: 1.5 }}>
              λ = 0 → fine-tune on pretrain distribution (no shift).<br />
              λ = 1 → fine-tune on pure noise (maximum shift).
            </p>
          </div>

          <div style={{
            background: '#FFFFFF', padding: '18px 22px',
            border: '1px solid #E5E5E5', borderRadius: 10,
          }}>
            <div className="flex items-baseline justify-between" style={{ marginBottom: 10 }}>
              <span style={{ fontFamily: 'Atkinson Hyperlegible', fontSize: 18, color: '#4F46E5', fontWeight: 700, letterSpacing: 0.3 }}>
                λ = DISTRIBUTION SHIFT
              </span>
              <span style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 28, color: '#4F46E5', fontWeight: 700 }}>
                {lambda.toFixed(2)}
              </span>
            </div>
            <input
              type="range"
              min={0} max={1} step={0.01}
              value={lambda}
              onChange={e => setLambda(parseFloat(e.target.value))}
              style={{ width: '100%', accentColor: '#6366F1' }}
            />
            <div className="flex justify-between" style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 14, color: '#6B7280', marginTop: 4 }}>
              <span>0 · structured</span>
              <span>1 · uniform</span>
            </div>
          </div>

          <div style={{
            background: '#FFFBEB', padding: '14px 20px',
            border: '1px solid #FDE68A', borderLeft: '4px solid #F59E0B',
            borderRadius: 8,
          }}>
            <p style={{ fontFamily: 'Atkinson Hyperlegible', fontSize: 17, color: '#78350F', lineHeight: 1.5 }}>
              <strong>Setup:</strong> 33K vocabulary · 2M tokens · orders k ∈ {'{3, 4, 5}'} · 5,000 × 50-token samples.
            </p>
          </div>
        </motion.div>
      </div>
    </Slide>
  )
}
