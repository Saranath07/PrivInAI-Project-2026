import { useEffect, useState } from 'react'
import { motion } from 'framer-motion'
import { Slide } from '../deck/Slide'

// ── Public data cloud: 10 classes, coloured dots
const PUBLIC_DOTS = Array.from({ length: 60 }, (_, i) => {
  const cls = i % 10
  const angle = (cls / 10) * Math.PI * 2 + (i * 0.4)
  const r = 28 + (i % 5) * 7
  return {
    x: 80 + Math.cos(angle) * r,
    y: 80 + Math.sin(angle) * r,
    cls,
    delay: i * 0.018,
  }
})

// ── Private data cloud: tighter, greyscale-shifted (MNIST-like)
const PRIVATE_DOTS = Array.from({ length: 50 }, (_, i) => {
  const cls = i % 10
  const angle = (cls / 10) * Math.PI * 2 + (i * 0.35 + 0.5)
  const r = 22 + (i % 4) * 8
  return {
    x: 80 + Math.cos(angle) * r * 0.85,
    y: 80 + Math.sin(angle) * r * 0.85,
    cls,
    delay: i * 0.018 + 0.6,
  }
})

const CLASS_COLORS = [
  '#6366F1','#EC4899','#10B981','#F59E0B','#3B82F6',
  '#8B5CF6','#EF4444','#14B8A6','#F97316','#84CC16',
]

// ── Neural network nodes
const NN_LAYERS = [[3, 3], [4, 4], [3, 3]]
function buildNN() {
  const layers: { x: number; y: number; layer: number; idx: number }[] = []
  const layerXs = [20, 52, 84]
  NN_LAYERS.forEach(([n], li) => {
    for (let i = 0; i < n; i++) {
      layers.push({ x: layerXs[li], y: 18 + i * 22, layer: li, idx: i })
    }
  })
  return layers
}
const NN_NODES = buildNN()

// Which edges to draw
function nnEdges() {
  const edges: [number, number][] = []
  NN_NODES.forEach((a, ai) => {
    NN_NODES.forEach((b, bi) => {
      if (b.layer === a.layer + 1) edges.push([ai, bi])
    })
  })
  return edges
}
const NN_EDGES = nnEdges()

function PublicCloud() {
  return (
    <svg width={160} height={160} viewBox="0 0 160 160">
      {/* Gaussian ellipse hint */}
      <ellipse cx={80} cy={80} rx={55} ry={52} fill="#6366F108" stroke="#6366F120" strokeWidth={1} />
      {PUBLIC_DOTS.map((d, i) => (
        <motion.circle
          key={i}
          cx={d.x} cy={d.y} r={3.5}
          fill={CLASS_COLORS[d.cls]}
          opacity={0}
          initial={{ opacity: 0, scale: 0 }}
          animate={{ opacity: 0.85, scale: 1 }}
          transition={{ delay: d.delay, duration: 0.3, ease: 'backOut' }}
        />
      ))}
      <motion.text x={80} y={150} textAnchor="middle" fill="#6366F1"
        fontSize={14} fontWeight={700} fontFamily="Atkinson Hyperlegible"
        initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 1.2 }}>
        Public (SVHN)
      </motion.text>
    </svg>
  )
}

function PrivateCloud() {
  return (
    <svg width={160} height={160} viewBox="0 0 160 160">
      <ellipse cx={80} cy={80} rx={48} ry={46} fill="#EC489908" stroke="#EC489920" strokeWidth={1} />
      {PRIVATE_DOTS.map((d, i) => (
        <motion.circle
          key={i}
          cx={d.x} cy={d.y} r={3}
          fill={CLASS_COLORS[d.cls]}
          opacity={0}
          initial={{ opacity: 0, scale: 0 }}
          animate={{ opacity: 0.7, scale: 1 }}
          transition={{ delay: d.delay, duration: 0.25, ease: 'backOut' }}
        />
      ))}
      {/* lock overlay */}
      <motion.text x={80} y={80} textAnchor="middle" dominantBaseline="middle"
        fontSize={20}
        initial={{ opacity: 0, scale: 0.5 }} animate={{ opacity: 1, scale: 1 }}
        transition={{ delay: 1.4, duration: 0.4, ease: 'backOut' }}>
        🔒
      </motion.text>
      <motion.text x={80} y={150} textAnchor="middle" fill="#EC4899"
        fontSize={14} fontWeight={700} fontFamily="Atkinson Hyperlegible"
        initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 1.5 }}>
        Private (MNIST)
      </motion.text>
    </svg>
  )
}

function NeuralNet() {
  const [pulse, setPulse] = useState(false)
  useEffect(() => {
    const t = setTimeout(() => setPulse(true), 1200)
    return () => clearTimeout(t)
  }, [])
  return (
    <svg width={110} height={100} viewBox="0 0 110 100">
      {/* Edges */}
      {NN_EDGES.map(([ai, bi], i) => {
        const a = NN_NODES[ai], b = NN_NODES[bi]
        return (
          <motion.line key={i}
            x1={a.x} y1={a.y + 4} x2={b.x} y2={b.y + 4}
            stroke={pulse ? '#6366F1' : '#E5E5E5'}
            strokeWidth={0.8}
            opacity={0.4}
            initial={{ pathLength: 0 }} animate={{ pathLength: 1 }}
            transition={{ delay: 0.8 + i * 0.02, duration: 0.3 }}
            style={{ transition: 'stroke 400ms' }}
          />
        )
      })}
      {/* Nodes */}
      {NN_NODES.map((n, i) => (
        <motion.circle key={i}
          cx={n.x} cy={n.y + 4} r={5}
          fill={pulse ? '#6366F1' : '#C7D2FE'}
          stroke="#6366F140" strokeWidth={1}
          initial={{ scale: 0, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          transition={{ delay: 0.7 + i * 0.04, duration: 0.3, ease: 'backOut' }}
          style={{ transition: 'fill 400ms' }}
        />
      ))}
      <motion.text x={55} y={100} textAnchor="middle" fill="#6366F1"
        fontSize={9} fontFamily="Atkinson Hyperlegible"
        initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 1.3 }}>
        SmallCNN
      </motion.text>
    </svg>
  )
}

// Flowing particles from public → model → private
function FlowParticles({ from, to, color, count = 5, startDelay = 0, repeat = true }:
  { from: [number, number]; to: [number, number]; color: string; count?: number; startDelay?: number; repeat?: boolean }) {
  return (
    <>
      {Array.from({ length: count }, (_, i) => (
        <motion.circle
          key={i}
          r={3}
          fill={color}
          opacity={0.8}
          initial={{ cx: from[0], cy: from[1], opacity: 0, scale: 0 }}
          animate={repeat ? {
            cx: [from[0], to[0]],
            cy: [from[1], to[1]],
            opacity: [0, 1, 1, 0],
            scale: [0, 1, 1, 0],
          } : {
            cx: to[0], cy: to[1], opacity: [0, 1, 0], scale: [0, 1, 0],
          }}
          transition={{
            duration: 1.2,
            delay: startDelay + i * 0.22,
            repeat: Infinity,
            repeatDelay: 0.5,
            ease: 'easeInOut',
          }}
        />
      ))}
    </>
  )
}

// Gaussian PDF value
function gauss(x: number, mu: number, sigma: number) {
  return Math.exp(-0.5 * ((x - mu) / sigma) ** 2)
}

// Build an SVG path for a Gaussian curve
function gaussPath(mu: number, sigma: number, W: number, _H: number, baseline: number, amplitude: number) {
  const pts: string[] = []
  for (let px = 0; px <= W; px += 2) {
    const xVal = (px / W) * 6 - 3          // map px → [-3, 3] in std-dev space
    const y = baseline - gauss(xVal, (mu / W) * 6 - 3, sigma) * amplitude
    pts.push(`${px},${y.toFixed(1)}`)
  }
  return 'M ' + pts.join(' L ')
}

function GapAnnotation() {
  const [eps, setEps] = useState(5.0)

  const W = 420, H = 130, baseline = H - 10, amplitude = H - 22
  const sigma = 0.55
  // P centred at 35% of width; Q shifts right as ε increases
  const pMu = W * 0.35
  const shift = eps * 14                   // pixels of separation per ε unit
  const qMu = pMu + shift

  const pPath = gaussPath(pMu, sigma, W, H, baseline, amplitude)
  const qPath = gaussPath(qMu, sigma, W, H, baseline, amplitude)

  // Privacy label
  const label = eps <= 1 ? 'STRONG PRIVACY' : eps <= 4 ? 'MODERATE PRIVACY' : 'WEAK PRIVACY'
  const labelColor = eps <= 1 ? '#10B981' : eps <= 4 ? '#F59E0B' : '#EF4444'

  return (
    <motion.div
      initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 1.8, duration: 0.5 }}
      style={{
        width: '100%', background: '#FFFFFF',
        border: '1px solid #E5E5E5', borderRadius: 14,
        padding: '18px 28px', display: 'flex', alignItems: 'center', gap: 32,
      }}>

      {/* Left: animated distributions */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 10 }}>
        <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} style={{ overflow: 'visible' }}>
          <defs>
            <linearGradient id="overlapGrad" x1="0" x2="1" y1="0" y2="0">
              <stop offset="0%" stopColor="#6366F1" stopOpacity={0.25} />
              <stop offset="100%" stopColor="#EC4899" stopOpacity={0.25} />
            </linearGradient>
          </defs>
          {/* baseline */}
          <line x1={0} y1={baseline} x2={W} y2={baseline} stroke="#E5E5E5" strokeWidth={1} />
          {/* P fill */}
          <path d={pPath + ` L ${W},${baseline} L 0,${baseline} Z`} fill="#6366F118" />
          {/* Q fill */}
          <path d={qPath + ` L ${W},${baseline} L 0,${baseline} Z`} fill="#EC489918" />
          {/* P curve */}
          <path d={pPath} fill="none" stroke="#6366F1" strokeWidth={2.5} />
          {/* Q curve */}
          <path d={qPath} fill="none" stroke="#EC4899" strokeWidth={2.5} />
          {/* gap arrow */}
          {shift > 20 && (
            <>
              <line
                x1={pMu} x2={qMu} y1={baseline - amplitude * 0.55} y2={baseline - amplitude * 0.55}
                stroke="#4B5563" strokeWidth={1.5} strokeDasharray="5,3"
                markerStart="url(#arrowL)" markerEnd="url(#arrowR)"
              />
              <defs>
                <marker id="arrowL" markerWidth={7} markerHeight={7} refX={4} refY={3.5} orient="auto">
                  <polygon points="7,0 0,3.5 7,7" fill="#4B5563" />
                </marker>
                <marker id="arrowR" markerWidth={7} markerHeight={7} refX={3} refY={3.5} orient="auto">
                  <polygon points="0,0 7,3.5 0,7" fill="#4B5563" />
                </marker>
              </defs>
              <text x={(pMu + qMu) / 2} y={baseline - amplitude * 0.55 - 6}
                textAnchor="middle" fontSize={13} fill="#4B5563"
                fontFamily="Atkinson Hyperlegible" fontWeight={700}>
                distribution gap
              </text>
            </>
          )}
          {/* labels */}
          <text x={pMu} y={baseline + 16} textAnchor="middle" fill="#6366F1"
            fontSize={14} fontFamily="Atkinson Hyperlegible" fontWeight={700}>P (public)</text>
          <text x={qMu} y={baseline + 16} textAnchor="middle" fill="#EC4899"
            fontSize={14} fontFamily="Atkinson Hyperlegible" fontWeight={700}>Q (private)</text>
        </svg>

        {/* ε slider */}
        <div style={{ width: W, display: 'flex', flexDirection: 'column', gap: 6 }}>
          <div className="flex items-center justify-between">
            <span style={{ fontFamily: 'Atkinson Hyperlegible', fontSize: 16, fontWeight: 700, color: '#10B981' }}>
              Strong privacy (small ε)
            </span>
            <span style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 22, fontWeight: 700, color: labelColor }}>
              ε = {eps.toFixed(1)} · <span style={{ color: labelColor }}>{label}</span>
            </span>
            <span style={{ fontFamily: 'Atkinson Hyperlegible', fontSize: 16, fontWeight: 700, color: '#EF4444' }}>
              Weak privacy (large ε)
            </span>
          </div>
          <input
            type="range" min={0.5} max={8} step={0.1}
            value={eps}
            onChange={e => setEps(parseFloat(e.target.value))}
            style={{
              width: '100%', height: 8, borderRadius: 4,
              background: `linear-gradient(to right, #10B981, #F59E0B, #EF4444)`,
              accentColor: labelColor,
            }}
          />
        </div>
      </div>

      {/* Right: explanatory text */}
      <div style={{ width: 340, flexShrink: 0, display: 'flex', flexDirection: 'column', gap: 12, justifyContent: 'center' }}>
        <p style={{
          fontFamily: 'Crimson Pro, serif', fontSize: 22, color: '#171717',
          lineHeight: 1.45, fontStyle: 'italic',
        }}>
          The <strong style={{ color: '#4F46E5' }}>public</strong> distribution P and
          the <strong style={{ color: '#BE185D' }}>private</strong> distribution Q rarely match.
        </p>
        <p style={{
          fontFamily: 'Atkinson Hyperlegible', fontSize: 18, color: '#374151', lineHeight: 1.5,
        }}>
          Their overlap - the <em>distribution gap</em> - determines how much of public
          pretraining transfers. Small gap ⇒ features almost work; large gap ⇒ DP-SGD must compensate.
        </p>
        <p style={{
          fontFamily: 'Atkinson Hyperlegible', fontSize: 19, color: '#171717', fontWeight: 700, lineHeight: 1.4,
        }}>
          Does the gap size predict final DP accuracy?
        </p>
      </div>
    </motion.div>
  )
}

export function Slide01() {
  return (
    <Slide track="intro">
      <div className="flex flex-col h-full gap-3">
        <motion.h2
          initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }}
          className="font-serif-display"
          style={{ fontSize: 40, fontWeight: 600, color: '#171717', maxWidth: '95%', lineHeight: 1.2 }}>
          When does the pretrain–finetune gap predict DP accuracy?
        </motion.h2>

        {/* Main pipeline */}
        <div className="flex items-center justify-center flex-1 gap-0" style={{ minHeight: 0 }}>
          {/* Public data distribution */}
          <motion.div
            initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.1 }}>
            <PublicCloud />
          </motion.div>

          {/* Pretrain arrow with flowing particles */}
          <div className="flex flex-col items-center" style={{ width: 90, flexShrink: 0 }}>
            <svg width={90} height={40} style={{ overflow: 'visible' }}>
              <motion.line x1={5} y1={20} x2={80} y2={20} stroke="#6366F1" strokeWidth={1.5}
                initial={{ pathLength: 0 }} animate={{ pathLength: 1 }}
                transition={{ delay: 0.5, duration: 0.4 }} />
              <motion.polygon points="76,14 88,20 76,26" fill="#6366F1"
                initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.85 }} />
              <FlowParticles from={[8, 20]} to={[76, 20]} color="#6366F1" count={4} startDelay={1.0} />
            </svg>
            <motion.span
              initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.9 }}
              style={{ fontFamily: 'Atkinson Hyperlegible', fontSize: 16, fontWeight: 700, color: '#6366F1', marginTop: -6 }}>
              Pretrain
            </motion.span>
          </div>

          {/* Neural net */}
          <motion.div
            initial={{ opacity: 0, scale: 0.8 }} animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 0.6, type: 'spring', stiffness: 200 }}
            style={{
              padding: '10px 12px', background: '#EEF2FF', borderRadius: 12,
              border: '2px solid #6366F1', flexShrink: 0,
            }}>
            <NeuralNet />
          </motion.div>

          {/* DP-SGD arrow with noise particles */}
          <div className="flex flex-col items-center" style={{ width: 140, flexShrink: 0 }}>
            <svg width={140} height={44} style={{ overflow: 'visible' }}>
              <motion.line x1={5} y1={22} x2={128} y2={22} stroke="#EC4899" strokeWidth={2}
                strokeDasharray="6,4"
                initial={{ pathLength: 0 }} animate={{ pathLength: 1 }}
                transition={{ delay: 1.1, duration: 0.4 }} />
              <motion.polygon points="124,15 136,22 124,29" fill="#EC4899"
                initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 1.45 }} />
              {/* noise blobs */}
              {[18, 42, 68, 95].map((cx, i) => (
                <motion.circle key={i} cx={cx} cy={22 + (i % 2 === 0 ? -7 : 7)} r={2.5}
                  fill="#EC4899" opacity={0.4}
                  initial={{ opacity: 0, y: 0 }}
                  animate={{ opacity: [0, 0.6, 0], y: [0, -5, 0] }}
                  transition={{ delay: 1.5 + i * 0.15, duration: 0.6, repeat: Infinity, repeatDelay: 1.2 }} />
              ))}
            </svg>
            <motion.span
              initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 1.5 }}
              style={{ fontFamily: 'Atkinson Hyperlegible', fontSize: 14, color: '#EC4899', fontWeight: 700, marginTop: -6, whiteSpace: 'nowrap' }}>
              DP-SGD + 𝒩(0,σ²)
            </motion.span>
          </div>

          {/* Private data distribution */}
          <motion.div
            initial={{ opacity: 0, x: 20 }} animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.2 }}>
            <PrivateCloud />
          </motion.div>
        </div>

        {/* Gap annotation with overlapping distributions */}
        <GapAnnotation />

      </div>
    </Slide>
  )
}
