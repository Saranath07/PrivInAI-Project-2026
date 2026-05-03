import { motion } from 'framer-motion'
import { Slide } from '../deck/Slide'
import { Math as TeX } from '../components/Math'

// Clean gradient lifecycle: 3 stages stacked vertically
function GradientFlow() {
  const W = 380
  const stageY = [40, 180, 320]
  const cx = W / 2

  return (
    <svg width={W} height={540} viewBox={`0 0 ${W} 440`} style={{ overflow: 'visible' }}>
      {/* Stage 1: raw gradient (clean arrows radiating out, uneven length) */}
      <g transform={`translate(${cx}, ${stageY[0]})`}>
        <motion.text
          x={0} y={-58} textAnchor="middle" fontSize={15} fontFamily="Atkinson Hyperlegible"
          fill="#171717" fontWeight={700}
          initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
          raw gradient · g
        </motion.text>
        {[
          { dx: 48, dy: -22, col: '#EF4444' },
          { dx: -38, dy: -30, col: '#F59E0B' },
          { dx: 24, dy: 36, col: '#10B981' },
          { dx: -44, dy: 14, col: '#6366F1' },
        ].map((a, i) => (
          <motion.line
            key={i}
            x1={0} y1={0} x2={a.dx} y2={a.dy}
            stroke={a.col} strokeWidth={3} strokeLinecap="round"
            initial={{ pathLength: 0, opacity: 0 }}
            animate={{ pathLength: 1, opacity: 0.9 }}
            transition={{ delay: 0.1 + i * 0.08, duration: 0.4 }}
          />
        ))}
      </g>

      {/* Arrow 1→2 with label */}
      <motion.g
        initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.7 }}>
        <line x1={cx} y1={stageY[0] + 62} x2={cx} y2={stageY[1] - 52} stroke="#6366F1" strokeWidth={2} />
        <polygon points={`${cx - 5},${stageY[1] - 56} ${cx + 5},${stageY[1] - 56} ${cx},${stageY[1] - 46}`} fill="#6366F1" />
        <rect x={cx + 14} y={stageY[0] + 86} width={140} height={24} rx={4} fill="#EEF2FF" stroke="#C7D2FE" />
        <text x={cx + 84} y={stageY[0] + 103} textAnchor="middle" fontSize={13}
          fontFamily="JetBrains Mono, monospace" fill="#4F46E5" fontWeight={700}>
          ÷ max(1, ∥g∥/C)
        </text>
      </motion.g>

      {/* Stage 2: clipped ball */}
      <g transform={`translate(${cx}, ${stageY[1]})`}>
        <motion.text
          x={0} y={-58} textAnchor="middle" fontSize={15} fontFamily="Atkinson Hyperlegible"
          fill="#171717" fontWeight={700}
          initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.8 }}>
          clipped to C-ball
        </motion.text>
        <motion.circle
          cx={0} cy={0} r={42} fill="#EEF2FF" stroke="#6366F1" strokeWidth={2} strokeDasharray="5,3"
          initial={{ scale: 0 }} animate={{ scale: 1 }} transition={{ delay: 0.9, duration: 0.35 }}
        />
        <motion.text
          x={48} y={4} fontSize={14} fontFamily="JetBrains Mono, monospace" fill="#4F46E5" fontWeight={700}
          initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 1.1 }}>
          C
        </motion.text>
        {[
          { dx: 22, dy: -14, col: '#EF4444' },
          { dx: -18, dy: -18, col: '#F59E0B' },
          { dx: 12, dy: 22, col: '#10B981' },
          { dx: -24, dy: 8, col: '#6366F1' },
        ].map((a, i) => (
          <motion.line
            key={i}
            x1={0} y1={0} x2={a.dx} y2={a.dy}
            stroke={a.col} strokeWidth={2.5} strokeLinecap="round"
            initial={{ pathLength: 0, opacity: 0 }}
            animate={{ pathLength: 1, opacity: 0.8 }}
            transition={{ delay: 1.0 + i * 0.06, duration: 0.3 }}
          />
        ))}
      </g>

      {/* Arrow 2→3 */}
      <motion.g
        initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 1.4 }}>
        <line x1={cx} y1={stageY[1] + 54} x2={cx} y2={stageY[2] - 56} stroke="#EC4899" strokeWidth={2} strokeDasharray="5,3" />
        <polygon points={`${cx - 5},${stageY[2] - 60} ${cx + 5},${stageY[2] - 60} ${cx},${stageY[2] - 50}`} fill="#EC4899" />
        <rect x={cx + 14} y={stageY[1] + 78} width={140} height={24} rx={4} fill="#FDF2F8" stroke="#FBCFE8" />
        <text x={cx + 84} y={stageY[1] + 95} textAnchor="middle" fontSize={13}
          fontFamily="JetBrains Mono, monospace" fill="#DB2777" fontWeight={700}>
          + 𝒩(0, σ²C²)
        </text>
      </motion.g>

      {/* Stage 3: noisy output */}
      <g transform={`translate(${cx}, ${stageY[2]})`}>
        <motion.text
          x={0} y={-62} textAnchor="middle" fontSize={15} fontFamily="Atkinson Hyperlegible"
          fill="#BE185D" fontWeight={700}
          initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 1.5 }}>
          noisy gradient · g̃
        </motion.text>
        <motion.circle
          cx={0} cy={0} r={52} fill="#FDF2F8" stroke="#EC4899" strokeWidth={1.5} strokeDasharray="3,3"
          initial={{ scale: 0 }} animate={{ scale: 1 }} transition={{ delay: 1.6, duration: 0.35 }}
        />
        {Array.from({ length: 14 }).map((_, i) => {
          const a = (i / 14) * Math.PI * 2
          const r = 22 + (i % 3) * 10
          return (
            <motion.line
              key={i}
              x1={0} y1={0}
              x2={Math.cos(a) * r} y2={Math.sin(a) * r}
              stroke="#EC4899" strokeWidth={1.2} opacity={0.4}
              initial={{ pathLength: 0 }}
              animate={{ pathLength: 1 }}
              transition={{ delay: 1.75 + i * 0.03, duration: 0.25 }}
            />
          )
        })}
        <motion.text
          x={0} y={4} textAnchor="middle" fontSize={12} fontFamily="Atkinson Hyperlegible"
          fill="#9F1239" fontWeight={700}
          initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 2.3 }}>
          signal ≪ noise
        </motion.text>
      </g>
    </svg>
  )
}

export function Slide12() {
  return (
    <Slide track="image" trackIndex={3} trackTotal={5}>
      <motion.h2
        initial={{ opacity: 0 }} animate={{ opacity: 1 }}
        className="font-serif-display"
        style={{ fontSize: 30, fontWeight: 600, color: '#171717', marginBottom: 4 }}>
        Why is ε almost irrelevant? DP-SGD Forces a Linear Head
      </motion.h2>
      <motion.div
        initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.08 }}
        style={{
          background: '#FEF3C7', border: '1px solid #FCD34D',
          borderLeft: '4px solid #F59E0B',
          borderRadius: 8, padding: '8px 14px', marginBottom: 12,
        }}>
        <p style={{ fontFamily: 'Atkinson Hyperlegible', fontSize: 22, color: '#92400E', lineHeight: 1.5 }}>
          <strong>Why does this slide matter?</strong> The previous slide showed accuracy is flat across ε.
          That shouldn't happen — tighter privacy should hurt more. This slide explains <em>why</em> it doesn't:
          DP-SGD silently freezes the deep layers, so shrinking ε only perturbs a tiny, already-stable linear head.
        </p>
      </motion.div>

      <div className="flex gap-6 flex-1 min-h-0">
        {/* Left: math + bullets */}
        <div className="flex flex-col gap-5" style={{ flex: 1, minWidth: 0 }}>
          <motion.div
            initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.2 }}
            style={{
              background: '#FFFFFF', padding: '14px 18px',
              border: '1px solid #E5E5E5', borderRadius: 10,
            }}>
            <TeX latex="\tilde{g} = \frac{g}{\max\!\left(1,\frac{\|g\|_2}{C}\right)} + \mathcal{N}(0, \sigma^2 C^2 I)" block size={1.1} />
          </motion.div>

          <div className="flex flex-col gap-3">
            {[
              { bold: 'Clipping', rest: ' caps each sample\'s influence to a ball of radius C.' },
              { bold: 'Noise', rest: ' of scale σC is added — per-coordinate, regardless of depth.' },
              { bold: 'Signal-to-noise', rest: ' is worst in deep conv layers; best in the linear head.' },
            ].map((b, i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0, x: -10 }} animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.35 + i * 0.12 }}
                className="flex items-start gap-3">
                <div style={{
                  width: 10, height: 10, borderRadius: '50%', background: '#6366F1',
                  flexShrink: 0, marginTop: 7,
                }} />
                <p style={{ fontFamily: 'Atkinson Hyperlegible', fontSize: 22, color: '#171717', lineHeight: 1.5 }}>
                  <strong style={{ color: '#4F46E5' }}>{b.bold}</strong>{b.rest}
                </p>
              </motion.div>
            ))}
          </div>

          <motion.div
            initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.8 }}
            className="flex gap-3 items-center mt-1">
            <div style={{
              padding: '10px 18px', background: '#FEF2F2', borderRadius: 10,
              border: '1px solid #FECACA', textAlign: 'center', flex: 1,
            }}>
              <div style={{ fontFamily: 'Atkinson Hyperlegible', fontSize: 20, color: '#6B7280', fontWeight: 700 }}>Full model</div>
              <div style={{ fontFamily: 'JetBrains Mono', fontSize: 22, fontWeight: 700, color: '#DC2626' }}>~500K</div>
              <div style={{ fontFamily: 'Atkinson Hyperlegible', fontSize: 20, color: '#9CA3AF' }}>trainable params</div>
            </div>
            <div style={{ fontSize: 22, color: '#6B7280', fontWeight: 700 }}>→</div>
            <div style={{
              padding: '10px 18px', background: '#F0FDF4', borderRadius: 10,
              border: '1px solid #A7F3D0', textAlign: 'center', flex: 1,
            }}>
              <div style={{ fontFamily: 'Atkinson Hyperlegible', fontSize: 20, color: '#6B7280', fontWeight: 700 }}>Under DP-SGD</div>
              <div style={{ fontFamily: 'JetBrains Mono', fontSize: 22, fontWeight: 700, color: '#059669' }}>~5K</div>
              <div style={{ fontFamily: 'Atkinson Hyperlegible', fontSize: 20, color: '#9CA3AF' }}>effective (linear head)</div>
            </div>
          </motion.div>
        </div>

        {/* Right: gradient flow visualization */}
        <motion.div
          initial={{ opacity: 0, x: 16 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.3 }}
          style={{
            width: 420, flexShrink: 0,
            background: '#FFFFFF',
            border: '1px solid #E5E5E5', borderRadius: 12,
            padding: '16px 12px',
            display: 'flex', flexDirection: 'column', alignItems: 'center',
          }}>
          <p style={{ fontFamily: 'Atkinson Hyperlegible', fontSize: 20, color: '#6B7280', fontWeight: 700, marginBottom: 8 }}>
            Gradient lifecycle under DP-SGD
          </p>
          <GradientFlow />
        </motion.div>
      </div>
    </Slide>
  )
}
