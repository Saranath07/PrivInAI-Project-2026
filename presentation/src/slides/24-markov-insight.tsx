import { motion } from 'framer-motion'
import { Slide } from '../deck/Slide'
import { Math as TeX } from '../components/Math'

export function Slide24() {
  return (
    <Slide track="markov" trackIndex={5} trackTotal={5}>
      <motion.h2
        initial={{ opacity: 0 }} animate={{ opacity: 1 }}
        className="font-serif-display"
        style={{ fontSize: 38, fontWeight: 600, color: '#171717', marginBottom: 4 }}>
        A Rényi Threshold Governs DP Utility
      </motion.h2>
      <motion.p
        initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.1 }}
        style={{ fontFamily: 'Atkinson Hyperlegible', fontSize: 22, color: '#6B7280', marginBottom: 20 }}>
        Across both chain orders, DP-SGD collapses once D<sub>α</sub>(P‖Q) exceeds a critical value — matching the Setlur et al. (2025) lower bound for public-private learning.
      </motion.p>

      <div className="flex gap-6 flex-1 min-h-0">
        {/* Left: schematic of the threshold */}
        <motion.div
          initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.2 }}
          style={{
            flex: 1, background: '#FFFFFF', border: '1px solid #E5E5E5', borderRadius: 12,
            padding: '20px 28px', display: 'flex', flexDirection: 'column',
          }}>
          <p style={{ fontFamily: 'Atkinson Hyperlegible', fontSize: 15, color: '#6B7280', fontWeight: 700, letterSpacing: 0.3, marginBottom: 12 }}>
            PERPLEXITY vs DISTRIBUTION SHIFT (schematic)
          </p>
          <svg viewBox="0 0 520 320" style={{ width: '100%', flex: 1 }}>
            {/* axes */}
            <line x1={60} y1={270} x2={500} y2={270} stroke="#9CA3AF" strokeWidth={1.5} />
            <line x1={60} y1={30} x2={60} y2={270} stroke="#9CA3AF" strokeWidth={1.5} />
            <text x={280} y={305} textAnchor="middle" fontFamily="Atkinson Hyperlegible" fontSize={14} fill="#6B7280">
              Rényi divergence D<tspan baselineShift="sub" fontSize={10}>α</tspan>(P ‖ Q) →
            </text>
            <text x={16} y={150} textAnchor="middle" fontFamily="Atkinson Hyperlegible" fontSize={14} fill="#6B7280"
              transform="rotate(-90, 16, 150)">
              Perplexity →
            </text>

            {/* threshold line */}
            <motion.line
              x1={300} x2={300} y1={30} y2={270}
              stroke="#EF4444" strokeWidth={2} strokeDasharray="6,4"
              initial={{ pathLength: 0 }} animate={{ pathLength: 1 }} transition={{ delay: 0.5, duration: 0.6 }}
            />
            <text x={300} y={22} textAnchor="middle" fontFamily="JetBrains Mono, monospace"
              fontSize={13} fill="#EF4444" fontWeight={700}>
              D* · critical threshold
            </text>

            {/* SGD curve — smooth rise */}
            <motion.path
              d="M 60 230 C 180 218, 260 205, 300 195 S 420 160, 500 130"
              fill="none" stroke="#2563EB" strokeWidth={3}
              initial={{ pathLength: 0 }} animate={{ pathLength: 1 }} transition={{ delay: 0.3, duration: 0.9 }}
            />

            {/* DP-SGD curve — flat then cliff */}
            <motion.path
              d="M 60 240 C 160 238, 240 236, 300 232 C 315 232, 325 220, 340 180 C 360 120, 420 80, 500 70"
              fill="none" stroke="#EA580C" strokeWidth={3}
              initial={{ pathLength: 0 }} animate={{ pathLength: 1 }} transition={{ delay: 0.6, duration: 0.9 }}
            />

            {/* legend */}
            <g transform="translate(380, 240)">
              <rect x={0} y={0} width={118} height={54} rx={6} fill="#FAFAF7" stroke="#E5E5E5" />
              <line x1={8} x2={26} y1={18} y2={18} stroke="#2563EB" strokeWidth={3} />
              <text x={32} y={22} fontFamily="Atkinson Hyperlegible" fontSize={12} fill="#171717">vanilla SGD</text>
              <line x1={8} x2={26} y1={38} y2={38} stroke="#EA580C" strokeWidth={3} />
              <text x={32} y={42} fontFamily="Atkinson Hyperlegible" fontSize={12} fill="#171717">DP-SGD</text>
            </g>
          </svg>
        </motion.div>

        {/* Right: takeaways */}
        <motion.div
          initial={{ opacity: 0, x: 16 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.3 }}
          style={{ width: 420, flexShrink: 0, display: 'flex', flexDirection: 'column', gap: 14, justifyContent: 'center' }}>

          <div style={{
            background: '#FFFFFF', padding: '16px 20px',
            border: '1px solid #E5E5E5', borderRadius: 10,
          }}>
            <p style={{ fontFamily: 'Atkinson Hyperlegible', fontSize: 15, color: '#6B7280', fontWeight: 700, letterSpacing: 0.3, marginBottom: 8 }}>
              SETLUR ET AL. (2025) BOUND
            </p>
            <TeX
              latex="\text{Utility}_{\mathrm{DP}}(Q) \;\leq\; f\!\left(D_\alpha(P\|Q), \varepsilon\right)"
              block size={1.0}
            />
            <p style={{ fontFamily: 'Atkinson Hyperlegible', fontSize: 16, color: '#4B5563', marginTop: 8, lineHeight: 1.5 }}>
              Public pretraining only helps private fine-tuning while P stays close to Q.
            </p>
          </div>

          <div style={{
            background: '#EEF2FF', padding: '14px 18px',
            border: '1px solid #C7D2FE', borderLeft: '4px solid #6366F1',
            borderRadius: 10,
          }}>
            <p style={{ fontFamily: 'Atkinson Hyperlegible', fontSize: 17, color: '#171717', lineHeight: 1.5, marginBottom: 8 }}>
              <strong style={{ color: '#4F46E5' }}>Below D*.</strong> SGD and DP-SGD curves track closely — DP cost is small.
            </p>
            <p style={{ fontFamily: 'Atkinson Hyperlegible', fontSize: 17, color: '#171717', lineHeight: 1.5 }}>
              <strong style={{ color: '#4F46E5' }}>Above D*.</strong> DP-SGD sharply diverges — noise overwhelms signal once P and Q disagree.
            </p>
          </div>

          <div style={{
            background: '#F0FDF4', padding: '12px 18px',
            border: '1px solid #A7F3D0', borderLeft: '4px solid #10B981',
            borderRadius: 8,
          }}>
            <p style={{ fontFamily: 'Atkinson Hyperlegible', fontSize: 16, color: '#065F46', lineHeight: 1.5 }}>
              ⇒ Practical rule: pick pretraining data so D<sub>α</sub>(P‖Q) ≪ D* before committing to DP fine-tuning.
            </p>
          </div>
        </motion.div>
      </div>
    </Slide>
  )
}
