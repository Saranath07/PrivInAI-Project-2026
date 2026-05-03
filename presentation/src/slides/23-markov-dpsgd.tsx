import { motion } from 'framer-motion'
import { Slide } from '../deck/Slide'

export function Slide23() {
  const base = import.meta.env.BASE_URL
  return (
    <Slide track="markov" trackIndex={4} trackTotal={5}>
      <motion.h2
        initial={{ opacity: 0 }} animate={{ opacity: 1 }}
        className="font-serif-display"
        style={{ fontSize: 36, fontWeight: 600, color: '#171717', marginBottom: 4 }}>
        Order-4 Chains: Longer Context Softens — but Does Not Remove — the Cliff
      </motion.h2>
      <motion.p
        initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.1 }}
        style={{ fontFamily: 'Atkinson Hyperlegible', fontSize: 20, color: '#6B7280', marginBottom: 14 }}>
        Same axes as order-3, with k = 4. Richer context means the pretrain distribution is sharper, so λ must go further before DP-SGD collapses.
      </motion.p>

      <div className="flex gap-6 flex-1 min-h-0">
        {/* Left: figure */}
        <motion.div
          initial={{ opacity: 0, scale: 0.98 }} animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.2, duration: 0.4 }}
          style={{
            flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center',
            background: '#FFFFFF', border: '1px solid #E5E5E5', borderRadius: 10, padding: 10,
            minWidth: 0, minHeight: 0,
          }}>
          <img
            src={`${base}markov/results-order-4.png`}
            alt="Order-4 Markov results: perplexity and MAUVE vs λ"
            style={{ maxWidth: '100%', maxHeight: '100%', objectFit: 'contain' }}
          />
        </motion.div>

        {/* Right: takeaways + comparison */}
        <motion.div
          initial={{ opacity: 0, x: 16 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.3 }}
          style={{ width: 360, flexShrink: 0, display: 'flex', flexDirection: 'column', gap: 14, justifyContent: 'center' }}>

          <div style={{
            background: '#EEF2FF', padding: '14px 18px',
            border: '1px solid #C7D2FE', borderLeft: '4px solid #6366F1',
            borderRadius: 10,
          }}>
            <p style={{ fontFamily: 'Atkinson Hyperlegible', fontSize: 15, color: '#4F46E5', fontWeight: 700, letterSpacing: 0.3, marginBottom: 8 }}>
              ORDER 4 · k = 4
            </p>
            <p style={{ fontFamily: 'Atkinson Hyperlegible', fontSize: 17, color: '#171717', lineHeight: 1.5, marginBottom: 8 }}>
              <strong style={{ color: '#2563EB' }}>SGD</strong> remains smooth and roughly parallel to order-3 — context helps, but does not change the shape.
            </p>
            <p style={{ fontFamily: 'Atkinson Hyperlegible', fontSize: 17, color: '#171717', lineHeight: 1.5 }}>
              <strong style={{ color: '#EA580C' }}>DP-SGD</strong> cliff shifts to the right: more structure ⇒ more shift tolerated before collapse.
            </p>
          </div>

          <div style={{
            background: '#FEF3C7', padding: '12px 18px',
            border: '1px solid #FCD34D', borderLeft: '4px solid #F59E0B',
            borderRadius: 8,
          }}>
            <p style={{ fontFamily: 'Atkinson Hyperlegible', fontSize: 16, color: '#92400E', lineHeight: 1.5 }}>
              <strong>Order-3 vs Order-4.</strong> Increasing k moves — but never erases — the divergence threshold.
            </p>
          </div>

          <div style={{
            background: '#F0FDF4', padding: '12px 18px',
            border: '1px solid #A7F3D0', borderLeft: '4px solid #10B981',
            borderRadius: 8,
          }}>
            <p style={{ fontFamily: 'Atkinson Hyperlegible', fontSize: 16, color: '#065F46', lineHeight: 1.5 }}>
              The cliff is <strong>structural</strong>, not an artefact of a particular chain order.
            </p>
          </div>
        </motion.div>
      </div>
    </Slide>
  )
}
