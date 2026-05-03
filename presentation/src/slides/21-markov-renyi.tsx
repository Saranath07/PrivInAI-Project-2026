import { motion } from 'framer-motion'
import { Slide } from '../deck/Slide'
import { Math as TeX } from '../components/Math'

export function Slide21() {
  const base = import.meta.env.BASE_URL
  return (
    <Slide track="markov" trackIndex={2} trackTotal={5}>
      <motion.h2
        initial={{ opacity: 0 }} animate={{ opacity: 1 }}
        className="font-serif-display"
        style={{ fontSize: 36, fontWeight: 600, color: '#171717', marginBottom: 4 }}>
        Rényi Divergence Grows Monotonically with λ
      </motion.h2>
      <motion.p
        initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.1 }}
        style={{ fontFamily: 'Atkinson Hyperlegible', fontSize: 20, color: '#6B7280', marginBottom: 14 }}>
        The Renyi Divergence between P and Q is then calculated to quantify the shift due to induced noise through convex interpolation. This lambda is our controllable benchmark. Seen to be monotonically increasing, with greater values for lesser k.
      </motion.p>

      <div className="flex gap-6 flex-1 min-h-0">
        {/* Left: perturbed-transition equation + figure */}
        <motion.div
          initial={{ opacity: 0, scale: 0.98 }} animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.2, duration: 0.4 }}
          style={{
            flex: 1, display: 'flex', flexDirection: 'column', gap: 12, minWidth: 0,
          }}>
          <div style={{
            background: '#FFFFFF', border: '1px solid #E5E5E5', borderRadius: 10,
            padding: '14px 20px', display: 'flex', flexDirection: 'column', alignItems: 'center',
          }}>
            <p style={{ fontFamily: 'Atkinson Hyperlegible', fontSize: 14, color: '#6B7280', fontWeight: 700, letterSpacing: 0.3, marginBottom: 6 }}>
              PERTURBED TRANSITION (the shift we are measuring)
            </p>
            <TeX
              latex="[T]_Q(w' \mid c) = (1-\lambda)[T]_P(w' \mid c) + \lambda \cdot U(V)"
              block size={1.15}
            />
          </div>
          <div style={{
            flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center',
            background: '#FFFFFF', border: '1px solid #E5E5E5', borderRadius: 10, padding: 12,
            minHeight: 0,
          }}>
            <img
              src={`${base}markov/renyi_vs_lambda.jpeg`}
              alt="Rényi divergence vs λ across Markov orders"
              style={{ maxWidth: '100%', maxHeight: '100%', objectFit: 'contain' }}
            />
          </div>
        </motion.div>

        {/* Right: definition + takeaways */}
        <motion.div
          initial={{ opacity: 0, x: 16 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.3 }}
          style={{ width: 360, flexShrink: 0, display: 'flex', flexDirection: 'column', gap: 14, justifyContent: 'center' }}>

          <div style={{
            background: '#FFFFFF', padding: '16px 18px',
            border: '1px solid #E5E5E5', borderRadius: 10,
          }}>
            <p style={{ fontFamily: 'Atkinson Hyperlegible', fontSize: 15, color: '#6B7280', fontWeight: 700, letterSpacing: 0.3, marginBottom: 10 }}>
              RÉNYI DIVERGENCE
            </p>
            <TeX
              latex="D_\alpha(P \| Q) = \frac{1}{\alpha-1}\log \sum_x P(x)^\alpha Q(x)^{1-\alpha}"
              block size={0.9}
            />
          </div>

          <div style={{
            background: '#EEF2FF', padding: '14px 18px',
            border: '1px solid #C7D2FE', borderLeft: '4px solid #6366F1',
            borderRadius: 10,
          }}>
            <p style={{ fontFamily: 'Atkinson Hyperlegible', fontSize: 17, color: '#171717', lineHeight: 1.5, marginBottom: 8 }}>
              <strong style={{ color: '#4F46E5' }}>Monotonic in λ.</strong> Divergence rises smoothly from 0 as we mix in uniform noise.
            </p>
            <p style={{ fontFamily: 'Atkinson Hyperlegible', fontSize: 17, color: '#171717', lineHeight: 1.5, marginBottom: 8 }}>
              <strong style={{ color: '#4F46E5' }}>Order-3 dominates.</strong> Shorter contexts have flatter P, so uniform noise perturbs them most.
            </p>
            <p style={{ fontFamily: 'Atkinson Hyperlegible', fontSize: 17, color: '#171717', lineHeight: 1.5 }}>
              <strong style={{ color: '#4F46E5' }}>α inflates the scale</strong> but preserves rank — larger α weighs tail events more heavily.
            </p>
          </div>

          <div style={{
            background: '#F0FDF4', padding: '12px 18px',
            border: '1px solid #A7F3D0', borderLeft: '4px solid #10B981',
            borderRadius: 8,
          }}>
            <p style={{ fontFamily: 'Atkinson Hyperlegible', fontSize: 16, color: '#065F46', lineHeight: 1.5 }}>
              ⇒ λ is a <strong>calibrated</strong> shift knob. Now test what happens when we train under it.
            </p>
          </div>
        </motion.div>
      </div>
    </Slide>
  )
}
