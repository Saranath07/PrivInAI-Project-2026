import { motion } from 'framer-motion'
import { Slide } from '../deck/Slide'

export function Slide24() {
  const base = import.meta.env.BASE_URL
  return (
    <Slide track="markov" trackIndex={5} trackTotal={5}>
      <motion.h2
        initial={{ opacity: 0 }} animate={{ opacity: 1 }}
        className="font-serif-display"
        style={{ fontSize: 38, fontWeight: 600, color: '#171717', marginBottom: 4 }}>
        Correlated Noise Performs Worse Than DP-SGD
      </motion.h2>
      <motion.p
        initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.1 }}
        style={{ fontFamily: 'Atkinson Hyperlegible', fontSize: 20, color: '#6B7280', marginBottom: 14 }}>
        Order-3 chains with correlated noise injected at each step.
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
            src={`${base}markov/order-3-correlated-noise.png`}
            alt="Order-3 Markov results with correlated noise"
            style={{ maxWidth: '100%', maxHeight: '100%', objectFit: 'contain' }}
          />
        </motion.div>

        {/* Right: explanation */}
        <motion.div
          initial={{ opacity: 0, x: 16 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.3 }}
          style={{ width: 400, flexShrink: 0, display: 'flex', flexDirection: 'column', gap: 14, justifyContent: 'center' }}>

          <div style={{
            background: '#EEF2FF', padding: '14px 18px',
            border: '1px solid #C7D2FE', borderLeft: '4px solid #6366F1',
            borderRadius: 10,
          }}>
            <p style={{ fontFamily: 'Atkinson Hyperlegible', fontSize: 15, color: '#4F46E5', fontWeight: 700, letterSpacing: 0.3, marginBottom: 8 }}>
              CORRELATED NOISE · ORDER 3
            </p>
            <p style={{ fontFamily: 'Atkinson Hyperlegible', fontSize: 17, color: '#171717', lineHeight: 1.5 }}>
              Correlated noise sits even <strong>higher above DP-SGD</strong>, which is counterintuitive.
            </p>
          </div>

          <div style={{
            background: '#FEF3C7', padding: '14px 18px',
            border: '1px solid #FCD34D', borderLeft: '4px solid #F59E0B',
            borderRadius: 8,
          }}>
            <p style={{ fontFamily: 'Atkinson Hyperlegible', fontSize: 16, color: '#92400E', lineHeight: 1.6 }}>
              This is mainly because if the noise makes the gradient go along an incorrect direction, deriving subsequent noise from this will result in the gradient following that incorrect path — potentially leading to worse utility than uncorrelated noise.
            </p>
          </div>

        </motion.div>
      </div>
    </Slide>
  )
}
