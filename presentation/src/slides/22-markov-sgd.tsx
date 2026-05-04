import { motion } from 'framer-motion'
import { Slide } from '../deck/Slide'
import { Math as TeX } from '../components/Math'

export function Slide22() {
  const base = import.meta.env.BASE_URL
  return (
    <Slide track="markov" trackIndex={3} trackTotal={5}>
      <motion.h2
        initial={{ opacity: 0 }} animate={{ opacity: 1 }}
        className="font-serif-display"
        style={{ fontSize: 36, fontWeight: 600, color: '#171717', marginBottom: 4 }}>
        Order-3 Chains
      </motion.h2>
      <motion.p
        initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.1 }}
        style={{ fontFamily: 'Atkinson Hyperlegible', fontSize: 20, color: '#6B7280', marginBottom: 14 }}>
        Perplexity and MAUVE vs λ, evaluated across five Rényi orders α. Blue = vanilla SGD · Orange = DP-SGD.
      </motion.p>

      <div className="flex gap-6 flex-1 min-h-0">
        {/* Left: metric definitions above the figure */}
        <motion.div
          initial={{ opacity: 0, scale: 0.98 }} animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.2, duration: 0.4 }}
          style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: 10, minWidth: 0 }}>

          <div style={{
            background: '#FFFFFF', border: '1px solid #E5E5E5', borderRadius: 10,
            padding: '12px 18px', display: 'flex', gap: 28, alignItems: 'center', justifyContent: 'center',
          }}>
            <div>
              <p style={{ fontFamily: 'Atkinson Hyperlegible', fontSize: 12, color: '#6B7280', fontWeight: 700, letterSpacing: 0.3, marginBottom: 2 }}>PERPLEXITY</p>
              <TeX latex="\mathrm{PPL}(P) = \exp\!\left(-\tfrac{1}{N}\sum_i \log p_\theta(x_i \mid x_{<i})\right)" block size={0.85} />
            </div>
            <div style={{ width: 1, alignSelf: 'stretch', background: '#E5E5E5' }} />
            <div>
              <p style={{ fontFamily: 'Atkinson Hyperlegible', fontSize: 12, color: '#6B7280', fontWeight: 700, letterSpacing: 0.3, marginBottom: 2 }}>MAUVE</p>
              <TeX latex="\mathrm{MAUVE}(P,Q) = \mathrm{AUC}\Big\{\big(e^{-c\,\mathrm{KL}(Q\,\|\,R_\lambda)},\; e^{-c\,\mathrm{KL}(P\,\|\,R_\lambda)}\big) : \lambda \in (0,1)\Big\}" block size={0.7} />
              <p style={{ fontFamily: 'Atkinson Hyperlegible', fontSize: 10, color: '#9CA3AF', marginTop: 2 }}>
                where <TeX latex="R_\lambda = \lambda P + (1-\lambda) Q" />
              </p>
            </div>
          </div>

          <div style={{
            flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center',
            background: '#FFFFFF', border: '1px solid #E5E5E5', borderRadius: 10, padding: 8,
            minHeight: 0,
          }}>
            <img
              src={`${base}markov/results-order-3.png`}
              alt="Order-3 Markov results: perplexity and MAUVE vs λ"
              style={{ maxWidth: '100%', maxHeight: '100%', objectFit: 'contain' }}
            />
          </div>
        </motion.div>

        {/* Right: takeaways */}
        <motion.div
          initial={{ opacity: 0, x: 16 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.3 }}
          style={{ width: 340, flexShrink: 0, display: 'flex', flexDirection: 'column', gap: 14, justifyContent: 'center' }}>

          <div style={{
            background: '#EEF2FF', padding: '14px 18px',
            border: '1px solid #C7D2FE', borderLeft: '4px solid #6366F1',
            borderRadius: 10,
          }}>
            <p style={{ fontFamily: 'Atkinson Hyperlegible', fontSize: 15, color: '#4F46E5', fontWeight: 700, letterSpacing: 0.3, marginBottom: 8 }}>
              ORDER 3 · k = 3
            </p>
            <p style={{ fontFamily: 'Atkinson Hyperlegible', fontSize: 17, color: '#171717', lineHeight: 1.5, marginBottom: 8 }}>
              <strong style={{ color: '#4F46E5' }}>Perplexity.</strong> Increases for both SGD and DP-SGD with DP-SGD sitting above SGD due to added noise during training.
            </p>
            <p style={{ fontFamily: 'Atkinson Hyperlegible', fontSize: 17, color: '#171717', lineHeight: 1.5 }}>
              <strong style={{ color: '#4F46E5' }}>MAUVE.</strong> DP-SGD below SGD as generated test from P and Q are closer in SGD's case. Erratic behaviour observed due to the small size of dataset.
            </p>
          </div>
        </motion.div>
      </div>
    </Slide>
  )
}
