import { motion } from 'framer-motion'
import { Slide } from '../deck/Slide'
import { SpearmanBars } from '../components/SpearmanBars'
import { Math as TeX } from '../components/Math'

export function Slide13() {
  return (
    <Slide track="image" trackIndex={4} trackTotal={5}>
      <motion.h2
        initial={{ opacity: 0 }} animate={{ opacity: 1 }}
        className="font-serif-display"
        style={{ fontSize: 36, fontWeight: 600, color: '#171717', marginBottom: 6 }}>
        Standard Divergence Metrics Cannot Predict DP Accuracy
      </motion.h2>
      <motion.p
        initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.1 }}
        style={{ fontFamily: 'Atkinson Hyperlegible', fontSize: 22, color: '#6B7280', marginBottom: 12 }}>
        Spearman ρ between each metric and DP accuracy at ε = 0.5. <strong style={{ color: '#171717' }}>Click any bar</strong> for its formula and scatter.
      </motion.p>

      <div className="flex gap-4 flex-1 min-h-0">
        {/* Left: bars */}
        <motion.div
          initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.2 }}
          className="overflow-hidden"
          style={{ flex: 1 }}>
          <SpearmanBars />
        </motion.div>

        {/* Right: compact Spearman primer + MAUVE warning */}
        <motion.div
          initial={{ opacity: 0, x: 16 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.4 }}
          style={{ width: 320, flexShrink: 0, display: 'flex', flexDirection: 'column', gap: 14, justifyContent: 'center' }}>

          <div style={{
            padding: '16px 18px', background: '#EEF2FF',
            borderRadius: 10, border: '1px solid #C7D2FE',
            borderLeft: '4px solid #6366F1',
          }}>
            <p style={{ fontFamily: 'Atkinson Hyperlegible', fontSize: 18, color: '#4F46E5', fontWeight: 700, letterSpacing: 0.3, marginBottom: 8 }}>
              SPEARMAN ρ
            </p>
            <div style={{ background: '#FFFFFF', borderRadius: 6, padding: '8px 10px', marginBottom: 10 }}>
              <TeX latex="\rho = 1 - \frac{6 \sum d_i^2}{n(n^2 - 1)}" block size={0.95} />
            </div>
            <p style={{ fontFamily: 'Atkinson Hyperlegible', fontSize: 18, color: '#171717', lineHeight: 1.45 }}>
              Rank-correlation with DP accuracy.
            </p>
            <p style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 15, color: '#4B5563', lineHeight: 1.5, marginTop: 6 }}>
              +1 agree · 0 none · −1 inverted
            </p>
          </div>

          <div style={{
            padding: '14px 18px', background: '#FFFBEB',
            border: '1px solid #FDE68A', borderLeft: '4px solid #F59E0B',
            borderRadius: 8,
          }}>
            <p style={{ fontFamily: 'Atkinson Hyperlegible', fontSize: 18, color: '#92400E', fontWeight: 700, letterSpacing: 0.3, marginBottom: 6 }}>
              MAUVE SATURATES
            </p>
            <p style={{ fontFamily: 'Atkinson Hyperlegible', fontSize: 17, color: '#78350F', lineHeight: 1.5 }}>
              All 10 runs in <strong style={{ fontFamily: 'JetBrains Mono, monospace' }}>[0.996, 0.998]</strong> - no discriminative power.
            </p>
          </div>
        </motion.div>
      </div>
    </Slide>
  )
}
