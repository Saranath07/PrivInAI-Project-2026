import { motion } from 'framer-motion'
import { Slide } from '../deck/Slide'
import { Math } from '../components/Math'

export function Slide14() {
  const base = import.meta.env.BASE_URL

  return (
    <Slide track="image" trackIndex={5} trackTotal={6}>
      <motion.h2
        initial={{ opacity: 0 }} animate={{ opacity: 1 }}
        className="font-serif-display"
        style={{ fontSize: 28, fontWeight: 600, color: '#171717', marginBottom: 16 }}>
        Why MAUVE Fails: Feature Space Collapse
      </motion.h2>

      <div className="flex gap-8 flex-1 min-h-0">
        {/* Left: figure */}
        <div className="flex flex-col" style={{ flex: '0 0 480px' }}>
          <motion.img
            initial={{ opacity: 0, scale: 0.97 }} animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 0.2, duration: 0.4 }}
            src={`${base}figures/fig6_hypotheses.png`}
            alt="MAUVE failure hypothesis diagram"
            style={{
              maxWidth: '100%', maxHeight: 280, objectFit: 'contain',
              border: '1px solid #E5E5E5', borderRadius: 6,
            }}
          />
          <p style={{
            fontFamily: 'Atkinson Hyperlegible', fontStyle: 'italic', fontSize: 11,
            color: '#6B7280', textAlign: 'center', marginTop: 6,
          }}>
            Hypothesis diagram: MAUVE clusters collapse when applied to CNN features
          </p>
        </div>

        {/* Right: math + bullets */}
        <div className="flex flex-col gap-4 flex-1">
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.35 }}>
            <Math latex="\text{MAUVE}(P,Q) = \int_0^1 \phi\!\left(\text{KL}(R_\lambda \| Q), \text{KL}(R_\lambda \| P)\right) d\lambda" block size={0.95} />
            <Math latex="R_\lambda = \lambda P + (1-\lambda)Q" block size={0.9} />
          </motion.div>

          <div className="flex flex-col gap-2.5">
            {[
              'MAUVE was designed for text (GPT-2 embeddings), not CNN features',
              'CNN feature clusters for SVHN vs MNIST heavily overlap in PCA space',
              'All 10 experiments land in a 0.002-wide range - zero discriminative power',
            ].map((text, i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0, x: -8 }} animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.5 + i * 0.1 }}
                className="flex items-start gap-2.5">
                <div style={{ width: 7, height: 7, borderRadius: '50%', background: '#F59E0B', flexShrink: 0, marginTop: 6 }} />
                <p style={{ fontFamily: 'Atkinson Hyperlegible', fontSize: 14, color: '#171717' }}>{text}</p>
              </motion.div>
            ))}
          </div>

          <motion.div
            initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.85 }}
            style={{
              padding: '10px 16px', background: '#FFFBEB', borderRadius: 8,
              border: '1px solid #FDE68A', display: 'inline-block',
            }}>
            <span style={{ fontFamily: 'Atkinson Hyperlegible', fontSize: 11, color: '#92400E' }}>MAUVE range across all experiments: </span>
            <span style={{ fontFamily: 'JetBrains Mono', fontSize: 14, fontWeight: 700, color: '#92400E' }}>0.9962 – 0.9983</span>
          </motion.div>
        </div>
      </div>
    </Slide>
  )
}
