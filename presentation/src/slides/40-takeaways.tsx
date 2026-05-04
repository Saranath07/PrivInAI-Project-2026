import { motion } from 'framer-motion'
import { Slide } from '../deck/Slide'

const items = [
  {
    num: '1',
    bold: 'Privacy budget matters less than pretrain quality',
    detail: 'Once good features are in place, tightening ε from 8 to 0.5 costs < 0.02 accuracy on average.',
  },
  {
    num: '2',
    bold: 'Standard divergences fail to predict DP accuracy',
    detail: 'FID, KL, JSD, MAUVE - all |ρ| < 0.5. Wasserstein has the wrong sign (ρ = −0.71).',
  },
  {
    num: '3',
    bold: 'Linear probe accuracy is the right predictor',
    detail: 'ρ = 0.927 with DP accuracy. No DP-SGD required. Cheap, reliable, portable.',
  },
]

export function Slide40() {
  return (
    <Slide track="close">
      <motion.h2
        initial={{ opacity: 0 }} animate={{ opacity: 1 }}
        className="font-serif-display"
        style={{ fontSize: 36, fontWeight: 700, color: '#171717', textAlign: 'center', marginBottom: 36 }}>
        Three Things We Found
      </motion.h2>

      <div className="flex flex-col gap-6 flex-1 justify-center">
        {items.map((item, i) => (
          <motion.div
            key={item.num}
            initial={{ opacity: 0, x: -24 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.15 + i * 0.18, duration: 0.4, ease: [0, 0, 0.2, 1] }}
            className="flex items-start gap-5">
            <motion.span
              initial={{ scale: 0.5 }} animate={{ scale: 1 }}
              transition={{ delay: 0.15 + i * 0.18, type: 'spring', stiffness: 300, damping: 20 }}
              className="font-mono-data"
              style={{ fontSize: 48, fontWeight: 700, color: '#EEF2FF', lineHeight: 1, flexShrink: 0, WebkitTextStroke: '2px #6366F1' }}>
              {item.num}
            </motion.span>
            <div>
              <p className="font-serif-display" style={{ fontSize: 22, fontWeight: 600, color: '#171717', lineHeight: 1.3 }}>
                {item.bold}
              </p>
              <p style={{ fontFamily: 'Atkinson Hyperlegible', fontSize: 14, color: '#6B7280', marginTop: 4, lineHeight: 1.5 }}>
                {item.detail}
              </p>
            </div>
          </motion.div>
        ))}
      </div>
    </Slide>
  )
}
