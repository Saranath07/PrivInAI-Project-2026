import { motion } from 'framer-motion'
import { Slide } from '../deck/Slide'
import { LinearProbeScatter } from '../components/LinearProbeScatter'
import { CountUp } from '../components/CountUp'
import { Math as TeX } from '../components/Math'

export function Slide15() {
  return (
    <Slide track="image" trackIndex={5} trackTotal={5}>
      <motion.div
        initial={{ opacity: 0 }} animate={{ opacity: 1 }}
        style={{ marginBottom: 4 }}>
        <h2 className="font-serif-display" style={{ fontSize: 30, fontWeight: 700, color: '#171717' }}>
          Linear Probe Accuracy: ρ ={' '}
          <CountUp to={0.927} decimals={3} duration={900}
            className="font-mono-data" style={{ color: '#10B981', fontSize: 34 }} />
        </h2>
        <p style={{ fontFamily: 'Atkinson Hyperlegible', fontSize: 22, color: '#6B7280', marginTop: 2 }}>
          Our winning metric: freeze the encoder, train one linear layer, predict DP accuracy.
        </p>
      </motion.div>

      <div className="flex gap-5 flex-1 min-h-0">
        {/* Left: Linear probe definition + scatter */}
        <div className="flex flex-col gap-3" style={{ flex: 1, minWidth: 0 }}>
          <motion.div
            initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.15 }}
            style={{
              padding: '10px 14px', background: '#FFFFFF',
              border: '1px solid #E5E5E5', borderLeft: '4px solid #10B981',
              borderRadius: 10,
            }}>
            <p style={{ fontFamily: 'Atkinson Hyperlegible', fontSize: 20, color: '#059669', fontWeight: 700, letterSpacing: 0.3, marginBottom: 6 }}>
              WHAT IS A LINEAR PROBE?
            </p>
            <div className="flex gap-3 items-center">
              <div style={{ flex: 1 }}>
                <p style={{ fontFamily: 'Atkinson Hyperlegible', fontSize: 20, color: '#171717', lineHeight: 1.5 }}>
                  Freeze the encoder φ pretrained on <strong>P</strong> (public). Train <strong>only a linear head</strong> (W, b) on labeled <strong>Q</strong> (private). Report the test accuracy.
                </p>
                <p style={{ fontFamily: 'Atkinson Hyperlegible', fontSize: 20, color: '#4B5563', marginTop: 6, lineHeight: 1.5 }}>
                  Directly measures: <em>"are these frozen features already useful for the downstream task?"</em> - which is exactly what DP-SGD is effectively forced to do.
                </p>
              </div>
              <div style={{
                background: '#ECFDF5', padding: '10px 14px', borderRadius: 8, flexShrink: 0,
              }}>
                <TeX latex="\hat{y} = W \phi(x) + b" block size={0.95} />
              </div>
            </div>
          </motion.div>

          <motion.div
            initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.3 }}
            style={{ flex: 1, minHeight: 0 }}>
            <LinearProbeScatter width={520} height={260} />
          </motion.div>
        </div>

        {/* Right panel */}
        <div className="flex flex-col gap-4 justify-center" style={{ width: 230, flexShrink: 0 }}>
          <motion.div
            initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.4 }}
            className="flex flex-col items-center"
            style={{
              padding: '14px 10px', background: '#ECFDF5',
              borderRadius: 12, border: '1px solid #A7F3D0',
            }}>
            <CountUp to={0.927} decimals={3} duration={900}
              className="font-mono-data"
              style={{ fontSize: 48, fontWeight: 700, color: '#059669', lineHeight: 1 }} />
            <span style={{ fontFamily: 'Atkinson Hyperlegible', fontSize: 20, color: '#047857', marginTop: 6, textAlign: 'center' }}>
              Spearman ρ<br />(linear probe vs DP acc)
            </span>
          </motion.div>

          <div className="flex flex-col gap-2">
            {[
              'Best predictor - next best: MAUVE at 0.406',
              'No DP-SGD needed to compute it',
              'Stable across all ε values',
            ].map((text, i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0, x: 8 }} animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.6 + i * 0.1 }}
                className="flex items-start gap-2">
                <span style={{ color: '#10B981', fontSize: 22, marginTop: 0, flexShrink: 0, fontWeight: 700 }}>✓</span>
                <p style={{ fontFamily: 'Atkinson Hyperlegible', fontSize: 20, color: '#374151', lineHeight: 1.45 }}>{text}</p>
              </motion.div>
            ))}
          </div>

          <motion.div
            initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 1.1 }}
            style={{
              padding: '10px 14px',
              background: '#ECFDF5',
              borderRadius: 10,
              border: '1px solid #6EE7B7',
              borderLeft: '3px solid #10B981',
            }}>
            <p style={{ fontFamily: 'Atkinson Hyperlegible', fontSize: 20, color: '#065F46', lineHeight: 1.5 }}>
              <strong>Practical:</strong> rank candidate pretrain datasets with a cheap linear probe before spending DP budget.
            </p>
          </motion.div>
        </div>
      </div>
    </Slide>
  )
}
