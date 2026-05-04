import { motion } from 'framer-motion'
import { Slide } from '../deck/Slide'
import { Math as TeX } from '../components/Math'

const vocab: { tok: string; id: string; role: string }[] = [
  { tok: '0-9', id: '0-9',   role: 'digits' },
  { tok: '^',   id: '10',    role: 'BOS' },
  { tok: '$',   id: '11',    role: 'EOS' },
  { tok: '␣',   id: '12',    role: 'delimiter' },
  { tok: '_',   id: '13',    role: 'padding' },
]

const targets: { expr: string; kind: string; color: string; bg: string; border: string }[] = [
  { expr: '11 + x',     kind: 'additive · OOD step',   color: '#6366F1', bg: '#EEF2FF', border: '#C7D2FE' },
  { expr: 'x - 7',      kind: 'subtractive · reversed', color: '#0EA5E9', bg: '#F0F9FF', border: '#BAE6FD' },
  { expr: '2·x mod 20134', kind: 'multiplicative',      color: '#EC4899', bg: '#FDF2F8', border: '#FBCFE8' },
  { expr: 'x² mod 20134',  kind: 'polynomial',          color: '#F59E0B', bg: '#FFFBEB', border: '#FDE68A' },
]

const dp: { sigma: string; eps: string }[] = [
  { sigma: '1.79', eps: '7.17' },
  { sigma: '2.15', eps: '5.53' },
  { sigma: '2.90', eps: '3.72' },
  { sigma: '5.07', eps: '1.87' },
]

export function Slide34() {
  return (
    <Slide track="transformers" trackIndex={5} trackTotal={7}>
      <motion.h2
        initial={{ opacity: 0 }} animate={{ opacity: 1 }}
        className="font-serif-display"
        style={{ fontSize: 34, fontWeight: 600, color: '#171717', marginBottom: 4 }}>
        Math-Sequence Models
      </motion.h2>
      <motion.p
        initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.1 }}
        style={{ fontFamily: 'Atkinson Hyperlegible', fontSize: 16, color: '#6B7280', marginBottom: 14 }}>
        A cleaner transformer testbed: GPT-2 pretrained on arithmetic sequences, fine-tuned on four shifted operators with and without DP.
      </motion.p>

      <div className="flex gap-5 flex-1 min-h-0">
        {/* Left column: pretrain + vocab */}
        <motion.div
          initial={{ opacity: 0, x: -8 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.2 }}
          style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: 10, minWidth: 0 }}>

          <div style={{
            background: '#F0FDFA', border: '1px solid #99F6E4',
            borderLeft: '4px solid #0D9488', borderRadius: 10,
            padding: '12px 16px',
          }}>
            <p style={{ fontFamily: 'Atkinson Hyperlegible', fontSize: 12, color: '#0D9488', fontWeight: 700, letterSpacing: 0.4, marginBottom: 6 }}>
              PRETRAIN · GPT-2 BASE
            </p>
            <p style={{ fontFamily: 'Atkinson Hyperlegible', fontSize: 14, color: '#171717', lineHeight: 1.5 }}>
              GPT-2 based model pretrained on math sequences
              <span style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 13 }}> (2+x), (3+x), (4+x), (5+x), (7+x)</span>.
            </p>
          </div>

          <div style={{
            background: '#FFFFFF', border: '1px solid #E5E7EB', borderRadius: 10,
            padding: '12px 16px', flex: 1, display: 'flex', flexDirection: 'column', minHeight: 0,
          }}>
            <p style={{ fontFamily: 'Atkinson Hyperlegible', fontSize: 12, color: '#6B7280', fontWeight: 700, letterSpacing: 0.4, marginBottom: 8 }}>
              VOCABULARY · 14 TOKENS · max_ctx = 150
            </p>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 6 }}>
              {vocab.map(v => (
                <div key={v.id} style={{
                  display: 'flex', alignItems: 'center', gap: 8,
                  padding: '6px 10px',
                  background: '#F9FAFB', border: '1px solid #E5E7EB', borderRadius: 6,
                }}>
                  <span style={{
                    fontFamily: 'JetBrains Mono, monospace', fontSize: 14, fontWeight: 700,
                    color: '#0D9488', minWidth: 28,
                  }}>{v.tok}</span>
                  <span style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 11, color: '#9CA3AF' }}>
                    id {v.id}
                  </span>
                  <span style={{ fontFamily: 'Atkinson Hyperlegible', fontSize: 12, color: '#374151', marginLeft: 'auto' }}>
                    {v.role}
                  </span>
                </div>
              ))}
            </div>
            <p style={{
              fontFamily: 'Atkinson Hyperlegible', fontSize: 12, color: '#6B7280',
              lineHeight: 1.5, marginTop: 10,
            }}>
              Numbers are represented digit-by-digit; a space separates consecutive numbers; padding fills sequences up to max_ctx.
            </p>
          </div>
        </motion.div>

        {/* Right column: fine-tune targets + DP config */}
        <motion.div
          initial={{ opacity: 0, x: 8 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.3 }}
          style={{ flex: 1.1, display: 'flex', flexDirection: 'column', gap: 10, minWidth: 0 }}>

          <div style={{
            background: '#FFFFFF', border: '1px solid #E5E7EB', borderRadius: 10,
            padding: '12px 16px',
          }}>
            <p style={{ fontFamily: 'Atkinson Hyperlegible', fontSize: 12, color: '#6B7280', fontWeight: 700, letterSpacing: 0.4, marginBottom: 6 }}>
              FINE-TUNE · 2-LAYER TINYMEM · SIMILARITY GRADIENT
            </p>
            <p style={{ fontFamily: 'Atkinson Hyperlegible', fontSize: 13, color: '#374151', lineHeight: 1.5, marginBottom: 8 }}>
              Pretrained on <span style={{ fontFamily: 'JetBrains Mono, monospace' }}>7+x</span>; fine-tuned on four target sequences:
            </p>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 6 }}>
              {targets.map(t => (
                <div key={t.expr} style={{
                  padding: '8px 10px', background: t.bg,
                  border: `1px solid ${t.border}`, borderLeft: `3px solid ${t.color}`,
                  borderRadius: 6,
                }}>
                  <p style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 14, fontWeight: 700, color: t.color }}>
                    {t.expr}
                  </p>
                  <p style={{ fontFamily: 'Atkinson Hyperlegible', fontSize: 11, color: '#6B7280', marginTop: 2 }}>
                    {t.kind}
                  </p>
                </div>
              ))}
            </div>
          </div>

          <div style={{
            background: '#F9FAFB', border: '1px solid #E5E7EB', borderRadius: 10,
            padding: '12px 16px',
          }}>
            <p style={{ fontFamily: 'Atkinson Hyperlegible', fontSize: 12, color: '#6B7280', fontWeight: 700, letterSpacing: 0.4, marginBottom: 6 }}>
              TRAINING CONFIG
            </p>
            <p style={{ fontFamily: 'Atkinson Hyperlegible', fontSize: 13, color: '#374151', lineHeight: 1.6 }}>
              2000 seqs / target · <strong>1600 train · 200 eval · 200 test</strong> · 5 seeds · 100 epochs · batch 128.
              Baseline <strong>AdamW</strong> ("no-DP") vs <strong>DP-SGD</strong> at four noise multipliers.
            </p>
          </div>

          <div style={{
            background: '#EEF2FF', border: '1px solid #C7D2FE',
            borderLeft: '4px solid #6366F1', borderRadius: 10,
            padding: '12px 16px',
          }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 6 }}>
              <p style={{ fontFamily: 'Atkinson Hyperlegible', fontSize: 12, color: '#4F46E5', fontWeight: 700, letterSpacing: 0.4 }}>
                DP-SGD BUDGETS
              </p>
              <span style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 11, color: '#4F46E5' }}>
                <TeX latex="\delta = 5\times 10^{-4}" />
              </span>
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 6 }}>
              {dp.map(d => (
                <div key={d.sigma} style={{
                  padding: '6px 8px',
                  background: '#FFFFFF', border: '1px solid #C7D2FE', borderRadius: 6,
                  textAlign: 'center',
                }}>
                  <p style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 11, color: '#6B7280' }}>
                    σ = {d.sigma}
                  </p>
                  <p style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 14, fontWeight: 700, color: '#4F46E5', marginTop: 2 }}>
                    ε = {d.eps}
                  </p>
                </div>
              ))}
            </div>
          </div>
        </motion.div>
      </div>
    </Slide>
  )
}
