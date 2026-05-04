import { motion } from 'framer-motion'
import { Slide } from '../deck/Slide'

const INK = '#171717'
const MUTE = '#6B7280'
const TEAL = '#0D9488'
const TEAL_BG = '#F0FDFA'
const TEAL_BORDER = '#99F6E4'
const INDIGO = '#6366F1'
const INDIGO_BG = '#EEF2FF'
const INDIGO_BORDER = '#C7D2FE'
const PINK = '#EC4899'
const PINK_BG = '#FDF2F8'
const PINK_BORDER = '#FBCFE8'
const AMBER = '#F59E0B'
const AMBER_BG = '#FFFBEB'
const AMBER_BORDER = '#FDE68A'

function Box({
  x, y, w, h, fill, stroke, accent, title, sub, delay = 0,
}: {
  x: number; y: number; w: number; h: number;
  fill: string; stroke: string; accent: string;
  title: string; sub?: string; delay?: number;
}) {
  return (
    <motion.g
      initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }}
      transition={{ delay, duration: 0.35 }}>
      <rect x={x} y={y} width={w} height={h} rx={8} ry={8}
        fill={fill} stroke={stroke} strokeWidth={1} />
      <rect x={x} y={y} width={3} height={h} rx={1.5} ry={1.5} fill={accent} />
      <text x={x + w / 2} y={sub ? y + h / 2 - 4 : y + h / 2 + 5}
        textAnchor="middle"
        fontFamily="Atkinson Hyperlegible"
        fontSize={13} fontWeight={700} fill={INK}>
        {title}
      </text>
      {sub && (
        <text x={x + w / 2} y={y + h / 2 + 12}
          textAnchor="middle"
          fontFamily="JetBrains Mono, monospace"
          fontSize={10} fill={MUTE}>
          {sub}
        </text>
      )}
    </motion.g>
  )
}

function Arrow({ x1, y1, x2, y2, delay = 0 }: { x1: number; y1: number; x2: number; y2: number; delay?: number }) {
  return (
    <motion.line
      x1={x1} y1={y1} x2={x2} y2={y2}
      stroke="#9CA3AF" strokeWidth={1.5}
      markerEnd="url(#arrowhead)"
      initial={{ opacity: 0 }} animate={{ opacity: 1 }}
      transition={{ delay }}
    />
  )
}

export function Slide35() {
  // Column: x centered at 260, width 380 for main column
  const cx = 300
  const bw = 360
  const bx = cx - bw / 2

  return (
    <Slide track="transformers" trackIndex={6} trackTotal={7}>
      <motion.h2
        initial={{ opacity: 0 }} animate={{ opacity: 1 }}
        className="font-serif-display"
        style={{ fontSize: 34, fontWeight: 600, color: INK, marginBottom: 4 }}>
        Math Model Architecture
      </motion.h2>
      <motion.p
        initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.1 }}
        style={{ fontFamily: 'Atkinson Hyperlegible', fontSize: 16, color: MUTE, marginBottom: 10 }}>
        A small GPT-2-style decoder: 14-token vocabulary, 2 transformer blocks, next-token prediction over math sequences.
      </motion.p>

      <div className="flex gap-6 flex-1 min-h-0">
        {/* Left: SVG architecture diagram */}
        <div style={{
          flex: 1, minWidth: 0, minHeight: 0,
          background: '#FFFFFF', border: '1px solid #E5E7EB', borderRadius: 10,
          padding: 10, display: 'flex', alignItems: 'center', justifyContent: 'center',
        }}>
          <svg viewBox="0 0 600 560" width="100%" height="100%" preserveAspectRatio="xMidYMid meet">
            <defs>
              <marker id="arrowhead" viewBox="0 0 10 10" refX="9" refY="5"
                markerWidth="6" markerHeight="6" orient="auto-start-reverse">
                <path d="M 0 0 L 10 5 L 0 10 z" fill="#9CA3AF" />
              </marker>
            </defs>

            {/* Input sequence */}
            <motion.g
              initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.2 }}>
              <text x={cx} y={22} textAnchor="middle"
                fontFamily="Atkinson Hyperlegible" fontSize={11}
                fontWeight={700} fill={MUTE} letterSpacing={0.4}>
                INPUT SEQUENCE
              </text>
              {/* token chips */}
              {['^', '2', '+', 'x', '␣', '9', '$'].map((t, i) => (
                <g key={i}>
                  <rect x={bx + 30 + i * 44} y={32} width={36} height={26} rx={5}
                    fill={TEAL_BG} stroke={TEAL_BORDER} />
                  <text x={bx + 48 + i * 44} y={50}
                    textAnchor="middle"
                    fontFamily="JetBrains Mono, monospace"
                    fontSize={13} fontWeight={700} fill={TEAL}>
                    {t}
                  </text>
                </g>
              ))}
            </motion.g>

            <Arrow x1={cx} y1={62} x2={cx} y2={80} delay={0.25} />

            {/* Token embedding */}
            <Box x={bx} y={82} w={bw} h={42}
              fill={TEAL_BG} stroke={TEAL_BORDER} accent={TEAL}
              title="Token Embedding" sub="14 × d_model"
              delay={0.3} />

            <Arrow x1={cx} y1={124} x2={cx} y2={140} delay={0.35} />

            {/* Positional encoding */}
            <Box x={bx} y={142} w={bw} h={42}
              fill={TEAL_BG} stroke={TEAL_BORDER} accent={TEAL}
              title="Positional Encoding" sub="max_ctx = 150"
              delay={0.4} />

            <Arrow x1={cx} y1={184} x2={cx} y2={202} delay={0.45} />

            {/* Transformer block 1 */}
            <motion.g
              initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.5, duration: 0.35 }}>
              <rect x={bx} y={204} width={bw} height={130} rx={10}
                fill={INDIGO_BG} stroke={INDIGO_BORDER} strokeWidth={1} />
              <rect x={bx} y={204} width={3} height={130} rx={1.5} fill={INDIGO} />
              <text x={bx + 12} y={222}
                fontFamily="Atkinson Hyperlegible" fontSize={11} fontWeight={700}
                fill={INDIGO} letterSpacing={0.4}>
                TRANSFORMER BLOCK · ×2
              </text>
              {/* MHA */}
              <rect x={bx + 16} y={232} width={bw - 32} height={36} rx={6}
                fill="#FFFFFF" stroke={INDIGO_BORDER} />
              <text x={cx} y={255} textAnchor="middle"
                fontFamily="Atkinson Hyperlegible" fontSize={12} fontWeight={700} fill={INK}>
                Masked Multi-Head Self-Attention
              </text>
              {/* Arrow */}
              <line x1={cx} y1={268} x2={cx} y2={282}
                stroke="#9CA3AF" strokeWidth={1.5} markerEnd="url(#arrowhead)" />
              {/* FFN */}
              <rect x={bx + 16} y={286} width={bw - 32} height={36} rx={6}
                fill="#FFFFFF" stroke={INDIGO_BORDER} />
              <text x={cx} y={304} textAnchor="middle"
                fontFamily="Atkinson Hyperlegible" fontSize={12} fontWeight={700} fill={INK}>
                Feed-Forward + LayerNorm
              </text>
              <text x={cx} y={317} textAnchor="middle"
                fontFamily="JetBrains Mono, monospace" fontSize={9} fill={MUTE}>
                residual · pre-norm
              </text>
            </motion.g>

            <Arrow x1={cx} y1={334} x2={cx} y2={352} delay={0.6} />

            {/* Linear head */}
            <Box x={bx} y={354} w={bw} h={42}
              fill={PINK_BG} stroke={PINK_BORDER} accent={PINK}
              title="Linear Head" sub="d_model → 14 (tied)"
              delay={0.65} />

            <Arrow x1={cx} y1={396} x2={cx} y2={414} delay={0.7} />

            {/* Softmax / next-token logits */}
            <Box x={bx} y={416} w={bw} h={42}
              fill={AMBER_BG} stroke={AMBER_BORDER} accent={AMBER}
              title="Softmax · Next-Token Distribution"
              sub="p(x_t | x_<t) over 14 tokens"
              delay={0.75} />

            <Arrow x1={cx} y1={458} x2={cx} y2={476} delay={0.8} />

            {/* Output */}
            <motion.g
              initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.85 }}>
              <rect x={bx + 40} y={478} width={bw - 80} height={38} rx={19}
                fill="#FFFFFF" stroke={AMBER} strokeWidth={1.5} />
              <text x={cx} y={502} textAnchor="middle"
                fontFamily="JetBrains Mono, monospace" fontSize={12} fontWeight={700}
                fill={AMBER}>
                predicted digit / $ / ␣
              </text>
            </motion.g>

            {/* Side annotations */}
            <motion.g
              initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.9 }}>
              <line x1={bx + bw + 6} y1={103} x2={bx + bw + 40} y2={103} stroke="#D1D5DB" />
              <line x1={bx + bw + 6} y1={163} x2={bx + bw + 40} y2={163} stroke="#D1D5DB" />
              <text x={bx + bw + 46} y={137}
                fontFamily="Atkinson Hyperlegible" fontSize={10} fill={MUTE}>
                embed
              </text>
            </motion.g>
          </svg>
        </div>

        {/* Right: legend / specs */}
        <motion.div
          initial={{ opacity: 0, x: 8 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.3 }}
          style={{ width: 300, flexShrink: 0, display: 'flex', flexDirection: 'column', gap: 10 }}>

          <div style={{
            background: TEAL_BG, border: `1px solid ${TEAL_BORDER}`,
            borderLeft: `4px solid ${TEAL}`, borderRadius: 10,
            padding: '12px 14px',
          }}>
            <p style={{ fontFamily: 'Atkinson Hyperlegible', fontSize: 11, color: TEAL, fontWeight: 700, letterSpacing: 0.4, marginBottom: 6 }}>
              INPUT STAGE
            </p>
            <p style={{ fontFamily: 'Atkinson Hyperlegible', fontSize: 13, color: INK, lineHeight: 1.5 }}>
              Tokens from the <strong>14-token</strong> vocabulary are embedded and combined with learned positional encodings up to <span style={{ fontFamily: 'JetBrains Mono, monospace' }}>max_ctx = 150</span>.
            </p>
          </div>

          <div style={{
            background: INDIGO_BG, border: `1px solid ${INDIGO_BORDER}`,
            borderLeft: `4px solid ${INDIGO}`, borderRadius: 10,
            padding: '12px 14px',
          }}>
            <p style={{ fontFamily: 'Atkinson Hyperlegible', fontSize: 11, color: INDIGO, fontWeight: 700, letterSpacing: 0.4, marginBottom: 6 }}>
              BACKBONE · TINYMEM
            </p>
            <p style={{ fontFamily: 'Atkinson Hyperlegible', fontSize: 13, color: INK, lineHeight: 1.5 }}>
              Two stacked decoder blocks with masked self-attention and MLP sublayers, GPT-2 style pre-norm residuals.
            </p>
          </div>

          <div style={{
            background: PINK_BG, border: `1px solid ${PINK_BORDER}`,
            borderLeft: `4px solid ${PINK}`, borderRadius: 10,
            padding: '12px 14px',
          }}>
            <p style={{ fontFamily: 'Atkinson Hyperlegible', fontSize: 11, color: PINK, fontWeight: 700, letterSpacing: 0.4, marginBottom: 6 }}>
              OUTPUT HEAD
            </p>
            <p style={{ fontFamily: 'Atkinson Hyperlegible', fontSize: 13, color: INK, lineHeight: 1.5 }}>
              Tied linear projection back to the 14-token vocabulary, then softmax for next-token prediction.
            </p>
          </div>

          <div style={{
            background: '#F9FAFB', border: '1px solid #E5E7EB', borderRadius: 10,
            padding: '12px 14px',
          }}>
            <p style={{ fontFamily: 'Atkinson Hyperlegible', fontSize: 11, color: MUTE, fontWeight: 700, letterSpacing: 0.4, marginBottom: 6 }}>
              OBJECTIVE
            </p>
            <p style={{ fontFamily: 'Atkinson Hyperlegible', fontSize: 13, color: '#374151', lineHeight: 1.5 }}>
              Causal LM loss: <span style={{ fontFamily: 'JetBrains Mono, monospace' }}>−Σ log p(x_t | x_&lt;t)</span> over every digit/delimiter in the sequence.
            </p>
          </div>
        </motion.div>
      </div>
    </Slide>
  )
}
