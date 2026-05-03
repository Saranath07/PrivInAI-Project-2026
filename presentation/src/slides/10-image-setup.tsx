import { useState } from 'react'
import { motion } from 'framer-motion'
import { Slide } from '../deck/Slide'
import { Modal } from '../components/Modal'
import { Math as TeX } from '../components/Math'

interface Experiment {
  id: string
  label: string
  tag: string
  tagColor: string
  tagBg: string
  description: string
  visual: 'baseline' | 'full' | 'matched' | 'mismatched' | 'ae' | 'scarcity' | 'domain' | 'aug' | 'format'
  result: string
  accent: string
}

const experiments: Experiment[] = [
  {
    id: 'IMG-00', label: 'No Pretrain', tag: 'Baseline', tagColor: '#6B7280', tagBg: '#F3F4F6',
    description: 'Random weights, trained from scratch with DP-SGD on MNIST. Sets the floor — shows how much pretraining helps at all.',
    visual: 'baseline', result: 'Lowest accuracy (~92%) — confirms pretrain is essential', accent: '#6B7280',
  },
  {
    id: 'IMG-01', label: 'SVHN Full', tag: 'Domain Gap', tagColor: '#6366F1', tagBg: '#EEF2FF',
    description: 'Pretrain on all of SVHN (colour street-view digits), then privately finetune on MNIST (grey handwritten digits). Same 10 classes, very different image style.',
    visual: 'full', result: 'Strong accuracy (~96%) — same label space bridges the gap', accent: '#6366F1',
  },
  {
    id: 'IMG-02', label: 'SVHN {0-4}', tag: 'Matched Classes', tagColor: '#10B981', tagBg: '#ECFDF5',
    description: 'Pretrain only on SVHN digits 0-4; finetune on MNIST digits 0-4. Both data source AND label space are aligned — the purest test of domain gap.',
    visual: 'matched', result: 'Best DP accuracy (~98.5%) — matched classes win', accent: '#10B981',
  },
  {
    id: 'IMG-03', label: 'SVHN {5-9}', tag: 'Label Shift', tagColor: '#F59E0B', tagBg: '#FFFBEB',
    description: 'Pretrain on digits 5-9, finetune on digits 0-4. The label spaces are disjoint — tests whether feature transfer survives label mismatch.',
    visual: 'mismatched', result: 'Still ~96.8% — feature quality beats label alignment', accent: '#F59E0B',
  },
  {
    id: 'IMG-04', label: 'SVHN Autoencoder', tag: 'Unsupervised', tagColor: '#EC4899', tagBg: '#FDF2F8',
    description: 'Pretrain an autoencoder (no labels) on SVHN. Tests whether unsupervised feature learning produces discriminative features for DP finetuning.',
    visual: 'ae', result: 'Worst (~87.7%) — unsupervised features lack class discrimination', accent: '#EC4899',
  },
  {
    id: 'IMG-05-07', label: 'SVHN 10/25/50%', tag: 'Data Scarcity', tagColor: '#8B5CF6', tagBg: '#F5F3FF',
    description: 'Use only 10%, 25%, or 50% of SVHN for pretraining. Tests whether data quantity during pretraining predicts DP accuracy.',
    visual: 'scarcity', result: 'Accuracy drops modestly with less data but saturates early', accent: '#8B5CF6',
  },
  {
    id: 'IMG-08', label: 'CIFAR-10', tag: 'Large Gap', tagColor: '#EF4444', tagBg: '#FEF2F2',
    description: 'Pretrain on CIFAR-10 (natural photos — animals, vehicles). Much larger visual gap to MNIST than SVHN. Tests cross-domain transfer limits.',
    visual: 'domain', result: 'Lower (~94.2%) — bigger domain gap hurts but doesn\'t break it', accent: '#EF4444',
  },
  {
    id: 'IMG-09', label: 'SVHN+Aug', tag: 'Style Shift', tagColor: '#14B8A6', tagBg: '#F0FDFA',
    description: 'Pretrain on SVHN with heavy augmentation (colour jitter, blur). Makes pretrain distribution more varied — closer to or further from MNIST?',
    visual: 'aug', result: '~95.8% — augmentation does not help as much as label matching', accent: '#14B8A6',
  },
  {
    id: 'IMG-10', label: 'FashionMNIST', tag: 'Format Match', tagColor: '#F97316', tagBg: '#FFF7ED',
    description: 'Pretrain on FashionMNIST (28×28 greyscale clothing). Same image format as MNIST but completely different semantics (clothes vs digits).',
    visual: 'format', result: '~93.5% — same pixel format but wrong semantics = weak features', accent: '#F97316',
  },
]

function ExperimentVisual({ visual, accent }: { visual: Experiment['visual']; accent: string }) {
  const dots = (n: number, color: string, cx: number, cy: number, r: number) =>
    Array.from({ length: n }, (_, i) => {
      const a = (i / n) * Math.PI * 2
      return <circle key={i} cx={cx + Math.cos(a) * r} cy={cy + Math.sin(a) * r} r={3.5} fill={color} opacity={0.85} />
    })

  if (visual === 'baseline') return (
    <svg width={180} height={110} viewBox="0 0 180 110">
      <text x={90} y={50} textAnchor="middle" fontSize={42} dominantBaseline="middle">🎲</text>
      <text x={90} y={92} textAnchor="middle" fontSize={12} fill="#9CA3AF" fontFamily="Atkinson Hyperlegible">Random init — no pretrain</text>
    </svg>
  )

  if (visual === 'full') return (
    <svg width={240} height={110} viewBox="0 0 240 110">
      {dots(14, '#F59E0B', 55, 50, 28)}
      <text x={55} y={95} textAnchor="middle" fontSize={12} fill="#6B7280" fontFamily="Atkinson Hyperlegible">SVHN (colour)</text>
      <line x1={95} y1={50} x2={140} y2={50} stroke={accent} strokeWidth={2} />
      <polygon points="136,44 146,50 136,56" fill={accent} />
      {dots(12, '#94A3B8', 185, 50, 22)}
      <text x={185} y={95} textAnchor="middle" fontSize={12} fill="#6B7280" fontFamily="Atkinson Hyperlegible">MNIST (grey)</text>
    </svg>
  )

  if (visual === 'matched') return (
    <svg width={240} height={120} viewBox="0 0 240 120">
      {[0,1,2,3,4].map(cls => (
        <circle key={cls} cx={60 + cls * 22} cy={30} r={10}
          fill={['#6366F1','#EC4899','#10B981','#F59E0B','#3B82F6'][cls]} />
      ))}
      <text x={120} y={60} textAnchor="middle" fontSize={12} fill="#6B7280" fontFamily="Atkinson Hyperlegible">digits 0-4 only</text>
      <line x1={120} y1={70} x2={120} y2={82} stroke="#10B981" strokeWidth={2} />
      <polygon points="114,78 120,88 126,78" fill="#10B981" />
      {[0,1,2,3,4].map(cls => (
        <circle key={cls} cx={60 + cls * 22} cy={102} r={8}
          fill={['#6366F1','#EC4899','#10B981','#F59E0B','#3B82F6'][cls]} opacity={0.7} />
      ))}
    </svg>
  )

  if (visual === 'mismatched') return (
    <svg width={240} height={130} viewBox="0 0 240 130">
      {[5,6,7,8,9].map((cls, i) => (
        <circle key={cls} cx={55 + i * 22} cy={22} r={10}
          fill={['#8B5CF6','#EF4444','#14B8A6','#F97316','#84CC16'][i]} />
      ))}
      <text x={120} y={52} textAnchor="middle" fontSize={12} fill="#6B7280" fontFamily="Atkinson Hyperlegible">5-9 (pretrain)</text>
      <text x={120} y={78} textAnchor="middle" fontSize={24}>⚡</text>
      {[0,1,2,3,4].map((cls, i) => (
        <circle key={cls} cx={55 + i * 22} cy={102} r={8}
          fill={['#6366F1','#EC4899','#10B981','#F59E0B','#3B82F6'][i]} opacity={0.7} />
      ))}
      <text x={120} y={124} textAnchor="middle" fontSize={12} fill="#6B7280" fontFamily="Atkinson Hyperlegible">0-4 (finetune)</text>
    </svg>
  )

  if (visual === 'ae') return (
    <svg width={260} height={110} viewBox="0 0 260 110">
      <rect x={20} y={20} width={60} height={40} rx={6} fill="#FDF2F8" stroke="#EC4899" strokeWidth={1.5} />
      <text x={50} y={44} textAnchor="middle" fontSize={12} fill="#EC4899" fontFamily="Atkinson Hyperlegible" fontWeight={700}>Encoder</text>
      <line x1={80} y1={40} x2={108} y2={40} stroke="#E5E5E5" strokeWidth={2} />
      <circle cx={118} cy={40} r={10} fill="#F3F4F6" stroke="#9CA3AF" strokeWidth={1.5} />
      <line x1={128} y1={40} x2={156} y2={40} stroke="#E5E5E5" strokeWidth={2} />
      <rect x={156} y={20} width={60} height={40} rx={6} fill="#FDF2F8" stroke="#EC4899" strokeWidth={1.5} />
      <text x={186} y={44} textAnchor="middle" fontSize={12} fill="#EC4899" fontFamily="Atkinson Hyperlegible" fontWeight={700}>Decoder</text>
      <text x={130} y={90} textAnchor="middle" fontSize={12} fill="#9CA3AF" fontFamily="Atkinson Hyperlegible">No class labels used</text>
    </svg>
  )

  if (visual === 'scarcity') return (
    <svg width={240} height={130} viewBox="0 0 240 130">
      {[0,1,2].map(j => (
        <g key={j}>
          <rect x={30 + j * 65} y={100 - [20, 42, 72][j]} width={45} height={[20, 42, 72][j]} rx={5}
            fill={accent} opacity={0.55 + j * 0.18} />
          <text x={52 + j * 65} y={120} textAnchor="middle" fontSize={13} fill="#374151" fontFamily="Atkinson Hyperlegible" fontWeight={700}>
            {['10%','25%','50%'][j]}
          </text>
        </g>
      ))}
    </svg>
  )

  if (visual === 'domain') return (
    <svg width={260} height={120} viewBox="0 0 260 120">
      <rect x={10} y={10} width={70} height={70} rx={8} fill="#FEF3C7" stroke="#F59E0B" strokeWidth={1.5} />
      <text x={45} y={50} textAnchor="middle" fontSize={32} dominantBaseline="middle">🐱</text>
      <text x={45} y={100} textAnchor="middle" fontSize={12} fill="#6B7280" fontFamily="Atkinson Hyperlegible" fontWeight={700}>CIFAR-10</text>
      <text x={130} y={50} textAnchor="middle" fontSize={24}>→</text>
      <rect x={180} y={10} width={70} height={70} rx={8} fill="#F3F4F6" stroke="#9CA3AF" strokeWidth={1.5} />
      <text x={215} y={50} textAnchor="middle" fontSize={32} dominantBaseline="middle">✍</text>
      <text x={215} y={100} textAnchor="middle" fontSize={12} fill="#6B7280" fontFamily="Atkinson Hyperlegible" fontWeight={700}>MNIST</text>
    </svg>
  )

  if (visual === 'aug') return (
    <svg width={260} height={120} viewBox="0 0 260 120">
      <circle cx={65} cy={55} r={34} fill="#FCD34D" opacity={0.4} />
      <circle cx={65} cy={55} r={24} fill="#F59E0B" opacity={0.6} />
      <circle cx={65} cy={55} r={14} fill="#D97706" opacity={0.8} />
      <text x={65} y={105} textAnchor="middle" fontSize={12} fill="#6B7280" fontFamily="Atkinson Hyperlegible" fontWeight={700}>SVHN + Jitter</text>
      <line x1={105} y1={55} x2={150} y2={55} stroke={accent} strokeWidth={2} />
      <polygon points="146,49 156,55 146,61" fill={accent} />
      {dots(10, '#94A3B8', 200, 55, 26)}
      <text x={200} y={105} textAnchor="middle" fontSize={12} fill="#6B7280" fontFamily="Atkinson Hyperlegible" fontWeight={700}>MNIST</text>
    </svg>
  )

  // format
  return (
    <svg width={260} height={120} viewBox="0 0 260 120">
      <rect x={10} y={10} width={70} height={70} rx={8} fill="#FFF7ED" stroke="#F97316" strokeWidth={1.5} />
      <text x={45} y={50} textAnchor="middle" fontSize={32} dominantBaseline="middle">👗</text>
      <text x={45} y={100} textAnchor="middle" fontSize={12} fill="#6B7280" fontFamily="Atkinson Hyperlegible" fontWeight={700}>FashionMNIST</text>
      <line x1={90} y1={50} x2={170} y2={50} stroke={accent} strokeWidth={2} />
      <text x={130} y={40} textAnchor="middle" fontSize={12} fill={accent} fontFamily="Atkinson Hyperlegible" fontWeight={700}>28×28 grey</text>
      <polygon points="166,44 176,50 166,56" fill={accent} />
      <rect x={180} y={10} width={70} height={70} rx={8} fill="#F3F4F6" stroke="#9CA3AF" strokeWidth={1.5} />
      <text x={215} y={50} textAnchor="middle" fontSize={32} dominantBaseline="middle">✍</text>
      <text x={215} y={100} textAnchor="middle" fontSize={12} fill="#6B7280" fontFamily="Atkinson Hyperlegible" fontWeight={700}>MNIST</text>
    </svg>
  )
}

export function Slide10() {
  const [selected, setSelected] = useState<string | null>(null)
  const selectedExp = experiments.find(e => e.id === selected)

  return (
    <Slide track="image" trackIndex={1} trackTotal={5}>
      <motion.h2
        initial={{ opacity: 0 }} animate={{ opacity: 1 }}
        className="font-serif-display"
        style={{ fontSize: 38, fontWeight: 600, color: '#171717', marginBottom: 6 }}>
        What are we testing? 9 Pretrain Configurations
      </motion.h2>
      <motion.p
        initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.1 }}
        style={{ fontFamily: 'Atkinson Hyperlegible', fontSize: 20, color: '#4B5563', marginBottom: 16 }}>
        Each varies one aspect of the pretrain → DP-finetune setup. <strong style={{ color: '#171717' }}>Click any card</strong> to see it visually. 198 total runs · 3 seeds each.
      </motion.p>

      <div style={{
        display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 14,
        flex: 1, minHeight: 0, alignContent: 'start',
      }}>
        {experiments.map((exp, i) => (
          <motion.button
            key={exp.id}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.15 + i * 0.04 }}
            onClick={() => setSelected(exp.id)}
            style={{
              textAlign: 'left', padding: '16px 18px', borderRadius: 12, cursor: 'pointer',
              background: '#FFFFFF',
              border: `1.5px solid ${exp.accent}40`,
              borderLeft: `4px solid ${exp.accent}`,
              boxShadow: '0 1px 4px rgba(0,0,0,0.05)',
              transition: 'transform 200ms, box-shadow 200ms',
              display: 'flex', flexDirection: 'column', gap: 8,
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.transform = 'translateY(-2px)'
              e.currentTarget.style.boxShadow = `0 6px 16px ${exp.accent}22`
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.transform = 'translateY(0)'
              e.currentTarget.style.boxShadow = '0 1px 4px rgba(0,0,0,0.05)'
            }}
          >
            <div className="flex items-center justify-between gap-2">
              <span style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 22, color: exp.accent, fontWeight: 700 }}>{exp.id}</span>
              <span style={{
                fontFamily: 'Atkinson Hyperlegible', fontSize: 20, fontWeight: 700,
                color: exp.tagColor, background: exp.tagBg,
                padding: '4px 10px', borderRadius: 20, letterSpacing: 0.3,
              }}>{exp.tag}</span>
            </div>
            <div style={{ fontFamily: 'Atkinson Hyperlegible', fontWeight: 700, fontSize: 22, color: '#171717' }}>
              {exp.label}
            </div>
            <div style={{ fontFamily: 'Atkinson Hyperlegible', fontSize: 22, color: '#9CA3AF', display: 'flex', alignItems: 'center', gap: 6 }}>
              <span>→ click for details</span>
            </div>
          </motion.button>
        ))}
      </div>

      {/* Privacy budget pill */}
      <motion.div
        initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.6 }}
        style={{
          marginTop: 10, padding: '8px 14px', background: '#EEF2FF', borderRadius: 10,
          border: '1px solid #C7D2FE', display: 'inline-flex', alignItems: 'center', gap: 12,
          alignSelf: 'flex-start',
        }}>
        <span style={{ fontFamily: 'Atkinson Hyperlegible', fontSize: 20, color: '#4F46E5', fontWeight: 700 }}>
          Privacy budget:
        </span>
        <TeX latex="\varepsilon \in \{0.5,1,2,4,8,\infty\}" size={0.9} />
      </motion.div>

      {/* Centered overlay modal */}
      <Modal
        open={!!selectedExp}
        onClose={() => setSelected(null)}
        accent={selectedExp?.accent ?? '#6366F1'}
        accentBg={selectedExp?.tagBg ?? '#EEF2FF'}
        title={selectedExp?.label ?? ''}
        tag={selectedExp?.tag}
        width={600}
      >
        {selectedExp && (
          <div className="flex flex-col gap-4">
            <div style={{
              background: selectedExp.tagBg, borderRadius: 12, padding: '14px 10px',
              display: 'flex', justifyContent: 'center',
            }}>
              <ExperimentVisual visual={selectedExp.visual} accent={selectedExp.accent} />
            </div>
            <p style={{
              fontFamily: 'Atkinson Hyperlegible', fontSize: 22, color: '#171717', lineHeight: 1.6,
            }}>
              {selectedExp.description}
            </p>
            <div style={{
              padding: '12px 16px', borderRadius: 10, background: selectedExp.tagBg,
              borderLeft: `4px solid ${selectedExp.accent}`,
            }}>
              <p style={{ fontFamily: 'Atkinson Hyperlegible', fontSize: 22, color: selectedExp.tagColor, lineHeight: 1.5 }}>
                <strong>Result:</strong> {selectedExp.result}
              </p>
            </div>
          </div>
        )}
      </Modal>
    </Slide>
  )
}
