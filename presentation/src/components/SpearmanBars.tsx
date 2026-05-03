import { useState } from 'react'
import { motion } from 'framer-motion'
import { spearmanData } from '../data/spearman'
import type { DivRow } from '../data/divergences'
import { divergences } from '../data/divergences'
import { METRIC_DEFINITIONS } from '../data/metricDefinitions'
import { Modal } from './Modal'
import { Math as TeX } from './Math'

const METRIC_KEY_MAP: Record<string, keyof DivRow> = {
  fid: 'fid', kl_estimated: 'kl', jsd: 'jsd', total_variation: 'tv',
  mmd_rbf: 'mmd', proxy_a: 'proxyA', wasserstein: 'wasserstein',
  mauve: 'mauve', linear_probe: 'linearProbeAcc',
}

function barColor(metric: string, rho: number) {
  if (metric === 'linear_probe') return '#10B981'
  if (metric === 'wasserstein') return '#EF4444'
  if (rho < 0) return '#EC4899'
  return '#F59E0B'
}

function MiniScatter({ metric, color }: { metric: string; color: string }) {
  const key = METRIC_KEY_MAP[metric]
  if (!key) return null
  const points = divergences.map(d => ({
    x: d[key] as number,
    y: d.linearProbeAcc,
    label: d.shortLabel,
  }))
  const xs = points.map(p => p.x)
  const ys = points.map(p => p.y)
  const xMin = Math.min(...xs), xMax = Math.max(...xs)
  const yMin = Math.min(...ys) - 0.005, yMax = Math.max(...ys) + 0.005
  const W = 300, H = 180, pad = 36
  const toSX = (v: number) => pad + ((v - xMin) / (xMax - xMin || 1)) * (W - pad * 1.3)
  const toSY = (v: number) => H - pad - ((v - yMin) / (yMax - yMin || 1)) * (H - pad * 1.3)

  return (
    <svg width={W} height={H} style={{ overflow: 'visible' }}>
      <line x1={pad} x2={W - 8} y1={H - pad} y2={H - pad} stroke="#D1D5DB" />
      <line x1={pad} x2={pad} y1={8} y2={H - pad} stroke="#D1D5DB" />
      {points.map((p, i) => (
        <g key={i}>
          <circle cx={toSX(p.x)} cy={toSY(p.y)} r={6} fill={color} opacity={0.75} stroke="#FFFFFF" strokeWidth={1.5} />
          <title>{p.label}: ({p.x.toFixed(3)}, {p.y.toFixed(3)})</title>
        </g>
      ))}
      <text x={(W + pad) / 2} y={H - 8} textAnchor="middle" fontSize={11} fill="#6B7280" fontFamily="JetBrains Mono, monospace">
        metric value →
      </text>
      <text x={12} y={(H - pad) / 2} textAnchor="middle" fontSize={11} fill="#6B7280"
        fontFamily="JetBrains Mono, monospace" transform={`rotate(-90, 12, ${(H - pad) / 2})`}>
        DP accuracy →
      </text>
    </svg>
  )
}

export function SpearmanBars() {
  const [selected, setSelected] = useState<string | null>(null)
  const maxAbs = 1.0
  const barW = 340
  const rowH = 40
  const barH = 28

  const selectedDef = selected ? METRIC_DEFINITIONS[selected] : null
  const selectedSpearman = selected ? spearmanData.find(d => d.metric === selected) : null
  const color = selectedSpearman ? barColor(selectedSpearman.metric, selectedSpearman.rho) : '#6366F1'

  return (
    <>
      <div style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 4, flex: 1 }}>
          {spearmanData.map((d, i) => {
            const isPos = d.rho >= 0
            const absRho = Math.abs(d.rho)
            const pct = (absRho / maxAbs) * barW
            const barCol = barColor(d.metric, d.rho)

            return (
              <button
                key={d.metric}
                className="flex items-center gap-3 rounded px-1.5 text-left"
                style={{
                  transition: 'background 150ms',
                  cursor: 'pointer',
                  background: 'transparent',
                  height: rowH,
                }}
                onMouseEnter={(e) => (e.currentTarget.style.background = '#F9FAFB')}
                onMouseLeave={(e) => (e.currentTarget.style.background = 'transparent')}
                onClick={() => setSelected(d.metric)}
              >
                <span
                  style={{
                    width: 180, fontFamily: 'Atkinson Hyperlegible', color: '#171717',
                    flexShrink: 0, fontSize: 20, textAlign: 'right', fontWeight: 700,
                    textDecoration: 'underline', textDecorationColor: '#D1D5DB', textUnderlineOffset: 3,
                  }}
                >
                  {d.label}
                </span>
                <div className="relative" style={{ width: barW * 2 + 4, height: rowH }}>
                  <div
                    className="absolute top-0 bottom-0"
                    style={{ left: barW, width: 2, background: '#171717', opacity: 0.25 }}
                  />
                  <motion.div
                    className="absolute"
                    style={{
                      top: (rowH - barH) / 2, height: barH, borderRadius: 5,
                      background: barCol,
                      left: isPos ? barW + 2 : barW - pct,
                      width: 0,
                    }}
                    animate={{ width: pct }}
                    initial={{ width: 0 }}
                    transition={{ duration: 0.5, delay: i * 0.05, ease: 'easeOut' }}
                  />
                  <motion.span
                    style={{
                      position: 'absolute',
                      fontFamily: 'JetBrains Mono, monospace', color: barCol, fontWeight: 700,
                      fontSize: 17,
                      top: '50%', transform: 'translateY(-50%)',
                      left: isPos ? barW + pct + 10 : barW - pct - 68,
                    }}
                    initial={{ opacity: 0 }} animate={{ opacity: 1 }}
                    transition={{ delay: i * 0.05 + 0.4 }}
                  >
                    {d.rho.toFixed(3)}
                  </motion.span>
                </div>
              </button>
            )
          })}
        </div>
        <div className="flex" style={{ paddingLeft: 192, marginTop: 8 }}>
          <div className="text-center" style={{ width: barW, fontFamily: 'JetBrains Mono, monospace', color: '#6B7280', fontSize: 15, fontWeight: 700 }}>−1.0</div>
          <div className="text-center" style={{ width: 4 }}>0</div>
          <div className="text-center" style={{ width: barW, fontFamily: 'JetBrains Mono, monospace', color: '#6B7280', fontSize: 15, fontWeight: 700 }}>+1.0</div>
        </div>
      </div>

      <Modal
        open={!!selectedDef}
        onClose={() => setSelected(null)}
        accent={color}
        accentBg={`${color}18`}
        title={selectedDef?.name ?? ''}
        tag={selectedDef?.short}
        width={620}
      >
        {selectedDef && selectedSpearman && (
          <div className="flex flex-col gap-3">
            <div style={{
              background: '#FAFAF7', borderRadius: 10, padding: '14px 18px',
              border: '1px solid #E5E5E5',
            }}>
              <TeX latex={selectedDef.formula} block size={1} />
            </div>

            <p style={{ fontFamily: 'Atkinson Hyperlegible', fontSize: 14, color: '#171717', lineHeight: 1.6 }}>
              {selectedDef.definition}
            </p>

            <div className="flex gap-3 flex-wrap">
              <div style={{
                flex: 1, minWidth: 180, padding: '10px 14px',
                background: `${color}12`, borderRadius: 8,
                borderLeft: `3px solid ${color}`,
              }}>
                <p style={{ fontFamily: 'Atkinson Hyperlegible', fontSize: 11, color: '#6B7280', fontWeight: 700, letterSpacing: 0.3 }}>
                  INTUITION
                </p>
                <p style={{ fontFamily: 'Atkinson Hyperlegible', fontSize: 13, color: '#171717', marginTop: 2, lineHeight: 1.5 }}>
                  {selectedDef.intuition}
                </p>
              </div>
              <div style={{
                flex: 1, minWidth: 160, padding: '10px 14px',
                background: '#F3F4F6', borderRadius: 8,
                borderLeft: '3px solid #9CA3AF',
              }}>
                <p style={{ fontFamily: 'Atkinson Hyperlegible', fontSize: 11, color: '#6B7280', fontWeight: 700, letterSpacing: 0.3 }}>
                  RANGE
                </p>
                <p style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 12, color: '#171717', marginTop: 4 }}>
                  {selectedDef.range}
                </p>
              </div>
            </div>

            <div className="flex gap-4 items-center" style={{
              padding: '12px 16px', background: '#FFFFFF', border: '1px solid #E5E5E5', borderRadius: 10,
            }}>
              <div style={{ flexShrink: 0 }}>
                <p style={{ fontFamily: 'Atkinson Hyperlegible', fontSize: 11, color: '#6B7280', fontWeight: 700, letterSpacing: 0.3 }}>
                  SPEARMAN ρ
                </p>
                <p style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 28, color, fontWeight: 700 }}>
                  {selectedSpearman.rho >= 0 ? '+' : ''}{selectedSpearman.rho.toFixed(3)}
                </p>
                <p style={{ fontFamily: 'Atkinson Hyperlegible', fontSize: 11, color: '#6B7280' }}>
                  vs DP accuracy at ε = 0.5
                </p>
              </div>
              <div style={{ flex: 1 }}>
                <MiniScatter metric={selected!} color={color} />
              </div>
            </div>
          </div>
        )}
      </Modal>
    </>
  )
}
