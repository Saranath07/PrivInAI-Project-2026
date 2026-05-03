import { useState, useId } from 'react'
import { motion } from 'framer-motion'
import type { ResultRow } from '../data/results'
import { results, PALETTE, getAccForEpsilon } from '../data/results'

const EPS_VALUES = [0.5, 1, 2, 4, 8, Infinity] as const
const EPS_LABELS = ['0.5', '1', '2', '4', '8', '∞']
const DASH_PATTERNS = ['', '6,3', '3,3', '8,2,2,2', '4,2', '2,2']

interface Props {
  highlightEps?: number | null
  width?: number
  height?: number
  hideLegend?: boolean
}

export function AccuracyVsEpsilonChart({ highlightEps, width = 760, height = 360, hideLegend = false }: Props) {
  const [hovered, setHovered] = useState<{ row: ResultRow; eps: number; x: number; y: number } | null>(null)
  const [activeExperiment, setActiveExperiment] = useState<string | null>(null)
  const clipId = useId()

  const padL = 52, padR = 16, padT = 12, padB = 40
  const w = width - padL - padR
  const h = height - padT - padB

  const xPositions = EPS_VALUES.map((_, i) => (i / (EPS_VALUES.length - 1)) * w)
  const yMin = 0.86, yMax = 1.00
  const toY = (acc: number) => h - ((acc - yMin) / (yMax - yMin)) * h
  const yTicks = [0.86, 0.88, 0.90, 0.92, 0.94, 0.96, 0.98, 1.00]

  return (
    <div className="relative" style={{ width, height }}>
      <svg width={width} height={height} style={{ overflow: 'visible' }}>
        <defs>
          <clipPath id={clipId}>
            <rect x={padL} y={padT} width={w} height={h} />
          </clipPath>
        </defs>
        <g transform={`translate(${padL},${padT})`}>
          {/* Gridlines */}
          {yTicks.map(t => (
            <line key={t} x1={0} x2={w} y1={toY(t)} y2={toY(t)} stroke="#F3F4F6" strokeWidth={1} />
          ))}
          {/* ε highlight band */}
          {highlightEps !== null && highlightEps !== undefined && (() => {
            const idx = EPS_VALUES.findIndex(e => e === highlightEps)
            if (idx < 0) return null
            const x = xPositions[idx]
            return <rect x={x - 14} width={28} y={0} height={h} fill="#6366F118" rx={4} />
          })()}
          {/* Lines */}
          {results.map((row, ri) => {
            const points = EPS_VALUES.map((eps, i) => {
              const { mean } = getAccForEpsilon(row, eps)
              return `${xPositions[i]},${toY(mean)}`
            }).join(' ')
            const isActive = activeExperiment === row.experiment
            const isOther = activeExperiment !== null && !isActive
            return (
              <motion.polyline
                key={row.experiment}
                points={points}
                fill="none"
                stroke={PALETTE[ri % PALETTE.length]}
                strokeWidth={isActive ? 2.5 : isOther ? 1 : 2}
                strokeDasharray={DASH_PATTERNS[ri % DASH_PATTERNS.length]}
                opacity={isOther ? 0.3 : 1}
                strokeLinecap="round"
                strokeLinejoin="round"
                style={{ cursor: 'pointer', transition: 'opacity 200ms, stroke-width 200ms' }}
                onClick={() => setActiveExperiment(prev => prev === row.experiment ? null : row.experiment)}
                initial={{ pathLength: 0, opacity: 0 }}
                animate={{ pathLength: 1, opacity: isOther ? 0.3 : 1 }}
                transition={{ duration: 0.6, delay: ri * 0.04, ease: 'easeOut' }}
              />
            )
          })}
          {/* Hover dots */}
          {results.map((row, ri) => (
            EPS_VALUES.map((eps, i) => {
              const { mean } = getAccForEpsilon(row, eps)
              return (
                <circle
                  key={`${ri}-${i}`}
                  cx={xPositions[i]} cy={toY(mean)} r={5}
                  fill={PALETTE[ri % PALETTE.length]}
                  opacity={0}
                  style={{ cursor: 'pointer' }}
                  onMouseEnter={() => setHovered({ row, eps: EPS_VALUES[i], x: xPositions[i] + padL, y: toY(mean) + padT })}
                  onMouseLeave={() => setHovered(null)}
                />
              )
            })
          ))}
          {/* Y-axis labels */}
          {yTicks.map(t => (
            <text key={t} x={-8} y={toY(t) + 4} textAnchor="end"
              fill="#6B7280" fontSize={12} fontFamily="JetBrains Mono, monospace">
              {(t * 100).toFixed(0)}%
            </text>
          ))}
          {/* X-axis labels */}
          {EPS_LABELS.map((lbl, i) => (
            <text key={lbl} x={xPositions[i]} y={h + 22} textAnchor="middle"
              fill="#6B7280" fontSize={12} fontFamily="JetBrains Mono, monospace" fontWeight={600}>
              ε={lbl}
            </text>
          ))}
          {/* Axes */}
          <line x1={0} x2={0} y1={0} y2={h} stroke="#E5E5E5" strokeWidth={1} />
          <line x1={0} x2={w} y1={h} y2={h} stroke="#E5E5E5" strokeWidth={1} />
        </g>
      </svg>

      {/* Tooltip */}
      {hovered && (
        <div
          className="absolute pointer-events-none text-xs rounded-lg px-2.5 py-2 shadow-lg"
          style={{
            left: hovered.x + 10, top: hovered.y - 36,
            background: '#1F2937', color: '#F9FAFB',
            fontFamily: 'Atkinson Hyperlegible, sans-serif',
            whiteSpace: 'nowrap', zIndex: 20,
          }}
        >
          <div className="font-bold">{hovered.row.shortLabel}</div>
          <div>ε = {hovered.eps === Infinity ? '∞' : hovered.eps} · acc = {getAccForEpsilon(hovered.row, hovered.eps as any).mean.toFixed(3)}</div>
        </div>
      )}

      {/* Legend */}
      {!hideLegend && (
      <div className="flex flex-wrap gap-x-3 gap-y-1 mt-2 px-1" style={{ maxWidth: width, fontSize: 11 }}>
        {results.map((row, ri) => (
          <button
            key={row.experiment}
            className="flex items-center gap-1.5 text-xs rounded px-1"
            style={{
              fontFamily: 'Atkinson Hyperlegible, sans-serif',
              color: activeExperiment === row.experiment ? PALETTE[ri % PALETTE.length] : '#6B7280',
              opacity: activeExperiment && activeExperiment !== row.experiment ? 0.4 : 1,
              transition: 'opacity 200ms',
              cursor: 'pointer',
            }}
            onClick={() => setActiveExperiment(prev => prev === row.experiment ? null : row.experiment)}
          >
            <svg width={20} height={8}>
              <line x1={0} x2={20} y1={4} y2={4}
                stroke={PALETTE[ri % PALETTE.length]} strokeWidth={2}
                strokeDasharray={DASH_PATTERNS[ri % DASH_PATTERNS.length]} />
            </svg>
            {row.shortLabel}
          </button>
        ))}
      </div>
      )}
    </div>
  )
}
