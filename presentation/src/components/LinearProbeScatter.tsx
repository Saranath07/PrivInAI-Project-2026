import { useState } from 'react'
import { motion } from 'framer-motion'
import { linearProbeData } from '../data/linearProbe'

const COLORS = [
  '#6366F1','#EC4899','#10B981','#F59E0B','#3B82F6',
  '#8B5CF6','#EF4444','#14B8A6','#F97316','#84CC16',
]

const SHAPES = ['circle', 'square', 'diamond', 'triangle'] as const
type Shape = typeof SHAPES[number]

function ShapeMarker({ shape, cx, cy, r, fill }: { shape: Shape; cx: number; cy: number; r: number; fill: string }) {
  if (shape === 'circle') return <circle cx={cx} cy={cy} r={r} fill={fill} />
  if (shape === 'square') {
    const s = r * 1.6
    return <rect x={cx - s / 2} y={cy - s / 2} width={s} height={s} fill={fill} />
  }
  if (shape === 'diamond') {
    return <polygon points={`${cx},${cy - r * 1.4} ${cx + r * 1.2},${cy} ${cx},${cy + r * 1.4} ${cx - r * 1.2},${cy}`} fill={fill} />
  }
  // triangle
  return <polygon points={`${cx},${cy - r * 1.4} ${cx + r * 1.2},${cy + r} ${cx - r * 1.2},${cy + r}`} fill={fill} />
}

// Simple linear regression
function linReg(points: { x: number; y: number }[]) {
  const n = points.length
  const meanX = points.reduce((s, p) => s + p.x, 0) / n
  const meanY = points.reduce((s, p) => s + p.y, 0) / n
  const num = points.reduce((s, p) => s + (p.x - meanX) * (p.y - meanY), 0)
  const den = points.reduce((s, p) => s + (p.x - meanX) ** 2, 0)
  const slope = den ? num / den : 0
  const intercept = meanY - slope * meanX
  return { slope, intercept }
}

interface Props { width?: number; height?: number }

export function LinearProbeScatter({ width = 460, height = 360 }: Props) {
  const [hovered, setHovered] = useState<number | null>(null)
  const padL = 52, padR = 20, padT = 16, padB = 44
  const W = width - padL - padR
  const H = height - padT - padB

  const xMin = 0.88, xMax = 1.00
  const yMin = 0.86, yMax = 1.00
  const toX = (v: number) => ((v - xMin) / (xMax - xMin)) * W
  const toY = (v: number) => H - ((v - yMin) / (yMax - yMin)) * H

  const points = linearProbeData.map(d => ({ x: d.lpAcc, y: d.dpAccEps05 }))
  const { slope, intercept } = linReg(points)
  const rx1 = xMin, ry1 = slope * rx1 + intercept
  const rx2 = xMax, ry2 = slope * rx2 + intercept

  const xTicks = [0.88, 0.90, 0.92, 0.94, 0.96, 0.98, 1.00]
  const yTicks = [0.86, 0.88, 0.90, 0.92, 0.94, 0.96, 0.98, 1.00]

  return (
    <div className="relative" style={{ width, height }}>
      <svg width={width} height={height}>
        <g transform={`translate(${padL},${padT})`}>
          {/* Gridlines */}
          {yTicks.map(t => <line key={t} x1={0} x2={W} y1={toY(t)} y2={toY(t)} stroke="#F3F4F6" />)}
          {xTicks.map(t => <line key={t} x1={toX(t)} x2={toX(t)} y1={0} y2={H} stroke="#F3F4F6" />)}

          {/* Regression line */}
          <motion.line
            x1={toX(rx1)} y1={toY(ry1)} x2={toX(rx2)} y2={toY(ry2)}
            stroke="#10B981" strokeWidth={2} strokeDasharray="6,3"
            initial={{ pathLength: 0, opacity: 0 }}
            animate={{ pathLength: 1, opacity: 0.8 }}
            transition={{ duration: 0.7, delay: 0.5 }}
          />

          {/* Points */}
          {linearProbeData.map((d, i) => {
            const cx = toX(d.lpAcc), cy = toY(d.dpAccEps05)
            const isHov = hovered === i
            const shape = SHAPES[i % SHAPES.length]
            const color = COLORS[i % COLORS.length]
            return (
              <g key={d.id}
                style={{ cursor: 'pointer' }}
                onMouseEnter={() => setHovered(i)}
                onMouseLeave={() => setHovered(null)}
              >
                <motion.g
                  initial={{ scale: 0, opacity: 0 }}
                  animate={{ scale: 1, opacity: 1 }}
                  transition={{ delay: i * 0.04, duration: 0.3 }}
                  style={{ transformOrigin: `${cx}px ${cy}px` }}
                >
                  <ShapeMarker shape={shape} cx={cx} cy={cy} r={isHov ? 7 : 5} fill={color} />
                </motion.g>
                {/* Label */}
                <text
                  x={cx + 8} y={cy - 5}
                  fontSize={9}
                  fill={color}
                  fontFamily="Atkinson Hyperlegible"
                  opacity={isHov ? 1 : 0.7}
                >
                  {d.label}
                </text>
              </g>
            )
          })}

          {/* Axes */}
          <line x1={0} x2={0} y1={0} y2={H} stroke="#E5E5E5" />
          <line x1={0} x2={W} y1={H} y2={H} stroke="#E5E5E5" />

          {/* Y ticks */}
          {yTicks.map(t => (
            <text key={t} x={-6} y={toY(t) + 4} textAnchor="end"
              fill="#9CA3AF" fontSize={9} fontFamily="JetBrains Mono">
              {(t * 100).toFixed(0)}%
            </text>
          ))}
          {/* X ticks */}
          {xTicks.map(t => (
            <text key={t} x={toX(t)} y={H + 18} textAnchor="middle"
              fill="#9CA3AF" fontSize={9} fontFamily="JetBrains Mono">
              {(t * 100).toFixed(0)}%
            </text>
          ))}

          {/* Axis labels */}
          <text x={W / 2} y={H + 36} textAnchor="middle" fill="#6B7280" fontSize={11} fontFamily="Atkinson Hyperlegible">
            Linear Probe Accuracy
          </text>
          <text x={-H / 2} y={-38} textAnchor="middle" fill="#6B7280" fontSize={11}
            fontFamily="Atkinson Hyperlegible" transform="rotate(-90)">
            DP Accuracy (ε=0.5)
          </text>
        </g>
      </svg>

      {/* Tooltip */}
      {hovered !== null && (() => {
        const d = linearProbeData[hovered]
        const cx = padL + ((d.lpAcc - xMin) / (xMax - xMin)) * (width - padL - padR)
        const cy = padT + (height - padT - padB) - ((d.dpAccEps05 - yMin) / (yMax - yMin)) * (height - padT - padB)
        return (
          <div
            className="absolute pointer-events-none text-xs rounded-lg px-2.5 py-2 shadow-lg"
            style={{
              left: cx + 12, top: cy - 40,
              background: '#1F2937', color: '#F9FAFB',
              fontFamily: 'Atkinson Hyperlegible',
              zIndex: 20, whiteSpace: 'nowrap',
            }}
          >
            <div className="font-bold">{d.label}</div>
            <div>LP acc: {(d.lpAcc * 100).toFixed(1)}%</div>
            <div>DP acc: {(d.dpAccEps05 * 100).toFixed(1)}%</div>
          </div>
        )
      })()}
    </div>
  )
}
