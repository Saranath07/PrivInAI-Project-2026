import type { ReactNode } from 'react'
import { motion } from 'framer-motion'
import type { Track } from './nav'
import { TRACK_COLORS, TRACK_LABELS } from './nav'

interface Props {
  children: ReactNode
  track: Track
  trackIndex?: number
  trackTotal?: number
  className?: string
}

const variants = {
  enter:  { opacity: 0, y: 16 },
  center: { opacity: 1, y: 0 },
  exit:   { opacity: 0, y: -8 },
}

export function Slide({ children, track, trackIndex, trackTotal, className = '' }: Props) {
  const color = TRACK_COLORS[track]
  const label = TRACK_LABELS[track]

  return (
    <motion.div
      variants={variants}
      initial="enter"
      animate="center"
      exit="exit"
      transition={{ duration: 0.35, ease: [0.0, 0.0, 0.2, 1] }}
      className={`absolute inset-0 flex flex-col ${className}`}
      style={{ background: '#FAFAF7', padding: '48px 72px' }}
    >
      {/* Track pill */}
      {track !== 'intro' && (
        <div
          className="absolute top-5 right-7 text-xs font-medium px-3 py-1 rounded-full"
          style={{
            background: color + '18',
            color,
            border: `1px solid ${color}30`,
            fontFamily: 'Atkinson Hyperlegible, sans-serif',
            letterSpacing: '0.02em',
          }}
        >
          {label}{trackIndex ? ` · ${trackIndex}/${trackTotal}` : ''}
        </div>
      )}

      {children}
    </motion.div>
  )
}
