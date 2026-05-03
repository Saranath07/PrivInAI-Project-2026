import { useEffect, useRef, useState } from 'react'
import type { Track } from './nav'
import { SEGMENT_DURATIONS, TRACK_COLORS } from './nav'

const TOTAL = 7 * 60
const TRACKS: Track[] = ['intro', 'image', 'markov', 'transformers', 'close']

function pad(n: number) { return String(n).padStart(2, '0') }

function formatTime(s: number) {
  const m = Math.floor(Math.abs(s) / 60)
  const sec = Math.abs(s) % 60
  return `${pad(m)}:${pad(sec)}`
}

function getTimerColor(elapsed: number) {
  if (elapsed >= TOTAL - 15) return '#EF4444'
  if (elapsed >= TOTAL - 60) return '#F59E0B'
  return '#6366F1'
}

interface Props {
  running: boolean
  onReset: () => void
  currentTrack: Track
}

export function Timer({ running, onReset, currentTrack }: Props) {
  const [elapsed, setElapsed] = useState(0)
  const startRef = useRef<number | null>(null)
  const baseRef = useRef(0)

  useEffect(() => {
    if (!running) {
      if (startRef.current !== null) {
        baseRef.current = elapsed
        startRef.current = null
      }
      return
    }
    startRef.current = performance.now()
    const id = setInterval(() => {
      const now = performance.now()
      setElapsed(Math.min(baseRef.current + Math.floor((now - startRef.current!) / 1000), TOTAL + 30))
    }, 250)
    return () => clearInterval(id)
  }, [running])

  useEffect(() => {
    setElapsed(0)
    baseRef.current = 0
    startRef.current = running ? performance.now() : null
  }, [onReset])

  const remaining = TOTAL - elapsed
  const color = getTimerColor(elapsed)

  // Segment progress bar
  let cumulative = 0
  const segments = TRACKS.map(t => {
    const dur = SEGMENT_DURATIONS[t]
    const start = cumulative
    cumulative += dur
    const end = cumulative
    const isActive = t === currentTrack
    const segElapsed = Math.max(0, Math.min(elapsed - start, dur))
    const pct = dur / TOTAL
    return { t, start, end, dur, pct, isActive, segElapsed }
  })

  return (
    <div className="fixed bottom-4 left-6 select-none z-50">
      <div
        className="font-mono-data text-sm px-3 py-1.5 rounded-full border"
        style={{
          color,
          borderColor: color + '40',
          background: '#FAFAF7',
          boxShadow: '0 1px 4px rgba(0,0,0,0.08)',
          letterSpacing: '0.04em',
        }}
      >
        {remaining < 0 ? '+' : ''}{formatTime(Math.abs(remaining))} / 7:00
      </div>
      {/* Segment bar */}
      <div className="flex mt-1 rounded-full overflow-hidden" style={{ width: 180, height: 3 }}>
        {segments.map(({ t, pct, isActive }) => (
          <div
            key={t}
            style={{
              width: `${pct * 100}%`,
              background: TRACK_COLORS[t],
              opacity: isActive ? 1 : 0.25,
              transition: 'opacity 300ms',
            }}
          />
        ))}
      </div>
    </div>
  )
}
