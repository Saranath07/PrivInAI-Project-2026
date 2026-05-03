import { useEffect, useRef, useState } from 'react'
import type { CSSProperties } from 'react'

interface Props {
  to: number
  decimals?: number
  duration?: number
  prefix?: string
  suffix?: string
  className?: string
  style?: CSSProperties
}

export function CountUp({ to, decimals = 3, duration = 900, prefix = '', suffix = '', className, style }: Props) {
  const [val, setVal] = useState(0)
  const startRef = useRef<number | null>(null)
  const prefersReduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches

  useEffect(() => {
    if (prefersReduced) { setVal(to); return }
    startRef.current = null
    setVal(0)
    const id = requestAnimationFrame(function tick(ts) {
      if (!startRef.current) startRef.current = ts
      const progress = Math.min((ts - startRef.current) / duration, 1)
      const eased = progress < 1 ? 1 - Math.pow(1 - progress, 3) : 1
      setVal(to * eased)
      if (progress < 1) requestAnimationFrame(tick)
    })
    return () => cancelAnimationFrame(id)
  }, [to, duration])

  return (
    <span className={className} style={style}>
      {prefix}{val.toFixed(decimals)}{suffix}
    </span>
  )
}
