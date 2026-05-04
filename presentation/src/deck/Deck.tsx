import { useCallback, useEffect, useState } from 'react'
import { AnimatePresence } from 'framer-motion'
import { SLIDES } from './nav'

// Lazy-load slides
import { Slide00 } from '../slides/00-title'
import { Slide01 } from '../slides/01-question'
import { Slide10 } from '../slides/10-image-setup'
import { Slide11 } from '../slides/11-image-epsilon'
import { Slide11b } from '../slides/11b-image-ranking'
import { Slide12 } from '../slides/12-image-mechanism'
import { Slide13 } from '../slides/13-image-divergences'
import { Slide15 } from '../slides/15-image-probe'
import { Slide20 } from '../slides/20-markov-setup'
import { Slide21 } from '../slides/21-markov-renyi'
import { Slide22 } from '../slides/22-markov-sgd'
import { Slide23 } from '../slides/23-markov-dpsgd'
import { Slide24 } from '../slides/24-markov-insight'
import { Slide30 } from '../slides/30-transformers'
import { Slide40 } from '../slides/40-takeaways'
import { Slide41 } from '../slides/41-thankyou'

const SLIDE_COMPONENTS = [
  Slide00, Slide01,
  Slide10, Slide11, Slide11b, Slide12, Slide13, Slide15,
  Slide20, Slide21, Slide22, Slide23, Slide24,
  Slide30,
  Slide40, Slide41,
]

function getIndexFromHash(): number {
  const hash = window.location.hash.replace('#/', '')
  const idx = SLIDES.findIndex(s => s.id === hash)
  return idx >= 0 ? idx : 0
}

function setHash(idx: number) {
  history.replaceState(null, '', `#/${SLIDES[idx].id}`)
}

export function Deck() {
  const [index, setIndex] = useState(getIndexFromHash)

  const go = useCallback((next: number) => {
    const clamped = Math.max(0, Math.min(SLIDES.length - 1, next))
    setIndex(clamped)
    setHash(clamped)
  }, [])

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'ArrowRight' || e.key === ' ') { e.preventDefault(); go(index + 1) }
      if (e.key === 'ArrowLeft') { e.preventDefault(); go(index - 1) }
      if (e.key === 'Home') go(0)
      if (e.key === 'End') go(SLIDES.length - 1)
      if (e.key === 'f' || e.key === 'F') {
        if (!document.fullscreenElement) document.documentElement.requestFullscreen()
        else document.exitFullscreen()
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [index, go])

  const SlideComponent = SLIDE_COMPONENTS[index]
  const entry = SLIDES[index]

  return (
    <div
      style={{
        width: '100vw',
        height: '100vh',
        background: '#111',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        overflow: 'hidden',
      }}
    >
      {/* Full-viewport canvas - slides stretch to fill the screen */}
      <div
        style={{
          position: 'relative',
          width: '100vw',
          height: '100vh',
          overflow: 'hidden',
        }}
      >
        <AnimatePresence mode="wait">
          <SlideComponent key={entry.id} />
        </AnimatePresence>
      </div>
    </div>
  )
}
