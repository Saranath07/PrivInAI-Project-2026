import { useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import type { ReactNode } from 'react'

interface Props {
  open: boolean
  onClose: () => void
  accent?: string
  accentBg?: string
  title: ReactNode
  tag?: string
  children: ReactNode
  width?: number
}

export function Modal({ open, onClose, accent = '#6366F1', accentBg = '#EEF2FF', title, tag, children, width = 560 }: Props) {
  useEffect(() => {
    if (!open) return
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        e.stopPropagation()
        onClose()
      }
    }
    window.addEventListener('keydown', onKey, true)
    return () => window.removeEventListener('keydown', onKey, true)
  }, [open, onClose])

  return (
    <AnimatePresence>
      {open && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.18 }}
          onClick={(e) => { e.stopPropagation(); onClose() }}
          style={{
            position: 'absolute', inset: 0, zIndex: 50,
            background: 'rgba(17, 24, 39, 0.45)',
            backdropFilter: 'blur(4px)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
          }}>
          <motion.div
            initial={{ opacity: 0, scale: 0.94, y: 10 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.96, y: 6 }}
            transition={{ duration: 0.22, ease: 'easeOut' }}
            onClick={(e) => e.stopPropagation()}
            style={{
              width, maxWidth: '85%',
              background: '#FFFFFF',
              borderRadius: 16,
              border: `2px solid ${accent}`,
              boxShadow: `0 24px 60px rgba(0,0,0,0.25)`,
              padding: '22px 26px',
            }}>
            <div className="flex items-center justify-between" style={{ marginBottom: 14 }}>
              <div className="flex items-center gap-2">
                {tag && (
                  <span style={{
                    fontFamily: 'Atkinson Hyperlegible', fontSize: 11, fontWeight: 700,
                    color: accent, background: accentBg,
                    padding: '3px 10px', borderRadius: 20, letterSpacing: 0.3,
                  }}>{tag}</span>
                )}
                <h3 style={{
                  fontFamily: 'Crimson Pro, serif', fontSize: 22, fontWeight: 600,
                  color: '#171717',
                }}>{title}</h3>
              </div>
              <button
                onClick={onClose}
                aria-label="Close"
                style={{
                  width: 28, height: 28, borderRadius: 8,
                  background: '#F3F4F6', border: '1px solid #E5E5E5',
                  color: '#6B7280', fontSize: 16, lineHeight: 1,
                  cursor: 'pointer',
                }}>
                ×
              </button>
            </div>
            {children}
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  )
}
