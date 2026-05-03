import { motion } from 'framer-motion'
import { Slide } from '../deck/Slide'

export function Slide41() {
  return (
    <Slide track="close">
      <div className="flex flex-col items-center justify-center h-full gap-5 text-center"
        style={{ background: 'radial-gradient(ellipse at 70% 80%, rgba(99,102,241,0.05) 0%, transparent 60%)' }}>
        <motion.h1
          initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.6 }}
          className="font-serif-display"
          style={{ fontSize: 68, fontWeight: 400, color: '#171717' }}>
          Thank You
        </motion.h1>

        <motion.p
          initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.25 }}
          style={{ fontFamily: 'Atkinson Hyperlegible', fontSize: 18, color: '#6B7280' }}>
          DP-ShiftBench — DA5001, Privacy in AI, IIT Madras
        </motion.p>

        <motion.p
          initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.35 }}
          style={{ fontFamily: 'Atkinson Hyperlegible', fontSize: 15, color: '#404040' }}>
          Saranath P · Mayank Sharma · Aniruddh Krishna
        </motion.p>

        <motion.p
          initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.5 }}
          className="font-serif-display"
          style={{ fontSize: 20, color: '#9CA3AF', fontStyle: 'italic', marginTop: 8 }}>
          Questions welcome
        </motion.p>

        <motion.a
          href="https://github.com/Saranath07/PrivInAI-Project-2026"
          target="_blank"
          initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.65 }}
          style={{
            position: 'absolute', bottom: 28, right: 72,
            fontFamily: 'JetBrains Mono, monospace', fontSize: 12,
            color: '#6366F1', textDecoration: 'none',
          }}>
          github.com/Saranath07/PrivInAI-Project-2026
        </motion.a>
      </div>
    </Slide>
  )
}
