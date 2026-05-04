import { motion } from 'framer-motion'
import { Slide } from '../deck/Slide'

const fade = (delay: number) => ({
  initial: { opacity: 0, y: 10 },
  animate: { opacity: 1, y: 0 },
  transition: { duration: 0.5, delay },
})

export function Slide00() {
  return (
    <Slide track="intro">
      <div className="flex flex-col items-center justify-center h-full text-center gap-8"
        style={{ background: 'radial-gradient(ellipse at 30% 20%, rgba(99,102,241,0.06) 0%, transparent 60%)' }}>
        <div>
          <motion.h1 {...fade(0)} className="font-serif-display"
            style={{ fontSize: 128, fontWeight: 700, color: '#171717', lineHeight: 1.1, marginBottom: 16 }}>
            DP-ShiftBench
          </motion.h1>
          <motion.p {...fade(0.12)} style={{ fontSize: 36, color: '#404040', fontFamily: 'Crimson Pro, serif', maxWidth: 10000 }}>
            How Distribution Shift Governs the Value of<br />Pretraining for Private Learning
          </motion.p>
        </div>

        <motion.div {...fade(0.24)} style={{ width: 480, height: 1, background: '#E5E5E5' }} />

        <motion.div {...fade(0.32)} className="flex gap-12 justify-center">
          {[
            { name: 'Saranath P', dept: 'Data Science & AI', email: 'da25s014@smail.iitm.ac.in' },
            { name: 'Mayank Sharma', dept: 'Electrical Engineering', email: 'ee23b045@smail.iitm.ac.in' },
            { name: 'Aniruddh Krishna', dept: 'Naval Architecture', email: 'na23b022@smail.iitm.ac.in' },
          ].map(a => (
            <div key={a.name} className="flex flex-col items-center gap-0.5">
              <span style={{ fontFamily: 'Atkinson Hyperlegible', fontWeight: 700, fontSize: 24, color: '#171717' }}>
                {a.name}
              </span>
              <span style={{ fontFamily: 'Atkinson Hyperlegible', fontSize: 20, color: '#6B7280' }}>{a.dept}</span>
              <span style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 20, color: '#9CA3AF' }}>{a.email}</span>
            </div>
          ))}
        </motion.div>

        <motion.p {...fade(0.44)}
          style={{ fontFamily: 'Atkinson Hyperlegible', fontSize: 12, color: '#9CA3AF', position: 'absolute', bottom: 28 }}>
          DA5001 - Privacy in AI · IIT Madras · May 2026
        </motion.p>
      </div>
    </Slide>
  )
}
