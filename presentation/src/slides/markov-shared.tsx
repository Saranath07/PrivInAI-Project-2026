import { motion } from 'framer-motion'
import { Slide } from '../deck/Slide'
import { Math } from '../components/Math'

interface Props {
  trackIndex: number
  headline: string
  imgFile: string
  caption: string
  formula?: string
  formulaCaption?: string
}

export function MarkovSlide({ trackIndex, headline, imgFile, caption, formula, formulaCaption }: Props) {
  const base = import.meta.env.BASE_URL
  return (
    <Slide track="markov" trackIndex={trackIndex} trackTotal={5}>
      <motion.h2
        initial={{ opacity: 0 }} animate={{ opacity: 1 }}
        className="font-serif-display"
        style={{ fontSize: 26, fontWeight: 600, color: '#171717', marginBottom: 12 }}>
        {headline}
      </motion.h2>

      <motion.div
        initial={{ opacity: 0, scale: 0.98 }} animate={{ opacity: 1, scale: 1 }}
        transition={{ delay: 0.2, duration: 0.4 }}
        className="flex-1 flex flex-col items-center justify-center gap-3 min-h-0">
        <div className="relative">
          <img
            src={`${base}markov/${imgFile}`}
            alt={headline}
            style={{
              maxWidth: '100%', maxHeight: 360, objectFit: 'contain',
              border: '1px solid #E5E5E5', borderRadius: 6,
              boxShadow: '0 2px 12px rgba(0,0,0,0.06)',
            }}
          />
          {formula && (
            <div style={{
              position: 'absolute', bottom: 12, left: 12,
              background: 'rgba(255,255,255,0.96)', borderRadius: 6,
              padding: '8px 12px', boxShadow: '0 1px 6px rgba(0,0,0,0.1)',
              maxWidth: 480,
            }}>
              <Math latex={formula} block size={0.85} />
              {formulaCaption && (
                <p style={{ fontFamily: 'Atkinson Hyperlegible', fontSize: 10, color: '#6B7280', marginTop: 4 }}>
                  {formulaCaption}
                </p>
              )}
            </div>
          )}
        </div>
        <p style={{
          fontFamily: 'Atkinson Hyperlegible', fontStyle: 'italic', fontSize: 12,
          color: '#6B7280', textAlign: 'center', maxWidth: 700,
        }}>
          {caption}
        </p>
      </motion.div>
    </Slide>
  )
}
