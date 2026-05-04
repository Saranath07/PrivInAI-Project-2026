import { motion } from 'framer-motion'
import { Slide } from '../deck/Slide'

interface Ref {
  key: string
  authors: string
  year: string
  title: string
  venue: string
  url?: string
  track: 'image' | 'markov' | 'transformers' | 'core'
}

const REFS: Ref[] = [
  {
    key: 'abadi16',
    authors: 'Abadi, M., Chu, A., Goodfellow, I., McMahan, H. B., Mironov, I., Talwar, K., Zhang, L.',
    year: '2016',
    title: 'Deep Learning with Differential Privacy',
    venue: 'ACM CCS',
    url: 'https://arxiv.org/abs/1607.00133',
    track: 'core',
  },
  {
    key: 'mironov17',
    authors: 'Mironov, I.',
    year: '2017',
    title: 'Rényi Differential Privacy',
    venue: 'IEEE CSF',
    url: 'https://arxiv.org/abs/1702.07476',
    track: 'markov',
  },
  {
    key: 'tramer21',
    authors: 'Tramèr, F., Boneh, D.',
    year: '2021',
    title: 'Differentially Private Learning Needs Better Features (Or Much More Data)',
    venue: 'ICLR',
    url: 'https://arxiv.org/abs/2011.11660',
    track: 'image',
  },
  {
    key: 'de22',
    authors: 'De, S., Berrada, L., Hayes, J., Smith, S. L., Balle, B.',
    year: '2022',
    title: 'Unlocking High-Accuracy Differentially Private Image Classification through Scale',
    venue: 'arXiv:2204.13650',
    url: 'https://arxiv.org/abs/2204.13650',
    track: 'image',
  },
  {
    key: 'yu22',
    authors: 'Yu, D., Naik, S., Backurs, A., Gopi, S., Inan, H. A., Kamath, G., et al.',
    year: '2022',
    title: 'Differentially Private Fine-tuning of Language Models',
    venue: 'ICLR',
    url: 'https://arxiv.org/abs/2110.06500',
    track: 'transformers',
  },
  {
    key: 'pillutla21',
    authors: 'Pillutla, K., Swayamdipta, S., Zellers, R., Thickstun, J., Welleck, S., Choi, Y., Harchaoui, Z.',
    year: '2021',
    title: 'MAUVE: Measuring the Gap Between Neural Text and Human Text using Divergence Frontiers',
    venue: 'NeurIPS',
    url: 'https://arxiv.org/abs/2102.01454',
    track: 'transformers',
  },
  {
    key: 'heusel17',
    authors: 'Heusel, M., Ramsauer, H., Unterthiner, T., Nessler, B., Hochreiter, S.',
    year: '2017',
    title: 'GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium (FID)',
    venue: 'NeurIPS',
    url: 'https://arxiv.org/abs/1706.08500',
    track: 'image',
  },
  {
    key: 'choquette21',
    authors: 'Choquette-Choo, C. A., Dvijotham, K., Pillutla, K., Ganesh, A., Steinke, T., Thakurta, A.',
    year: '2023',
    title: 'Correlated Noise Provably Beats Independent Noise for Differentially Private Learning',
    venue: 'ICLR',
    url: 'https://arxiv.org/abs/2310.06771',
    track: 'markov',
  },
  {
    key: 'hu22',
    authors: 'Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., et al.',
    year: '2022',
    title: 'LoRA: Low-Rank Adaptation of Large Language Models',
    venue: 'ICLR',
    url: 'https://arxiv.org/abs/2106.09685',
    track: 'transformers',
  },
  {
    key: 'radford19',
    authors: 'Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., Sutskever, I.',
    year: '2019',
    title: 'Language Models are Unsupervised Multitask Learners (GPT-2)',
    venue: 'OpenAI Technical Report',
    track: 'transformers',
  },
  {
    key: 'opacus',
    authors: 'Yousefpour, A., Shilov, I., Sablayrolles, A., Testuggine, D., Prasad, K., et al.',
    year: '2021',
    title: 'Opacus: User-Friendly Differential Privacy Library in PyTorch',
    venue: 'NeurIPS Privacy-Preserving ML Workshop',
    url: 'https://arxiv.org/abs/2109.12298',
    track: 'core',
  },
  {
    key: 'dwork14',
    authors: 'Dwork, C., Roth, A.',
    year: '2014',
    title: 'The Algorithmic Foundations of Differential Privacy',
    venue: 'Foundations and Trends in Theoretical CS',
    track: 'core',
  },
]

const TRACK_STYLE: Record<Ref['track'], { bg: string; text: string; label: string }> = {
  core:         { bg: '#F3F4F6', text: '#374151', label: 'CORE' },
  image:        { bg: '#EEF2FF', text: '#4F46E5', label: 'IMAGE' },
  markov:       { bg: '#F5F3FF', text: '#7C3AED', label: 'MARKOV' },
  transformers: { bg: '#F0FDFA', text: '#0D9488', label: 'TRANSFORMER' },
}

export function Slide42() {
  return (
    <Slide track="close">
      <motion.h2
        initial={{ opacity: 0 }} animate={{ opacity: 1 }}
        className="font-serif-display"
        style={{ fontSize: 34, fontWeight: 600, color: '#171717', marginBottom: 4 }}>
        References
      </motion.h2>
      <motion.p
        initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.1 }}
        style={{ fontFamily: 'Atkinson Hyperlegible', fontSize: 15, color: '#6B7280', marginBottom: 14 }}>
        Sources consulted across the three tracks. Links open in a new tab.
      </motion.p>

      <div style={{
        display: 'grid', gridTemplateColumns: '1fr 1fr',
        gap: 8, flex: 1, minHeight: 0, overflow: 'auto',
      }}>
        {REFS.map((r, i) => {
          const s = TRACK_STYLE[r.track]
          const content = (
            <>
              <div style={{
                display: 'flex', gap: 8, alignItems: 'baseline',
                marginBottom: 4,
              }}>
                <span style={{
                  fontFamily: 'Atkinson Hyperlegible', fontSize: 9, fontWeight: 700,
                  color: s.text, background: s.bg, padding: '2px 6px',
                  borderRadius: 3, letterSpacing: 0.4,
                }}>{s.label}</span>
                <span style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 10, color: '#9CA3AF' }}>
                  [{i + 1}] {r.year}
                </span>
              </div>
              <p style={{ fontFamily: 'Atkinson Hyperlegible', fontSize: 12, color: '#171717', fontWeight: 700, lineHeight: 1.3, marginBottom: 2 }}>
                {r.title}
              </p>
              <p style={{ fontFamily: 'Atkinson Hyperlegible', fontSize: 11, color: '#6B7280', lineHeight: 1.4, marginBottom: 2 }}>
                {r.authors}
              </p>
              <p style={{ fontFamily: 'Atkinson Hyperlegible', fontSize: 11, color: '#9CA3AF', fontStyle: 'italic' }}>
                {r.venue}
              </p>
            </>
          )

          return (
            <motion.div
              key={r.key}
              initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.15 + i * 0.03 }}>
              {r.url ? (
                <a
                  href={r.url} target="_blank" rel="noopener noreferrer"
                  style={{
                    display: 'block', padding: '8px 12px',
                    background: '#FFFFFF',
                    border: '1px solid #E5E7EB',
                    borderLeft: `3px solid ${s.text}`,
                    borderRadius: 6,
                    textDecoration: 'none',
                    transition: 'all 0.15s',
                  }}
                  onMouseOver={e => { (e.currentTarget as HTMLElement).style.background = '#F9FAFB' }}
                  onMouseOut={e => { (e.currentTarget as HTMLElement).style.background = '#FFFFFF' }}>
                  {content}
                </a>
              ) : (
                <div style={{
                  padding: '8px 12px',
                  background: '#FFFFFF',
                  border: '1px solid #E5E7EB',
                  borderLeft: `3px solid ${s.text}`,
                  borderRadius: 6,
                }}>
                  {content}
                </div>
              )}
            </motion.div>
          )
        })}
      </div>
    </Slide>
  )
}
