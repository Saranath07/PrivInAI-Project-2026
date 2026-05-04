export type Track = 'intro' | 'image' | 'markov' | 'transformers' | 'close'

export interface SlideEntry {
  id: string
  track: Track
  trackIndex?: number
  trackTotal?: number
}

export const SLIDES: SlideEntry[] = [
  { id: '00-title',             track: 'intro' },
  { id: '01-question',          track: 'intro' },
  { id: '10-image-setup',       track: 'image',       trackIndex: 1, trackTotal: 6 },
  { id: '11-image-epsilon',     track: 'image',       trackIndex: 2, trackTotal: 6 },
  { id: '11b-image-ranking',    track: 'image',       trackIndex: 3, trackTotal: 6 },
  { id: '12-image-mechanism',   track: 'image',       trackIndex: 4, trackTotal: 6 },
  { id: '13-image-divergences', track: 'image',       trackIndex: 5, trackTotal: 6 },
  { id: '15-image-probe',       track: 'image',       trackIndex: 6, trackTotal: 6 },
  { id: '20-markov-setup',      track: 'markov',      trackIndex: 1, trackTotal: 5 },
  { id: '21-markov-renyi',      track: 'markov',      trackIndex: 2, trackTotal: 5 },
  { id: '22-markov-sgd',        track: 'markov',      trackIndex: 3, trackTotal: 5 },
  { id: '23-markov-dpsgd',      track: 'markov',      trackIndex: 4, trackTotal: 5 },
  { id: '24-markov-insight',    track: 'markov',      trackIndex: 5, trackTotal: 5 },
  { id: '30-transformers',          track: 'transformers', trackIndex: 1, trackTotal: 4 },
  { id: '31-transformer-shift',     track: 'transformers', trackIndex: 2, trackTotal: 4 },
  { id: '32-transformer-ppl',       track: 'transformers', trackIndex: 3, trackTotal: 4 },
  { id: '33-transformer-analytics', track: 'transformers', trackIndex: 4, trackTotal: 4 },
  { id: '40-takeaways',         track: 'close' },
  { id: '42-references',        track: 'close' },
  { id: '41-thankyou',          track: 'close' },
]

export const TRACK_COLORS: Record<Track, string> = {
  intro:         '#6366F1',
  image:         '#6366F1',
  markov:        '#7C3AED',
  transformers:  '#0D9488',
  close:         '#6B7280',
}

export const TRACK_LABELS: Record<Track, string> = {
  intro:        'Intro',
  image:        'Image Track',
  markov:       'Markov Chains',
  transformers: 'Transformers',
  close:        'Closing',
}

export const SEGMENT_DURATIONS: Record<Track, number> = {
  intro:        30,
  image:        165,
  markov:       135,
  transformers: 60,
  close:        50,
}
