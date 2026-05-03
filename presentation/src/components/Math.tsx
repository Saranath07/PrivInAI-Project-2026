import katex from 'katex'

interface Props {
  latex: string
  block?: boolean
  size?: number
}

export function Math({ latex, block = false, size = 1 }: Props) {
  let html = ''
  try {
    html = katex.renderToString(latex, {
      displayMode: block,
      throwOnError: false,
      trust: true,
    })
  } catch {
    return <code style={{ color: '#EF4444' }}>{latex}</code>
  }

  return block
    ? <div
        style={{ fontSize: `${size}em` }}
        dangerouslySetInnerHTML={{ __html: html }}
      />
    : <span dangerouslySetInnerHTML={{ __html: html }} />
}
