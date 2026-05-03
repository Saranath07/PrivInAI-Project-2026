import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'

interface Props {
  latex: string
  block?: boolean
  size?: number
}

export function Math({ latex, block = false, size = 1 }: Props) {
  try {
    return block
      ? <div style={{ fontSize: `${size}em` }}><BlockMath math={latex} /></div>
      : <InlineMath math={latex} />
  } catch {
    return <code className="text-red-500">{latex}</code>
  }
}
