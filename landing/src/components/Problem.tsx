import { useRef } from 'react'
import { motion, useInView } from 'framer-motion'

function FadeIn({ children, delay = 0 }: { children: React.ReactNode; delay?: number }) {
  const ref = useRef(null)
  const inView = useInView(ref, { once: true, margin: '-80px' })
  return (
    <motion.div
      ref={ref}
      initial={{ opacity: 0, y: 24 }}
      animate={inView ? { opacity: 1, y: 0 } : {}}
      transition={{ duration: 0.6, delay, ease: [0.22, 1, 0.36, 1] }}
    >
      {children}
    </motion.div>
  )
}

const attacks = [
  { label: 'smurf', count: 2807, pct: 88, color: '#ef4444' },
  { label: 'neptune', count: 1072, pct: 67, color: '#f97316' },
  { label: 'normal', count: 972, pct: 30, color: '#22c55e' },
  { label: 'back', count: 68, pct: 8, color: '#f59e0b' },
  { label: 'portsweep', count: 41, pct: 5, color: '#f59e0b' },
]

export default function Problem() {
  return (
    <section id="problem" className="py-28 relative">
      <div className="max-w-6xl mx-auto px-6">
        <FadeIn>
          <p className="text-purple-400 text-xs font-semibold tracking-widest uppercase mb-3">The problem</p>
          <h2 className="text-4xl font-bold text-white mb-6 max-w-2xl leading-tight">
            Federal networks process millions of connections. A fraction are attacks.
          </h2>
          <p className="text-slate-400 text-lg max-w-2xl leading-relaxed">
            Out of 494,021 network connections logged in the KDD Cup 99 dataset, 
            80% are attack traffic. Manual review is impossible. 
            A model that predicts "attack" for everything hits 80% accuracy — and catches nothing real.
          </p>
        </FadeIn>

        <div className="mt-16 grid lg:grid-cols-2 gap-12 items-center">
          {/* Visual — connection breakdown */}
          <FadeIn delay={0.1}>
            <div className="gradient-border rounded-2xl p-6">
              <p className="text-xs text-slate-500 font-medium uppercase tracking-wider mb-5">
                Connection breakdown — KDD Cup 99
              </p>
              <div className="space-y-3">
                {attacks.map(({ label, count, pct, color }) => (
                  <div key={label} className="flex items-center gap-3">
                    <span className="font-mono text-xs text-slate-500 w-20 text-right">{label}</span>
                    <div className="flex-1 h-5 bg-surface rounded-full overflow-hidden">
                      <motion.div
                        className="h-full rounded-full"
                        style={{ backgroundColor: color, opacity: 0.8 }}
                        initial={{ width: 0 }}
                        whileInView={{ width: `${pct}%` }}
                        transition={{ duration: 0.9, delay: 0.1, ease: 'easeOut' }}
                        viewport={{ once: true }}
                      />
                    </div>
                    <span className="font-mono text-xs text-slate-400 w-12">{count.toLocaleString()}</span>
                  </div>
                ))}
              </div>
              <p className="text-xs text-slate-600 mt-4">
                Sampled subset · green = normal · red/orange = attack variants
              </p>
            </div>
          </FadeIn>

          {/* Right — the challenge */}
          <div className="space-y-6">
            {[
              {
                title: 'Class imbalance',
                body: 'Raw dataset: 80% attacks, 20% normal. A trivial classifier wins on accuracy while being useless. We fix this with balanced sampling — equal classes in every train/test split.',
              },
              {
                title: 'Feature overlap',
                body: 'Attack and normal traffic share many statistical properties. Simple thresholds fail. The classifier needs to learn complex non-linear boundaries in 41-dimensional feature space.',
              },
              {
                title: 'The missed attack cost',
                body: 'In a real DoD deployment, a false negative (missed attack) costs far more than a false alarm. We optimise for recall over precision — the model must catch every attack.',
              },
            ].map(({ title, body }, i) => (
              <FadeIn key={title} delay={0.15 * i}>
                <div className="flex gap-4">
                  <div className="w-8 h-8 rounded-full border border-border flex items-center justify-center flex-shrink-0 mt-0.5">
                    <span className="text-xs text-slate-500">{i + 1}</span>
                  </div>
                  <div>
                    <h3 className="text-white font-semibold mb-1">{title}</h3>
                    <p className="text-slate-400 text-sm leading-relaxed">{body}</p>
                  </div>
                </div>
              </FadeIn>
            ))}
          </div>
        </div>
      </div>
    </section>
  )
}
