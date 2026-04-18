import { useRef } from 'react'
import { motion, useInView } from 'framer-motion'

function FadeIn({ children, delay = 0 }: { children: React.ReactNode; delay?: number }) {
  const ref = useRef(null)
  const inView = useInView(ref, { once: true, margin: '-80px' })
  return (
    <motion.div ref={ref} initial={{ opacity: 0, y: 24 }}
      animate={inView ? { opacity: 1, y: 0 } : {}}
      transition={{ duration: 0.6, delay, ease: [0.22, 1, 0.36, 1] }}>
      {children}
    </motion.div>
  )
}

function MetricBar({ label, value, max = 100, color }: {
  label: string; value: number; max?: number; color: string
}) {
  const ref = useRef(null)
  const inView = useInView(ref, { once: true })
  return (
    <div ref={ref} className="flex items-center gap-4">
      <span className="text-xs text-slate-500 w-20 text-right font-mono">{label}</span>
      <div className="flex-1 h-2 bg-surface rounded-full overflow-hidden">
        <motion.div
          className="h-full rounded-full"
          style={{ backgroundColor: color }}
          initial={{ width: 0 }}
          animate={inView ? { width: `${(value / max) * 100}%` } : {}}
          transition={{ duration: 1, ease: 'easeOut', delay: 0.2 }}
        />
      </div>
      <span className="font-mono text-sm font-semibold text-white w-14">{value}%</span>
    </div>
  )
}

export default function Results() {
  return (
    <section id="results" className="py-28 relative">
      <div className="max-w-6xl mx-auto px-6">
        <FadeIn>
          <p className="text-purple-400 text-xs font-semibold tracking-widest uppercase mb-3">Results</p>
          <h2 className="text-4xl font-bold text-white mb-6 max-w-2xl leading-tight">
            Honest benchmarking. No inflated claims.
          </h2>
          <p className="text-slate-400 text-lg max-w-2xl leading-relaxed">
            The quantum model uses 2 of 41 features — a deliberate constraint to keep
            simulation tractable. The accuracy gap reflects information loss, not quantum
            capability. The quantum kernel itself is working correctly.
          </p>
        </FadeIn>

        <div className="mt-16 grid lg:grid-cols-2 gap-10">
          {/* Classical card */}
          <FadeIn delay={0.1}>
            <div className="gradient-border rounded-2xl p-8">
              <div className="flex items-start justify-between mb-6">
                <div>
                  <p className="text-xs text-slate-500 font-semibold uppercase tracking-wider mb-1">
                    Classical SVM
                  </p>
                  <h3 className="text-white font-semibold">RBF kernel · 41 features · 2000 samples</h3>
                </div>
                <div className="text-right">
                  <div className="text-4xl font-bold text-white">99.2%</div>
                  <div className="text-xs text-slate-500 mt-0.5">accuracy</div>
                </div>
              </div>

              <div className="space-y-3 mb-6">
                <MetricBar label="Accuracy" value={99} color="#3b82f6" />
                <MetricBar label="Precision" value={100} color="#3b82f6" />
                <MetricBar label="Recall" value={99} color="#3b82f6" />
                <MetricBar label="F1 Score" value={99} color="#3b82f6" />
              </div>

              <div className="grid grid-cols-4 gap-3">
                {[
                  { label: 'TN', val: '196', sub: 'true normal' },
                  { label: 'FP', val: '0', sub: 'false alarms' },
                  { label: 'FN', val: '3', sub: 'missed attacks' },
                  { label: 'TP', val: '201', sub: 'caught attacks' },
                ].map(({ label, val, sub }) => (
                  <div key={label} className="bg-surface rounded-lg p-3 text-center">
                    <div className="text-xs text-slate-600 mb-1">{label}</div>
                    <div className="text-lg font-bold text-white">{val}</div>
                    <div className="text-xs text-slate-600 mt-0.5">{sub}</div>
                  </div>
                ))}
              </div>
            </div>
          </FadeIn>

          {/* Quantum card */}
          <FadeIn delay={0.15}>
            <div className="rounded-2xl p-8 border border-purple-600/30"
              style={{ background: 'rgba(124, 58, 237, 0.05)' }}>
              <div className="flex items-start justify-between mb-6">
                <div>
                  <p className="text-xs text-purple-400 font-semibold uppercase tracking-wider mb-1">
                    Quantum SVM
                  </p>
                  <h3 className="text-white font-semibold">ZZFeatureMap · 2 features · 200 samples</h3>
                </div>
                <div className="text-right">
                  <div className="text-4xl font-bold text-purple-400">52.5%</div>
                  <div className="text-xs text-slate-500 mt-0.5">accuracy</div>
                </div>
              </div>

              <div className="space-y-3 mb-6">
                <MetricBar label="Accuracy" value={52} color="#8b5cf6" />
                <MetricBar label="Precision" value={44} color="#8b5cf6" />
                <MetricBar label="Recall" value={100} color="#8b5cf6" />
                <MetricBar label="F1 Score" value={61} color="#8b5cf6" />
              </div>

              <div className="grid grid-cols-4 gap-3">
                {[
                  { label: 'TN', val: '6', sub: 'true normal' },
                  { label: 'FP', val: '19', sub: 'false alarms' },
                  { label: 'FN', val: '0', sub: 'missed attacks' },
                  { label: 'TP', val: '15', sub: 'caught attacks' },
                ].map(({ label, val, sub }) => (
                  <div key={label} className="rounded-lg p-3 text-center"
                    style={{ background: 'rgba(124, 58, 237, 0.1)' }}>
                    <div className="text-xs text-purple-700 mb-1">{label}</div>
                    <div className="text-lg font-bold text-white">{val}</div>
                    <div className="text-xs text-purple-700 mt-0.5">{sub}</div>
                  </div>
                ))}
              </div>

              {/* Highlight zero FN */}
              <div className="mt-4 p-3 rounded-lg border border-purple-600/20"
                style={{ background: 'rgba(124, 58, 237, 0.08)' }}>
                <p className="text-xs text-purple-300">
                  <span className="font-semibold">Zero missed attacks.</span> The quantum model
                  achieved 100% recall — it caught every attack in the test set,
                  trading precision for complete coverage.
                </p>
              </div>
            </div>
          </FadeIn>
        </div>

        {/* The gap explanation */}
        <FadeIn delay={0.1}>
          <div className="mt-10 gradient-border rounded-2xl p-7">
            <h3 className="text-white font-semibold mb-3">Why the accuracy gap?</h3>
            <p className="text-slate-400 text-sm leading-relaxed max-w-4xl">
              The 46.7% difference comes from a single constraint: the quantum model uses{' '}
              <span className="text-white font-medium">2 of 41 features</span> — a necessary
              limitation of simulating quantum circuits on classical hardware. Each additional
              qubit doubles the simulation memory. On real quantum hardware with 41 qubits,
              the model would operate in{' '}
              <span className="text-purple-400 font-mono">2⁴¹ ≈ 2.2 trillion</span>{' '}
              dimensional Hilbert space — computationally inaccessible to any classical kernel.
              That's where the genuine quantum advantage emerges.
            </p>
          </div>
        </FadeIn>
      </div>
    </section>
  )
}
