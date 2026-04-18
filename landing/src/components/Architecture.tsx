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

const stack = [
  { name: 'Qiskit 1.1',    role: 'Quantum circuits',   color: '#8b5cf6' },
  { name: 'qiskit-ml',     role: 'QSVC + kernel',       color: '#7c3aed' },
  { name: 'scikit-learn',  role: 'Classical SVM',       color: '#3b82f6' },
  { name: 'Streamlit',     role: 'Web app',             color: '#22c55e' },
  { name: 'Vercel',        role: 'Landing page',        color: '#f59e0b' },
  { name: 'KDD Cup 99',    role: 'Dataset',             color: '#64748b' },
]

function PipelineDiagram() {
  const box = "rounded-lg border border-border px-4 py-2.5 text-xs text-slate-300 text-center whitespace-nowrap"
  const label = "text-xs text-slate-500 text-center"
  const arrow = "flex justify-center my-2"
  const arrowSvg = (
    <svg width="12" height="20" viewBox="0 0 12 20" fill="none">
      <path d="M6 0v16M1 11l5 7 5-7" stroke="#3f3f46" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
    </svg>
  )
  const arrowH = (
    <div className="flex items-center px-2">
      <svg width="32" height="12" viewBox="0 0 32 12" fill="none">
        <path d="M0 6h28M23 1l7 5-7 5" stroke="#3f3f46" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
      </svg>
    </div>
  )

  return (
    <div className="gradient-border rounded-2xl p-8">
      {/* Row 1 — input */}
      <div className="flex justify-center">
        <div className={`${box} border-slate-700`}>
          Raw network connection — KDD Cup 99
        </div>
      </div>

      <div className={arrow}>{arrowSvg}</div>

      {/* Row 2 — preprocessing */}
      <div className="flex justify-center">
        <div className={`${box} border-slate-600 bg-surface`}>
          Preprocessing — encode · binarize · balance · MinMax scale
        </div>
      </div>

      <div className={arrow}>{arrowSvg}</div>

      {/* Row 3 — split */}
      <div className="flex justify-center gap-8 items-start">

        {/* Classical branch */}
        <div className="flex flex-col items-center gap-2 flex-1">
          <div className={`${box} border-blue-800 bg-blue-950/40 w-full`}>41 features</div>
          <div className={arrow}>{arrowSvg}</div>
          <div className={`${box} border-blue-700 bg-blue-950/30 w-full`}>RBF kernel<br />K(x,y) = exp(−γ‖x−y‖²)</div>
          <div className={arrow}>{arrowSvg}</div>
          <div className={`${box} border-blue-600 w-full`}>sklearn SVC</div>
          <p className={label}>Classical path</p>
        </div>

        {/* Divider */}
        <div className="flex items-center self-center pt-4">
          <div className="w-px h-24 bg-border" />
        </div>

        {/* Quantum branch */}
        <div className="flex flex-col items-center gap-2 flex-1">
          <div className={`${box} border-purple-800 bg-purple-950/40 w-full`}>2 features</div>
          <div className={arrow}>{arrowSvg}</div>
          <div className={`${box} border-purple-700 bg-purple-950/30 w-full`}>ZZFeatureMap<br />x → |φ(x)⟩ (2 qubits)</div>
          <div className={arrow}>{arrowSvg}</div>
          <div className={`${box} border-purple-700 bg-purple-950/30 w-full`}>FidelityQuantumKernel<br />K(x,y) = |⟨φ(x)|φ(y)⟩|²</div>
          <div className={arrow}>{arrowSvg}</div>
          <div className={`${box} border-purple-600 w-full`}>qiskit-ml QSVC</div>
          <p className={label}>Quantum path</p>
        </div>
      </div>

      <div className={arrow}>{arrowSvg}</div>

      {/* Output */}
      <div className="flex justify-center">
        <div className="rounded-lg border border-purple-600/50 bg-purple-900/20 px-6 py-2.5 text-sm text-purple-300 font-semibold">
          Normal / Attack classification
        </div>
      </div>
    </div>
  )
}

export default function Architecture() {
  return (
    <section id="architecture" className="py-28 relative">
      <div className="max-w-6xl mx-auto px-6">
        <FadeIn>
          <p className="text-purple-400 text-xs font-semibold tracking-widest uppercase mb-3">Architecture</p>
          <h2 className="text-4xl font-bold text-white mb-6 max-w-2xl leading-tight">
            Full pipeline — raw data to classification.
          </h2>
        </FadeIn>

        <FadeIn delay={0.1}>
          <PipelineDiagram />
        </FadeIn>

        {/* ZZFeatureMap */}
        <div className="mt-10 grid md:grid-cols-2 gap-6">
          <FadeIn delay={0.12}>
            <div className="gradient-border rounded-2xl p-7">
              <p className="text-xs text-slate-500 font-semibold uppercase tracking-wider mb-5">
                ZZFeatureMap circuit
              </p>
              <div className="space-y-3">
                {[
                  { label: 'q₀ |0⟩', gates: ['H', 'Rz(x₁)', 'CNOT', 'Rz(x₁·x₂)', 'M'] },
                  { label: 'q₁ |0⟩', gates: ['H', 'Rz(x₂)', 'CNOT', 'Rz(x₁·x₂)', 'M'] },
                ].map(({ label, gates }) => (
                  <div key={label} className="flex items-center gap-2">
                    <span className="font-mono text-xs text-slate-500 w-16 flex-shrink-0">{label}</span>
                    <div className="flex-1 h-px bg-border" />
                    {gates.map((g, i) => (
                      <div key={i} className="flex items-center gap-1">
                        <div className={`
                          px-2 py-1 rounded text-xs font-mono border flex-shrink-0
                          ${g === 'H' ? 'border-blue-700 text-blue-300 bg-blue-950/40' :
                            g.startsWith('Rz') ? 'border-purple-700 text-purple-300 bg-purple-950/40' :
                            g === 'CNOT' ? 'border-green-700 text-green-300 bg-green-950/40' :
                            'border-amber-700 text-amber-300 bg-amber-950/40'}
                        `}>{g}</div>
                        {i < gates.length - 1 && <div className="w-3 h-px bg-border flex-shrink-0" />}
                      </div>
                    ))}
                  </div>
                ))}
              </div>
              <div className="mt-5 grid grid-cols-2 gap-2 text-xs text-slate-500">
                <div><span className="text-blue-400">H</span> — superposition</div>
                <div><span className="text-purple-400">Rz(xᵢ)</span> — encodes feature</div>
                <div><span className="text-green-400">CNOT</span> — entangles qubits</div>
                <div><span className="text-amber-400">M</span> — measurement</div>
              </div>
            </div>
          </FadeIn>

          <FadeIn delay={0.15}>
            <div className="gradient-border rounded-2xl p-7">
              <p className="text-xs text-slate-500 font-semibold uppercase tracking-wider mb-5">
                Repository structure
              </p>
              <div className="space-y-1.5 font-mono text-xs">
                {[
                  { indent: 0, name: 'qsvm-anomaly-detection/', color: 'text-white' },
                  { indent: 1, name: 'landing/', color: 'text-blue-400' },
                  { indent: 2, name: 'src/components/', color: 'text-slate-400' },
                  { indent: 1, name: 'src/', color: 'text-purple-400' },
                  { indent: 2, name: 'preprocess.py', color: 'text-slate-400', note: 'load · encode · scale' },
                  { indent: 2, name: 'classical_svm.py', color: 'text-slate-400', note: 'RBF baseline' },
                  { indent: 2, name: 'quantum_svm.py', color: 'text-slate-400', note: 'ZZFeatureMap + QSVC' },
                  { indent: 2, name: 'compare.py', color: 'text-slate-400', note: 'full runner' },
                  { indent: 1, name: 'app.py', color: 'text-green-400', note: 'Streamlit app' },
                  { indent: 1, name: 'results/cached_results.json', color: 'text-amber-400', note: 'quantum cache' },
                  { indent: 1, name: 'requirements.txt', color: 'text-slate-500' },
                ].map(({ indent, name, color, note }, i) => (
                  <div key={i} className="flex items-center gap-2"
                    style={{ paddingLeft: `${indent * 14}px` }}>
                    {indent > 0 && <span className="text-border">├─</span>}
                    <span className={color}>{name}</span>
                    {note && <span className="text-slate-600 text-xs">{note}</span>}
                  </div>
                ))}
              </div>
            </div>
          </FadeIn>
        </div>

        {/* Tech stack */}
        <FadeIn delay={0.2}>
          <div className="mt-8">
            <p className="text-xs text-slate-500 font-semibold uppercase tracking-wider mb-4">Tech stack</p>
            <div className="flex flex-wrap gap-3">
              {stack.map(({ name, role, color }) => (
                <div key={name} className="gradient-border rounded-xl px-4 py-3 flex items-center gap-3">
                  <div className="w-2 h-2 rounded-full flex-shrink-0" style={{ backgroundColor: color }} />
                  <div>
                    <div className="text-sm font-semibold text-white">{name}</div>
                    <div className="text-xs text-slate-500">{role}</div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </FadeIn>
      </div>
    </section>
  )
}