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
  { name: 'Qiskit 1.1', role: 'Quantum circuits', color: '#8b5cf6' },
  { name: 'qiskit-ml', role: 'QSVC + kernel', color: '#7c3aed' },
  { name: 'scikit-learn', role: 'Classical SVM', color: '#3b82f6' },
  { name: 'Streamlit', role: 'Web app', color: '#22c55e' },
  { name: 'Vercel', role: 'Landing page', color: '#f59e0b' },
  { name: 'KDD Cup 99', role: 'Dataset', color: '#64748b' },
]

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

        {/* Pipeline diagram */}
        <FadeIn delay={0.1}>
          <div className="gradient-border rounded-2xl p-8 mt-10">
            <div className="overflow-x-auto">
              <pre className="font-mono text-xs text-slate-400 leading-6 whitespace-pre">{`
Raw network connection (KDD Cup 99)
        │
        ▼
┌───────────────────────────────────────────┐
│  Preprocessing                            │
│  encode → binarize → balance → scale      │
└──────────────┬────────────────────────────┘
               │
       ┌───────┴────────┐
       │                │
       ▼                ▼
 Classical path    Quantum path
 41 features       2 features
 RBF kernel        ZZFeatureMap
 K(x,y)=exp(…)    x → |φ(x)⟩ (2 qubits)
       │                │
       │      FidelityQuantumKernel
       │      K(x,y) = |⟨φ(x)|φ(y)⟩|²
       │                │
       ▼                ▼
 sklearn SVC      qiskit-ml QSVC
 trains on        trains on quantum
 kernel matrix    kernel matrix
       │                │
       └───────┬────────┘
               ▼
        Normal / Attack
              `.trim()}</pre>
            </div>
          </div>
        </FadeIn>

        {/* Tech stack pills */}
        <FadeIn delay={0.15}>
          <div className="mt-10">
            <p className="text-xs text-slate-500 font-semibold uppercase tracking-wider mb-5">Tech stack</p>
            <div className="flex flex-wrap gap-3">
              {stack.map(({ name, role, color }) => (
                <div key={name} className="gradient-border rounded-xl px-4 py-3 flex items-center gap-3">
                  <div className="w-2 h-2 rounded-full" style={{ backgroundColor: color }} />
                  <div>
                    <div className="text-sm font-semibold text-white">{name}</div>
                    <div className="text-xs text-slate-500">{role}</div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </FadeIn>

        {/* Repo structure */}
        <FadeIn delay={0.2}>
          <div className="mt-10 gradient-border rounded-2xl p-7">
            <p className="text-xs text-slate-500 font-semibold uppercase tracking-wider mb-5">Repository structure</p>
            <pre className="font-mono text-xs text-slate-400 leading-6">{`qsvm-anomaly-detection/
├── src/
│   ├── preprocess.py       load, encode, balance, scale
│   ├── classical_svm.py    RBF SVM baseline
│   ├── quantum_svm.py      ZZFeatureMap + QSVC
│   └── compare.py          full comparison runner
├── app.py                  Streamlit web app
├── precompute_results.py   cache quantum results
├── data/                   kddcup.data
├── results/
│   └── cached_results.json pre-computed quantum run
└── requirements.txt`}</pre>
          </div>
        </FadeIn>
      </div>
    </section>
  )
}
