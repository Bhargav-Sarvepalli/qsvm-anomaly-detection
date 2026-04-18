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

const steps = [
  {
    step: '01',
    title: 'Raw data → feature vector',
    body: 'Each network connection becomes a 41-dimensional numerical vector. Categorical fields (protocol, service) encoded as integers. All values scaled to [0, 1].',
    code: 'x = [0.00, 1.00, 0.03, 0.54, ...]',
  },
  {
    step: '02',
    title: 'Encode into quantum state',
    body: 'A ZZFeatureMap circuit encodes each value as a qubit rotation angle. CNOT gates entangle qubits, encoding feature interactions — the ZZ interaction term.',
    code: 'x → |φ(x)⟩  via ZZFeatureMap',
  },
  {
    step: '03',
    title: 'Compute quantum kernel',
    body: 'For every pair of training examples, the quantum kernel measures state overlap: K(x,y) = |⟨φ(x)|φ(y)⟩|². This is computed in 2ⁿ-dimensional Hilbert space.',
    code: 'K(x,y) = |⟨φ(x)|φ(y)⟩|²',
  },
  {
    step: '04',
    title: 'SVM trains on kernel matrix',
    body: 'A classical SVM optimizer finds the maximum-margin hyperplane using the quantum kernel matrix. Support vectors are identified. Decision boundary learned.',
    code: 'QSVC.fit(X_train, y_train)',
  },
]

export default function Solution() {
  return (
    <section id="solution" className="py-28 relative">
      {/* Section bg glow */}
      <div className="absolute inset-0 pointer-events-none">
        <div className="absolute bottom-0 right-0 w-96 h-96 rounded-full opacity-5"
          style={{ background: 'radial-gradient(circle, #7c3aed 0%, transparent 70%)' }} />
      </div>

      <div className="max-w-6xl mx-auto px-6">
        <FadeIn>
          <p className="text-purple-400 text-xs font-semibold tracking-widest uppercase mb-3">The solution</p>
          <h2 className="text-4xl font-bold text-white mb-6 max-w-2xl leading-tight">
            A quantum kernel that sees what classical methods can't.
          </h2>
          <p className="text-slate-400 text-lg max-w-2xl leading-relaxed">
            Classical SVM maps data into a high-dimensional space using a fixed kernel.
            A quantum kernel maps data into Hilbert space — 2ⁿ dimensions for n qubits.
            At 41 qubits, that's over 2 trillion dimensions. No classical computer can even represent this space.
          </p>
        </FadeIn>

        {/* Why quantum — two column */}
        <div className="mt-14 grid md:grid-cols-2 gap-5">
          <FadeIn delay={0.1}>
            <div className="gradient-border rounded-2xl p-7 h-full">
              <p className="text-xs text-slate-500 font-semibold uppercase tracking-wider mb-4">Classical SVM</p>
              <h3 className="text-white font-semibold text-lg mb-3">RBF kernel in implicit space</h3>
              <p className="text-slate-400 text-sm leading-relaxed">
                K(x,y) = exp(−γ‖x−y‖²). Powerful and well-understood, but
                the feature space is fixed. The kernel computes similarity
                analytically — fast but geometrically constrained.
              </p>
              <div className="mt-5 p-3 bg-surface rounded-lg">
                <code className="text-xs font-mono text-slate-400">
                  Feature space: R<sup>41</sup> → implicit R<sup>∞</sup>
                </code>
              </div>
            </div>
          </FadeIn>

          <FadeIn delay={0.15}>
            <div className="rounded-2xl p-7 h-full border border-purple-600/30"
              style={{ background: 'rgba(124, 58, 237, 0.06)' }}>
              <p className="text-xs text-purple-400 font-semibold uppercase tracking-wider mb-4">Quantum SVM</p>
              <h3 className="text-white font-semibold text-lg mb-3">ZZFeatureMap in Hilbert space</h3>
              <p className="text-slate-400 text-sm leading-relaxed">
                K(x,y) = |⟨φ(x)|φ(y)⟩|². The circuit encodes data into quantum
                states. Similarity is computed as circuit fidelity — a quantum
                operation that naturally operates in exponential-dimensional space.
              </p>
              <div className="mt-5 p-3 rounded-lg" style={{ background: 'rgba(124, 58, 237, 0.1)' }}>
                <code className="text-xs font-mono text-purple-300">
                  Feature space: R² → Hilbert space C⁴ (2 qubits)
                </code>
              </div>
            </div>
          </FadeIn>
        </div>

        {/* Pipeline steps */}
        <div className="mt-20">
          <FadeIn>
            <h3 className="text-white font-semibold text-xl mb-10">How it works — step by step</h3>
          </FadeIn>
          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-5">
            {steps.map(({ step, title, body, code }, i) => (
              <FadeIn key={step} delay={0.08 * i}>
                <div className="gradient-border rounded-xl p-6 h-full flex flex-col">
                  <span className="font-mono text-3xl font-bold text-purple-600/40 mb-4">{step}</span>
                  <h4 className="text-white font-semibold text-sm mb-2">{title}</h4>
                  <p className="text-slate-400 text-xs leading-relaxed flex-1">{body}</p>
                  <div className="mt-4 p-2.5 bg-surface rounded-lg">
                    <code className="font-mono text-xs text-purple-300">{code}</code>
                  </div>
                </div>
              </FadeIn>
            ))}
          </div>
        </div>

        {/* ZZFeatureMap circuit visualization */}
        <FadeIn delay={0.1}>
          <div className="mt-14 gradient-border rounded-2xl p-7">
            <p className="text-xs text-slate-500 font-semibold uppercase tracking-wider mb-5">
              ZZFeatureMap circuit — 2 qubits encoding x = [x₁, x₂]
            </p>
            <div className="overflow-x-auto">
              <pre className="font-mono text-sm text-slate-300 leading-7">{`
  q₀  |0⟩ ──[ H ]──[ Rz(x₁) ]────●────[ Rz(x₁·x₂) ]──── measure
                                   │
  q₁  |0⟩ ──[ H ]──[ Rz(x₂) ]────⊕────[ Rz(x₁·x₂) ]──── measure

  H        puts qubit in superposition
  Rz(xᵢ)  rotates by feature value  →  encodes your data
  CNOT     entangles qubits          →  creates feature correlation
  Rz(x·y)  encodes interaction term  →  the "ZZ" in ZZFeatureMap
              `.trim()}</pre>
            </div>
          </div>
        </FadeIn>
      </div>
    </section>
  )
}
