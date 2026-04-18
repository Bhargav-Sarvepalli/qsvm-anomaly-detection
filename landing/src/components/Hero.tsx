import { Suspense } from 'react'
import { motion } from 'framer-motion'
import { lazy } from 'react'

const QuantumOrb = lazy(() => import('./QuantumOrb'))

const STREAMLIT_URL = 'https://qsvm-anomaly-detection.streamlit.app'

function fadeUp(delay = 0) {
  return {
    initial: { opacity: 0, y: 28 },
    animate: { opacity: 1, y: 0 },
    transition: { duration: 0.7, delay, ease: [0.22, 1, 0.36, 1] as const },
  }
}

export default function Hero() {
  return (
    <section className="relative min-h-screen flex items-center overflow-hidden">
      <div className="absolute inset-0 pointer-events-none">
        <div
          className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[600px] rounded-full opacity-10"
          style={{ background: 'radial-gradient(circle, #7c3aed 0%, transparent 70%)' }}
        />
        <div
          className="absolute top-1/3 right-1/4 w-[300px] h-[300px] rounded-full opacity-5"
          style={{ background: 'radial-gradient(circle, #3b82f6 0%, transparent 70%)' }}
        />
      </div>

      <div className="max-w-6xl mx-auto px-6 pt-24 pb-16 grid lg:grid-cols-2 gap-16 items-center w-full">
        <div>
          <motion.h1 {...fadeUp(0.1)} className="text-5xl lg:text-6xl font-bold text-white leading-[1.05] tracking-tight mb-6">
            Network<br />
            intrusion detection<br />
            <span className="text-transparent bg-clip-text" style={{
              backgroundImage: 'linear-gradient(135deg, #a78bfa, #60a5fa)'
            }}>
              powered by quantum.
            </span>
          </motion.h1>

          <motion.p {...fadeUp(0.2)} className="text-slate-400 text-lg leading-relaxed mb-10 max-w-lg">
            A hybrid quantum-classical system that classifies federal network
            connections as normal or attack using a Quantum SVM — benchmarked
            honestly against a classical baseline.
          </motion.p>

          <motion.div {...fadeUp(0.3)} className="flex flex-wrap gap-4">
            <a
              href={STREAMLIT_URL}
              target="_blank"
              rel="noopener noreferrer"
              className="group relative px-7 py-3.5 bg-purple-600 hover:bg-purple-500 text-white font-medium rounded-full transition-all duration-200 flex items-center gap-2 text-sm"
            >
              <span>Run the model</span>
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" className="group-hover:translate-x-0.5 transition-transform">
                <path d="M5 12h14M12 5l7 7-7 7"/>
              </svg>
            </a>
            <a
              href="https://github.com/Bhargav-Sarvepalli/qsvm-anomaly-detection"
              target="_blank"
              rel="noopener noreferrer"
              className="px-7 py-3.5 border border-border hover:border-slate-600 text-slate-300 hover:text-white font-medium rounded-full transition-all duration-200 flex items-center gap-2 text-sm"
            >
              <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
                <path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0 0 24 12c0-6.63-5.37-12-12-12z"/>
              </svg>
              View source
            </a>
          </motion.div>

          <motion.div {...fadeUp(0.4)} className="flex gap-8 mt-12 pt-8 border-t border-border">
            {[
              { val: '99.2%', label: 'Classical accuracy' },
              { val: '41→2', label: 'Feature compression' },
              { val: '2⁴¹', label: 'Hilbert dimensions' },
            ].map(({ val, label }) => (
              <div key={label}>
                <div className="text-xl font-bold text-white">{val}</div>
                <div className="text-xs text-slate-500 mt-0.5">{label}</div>
              </div>
            ))}
          </motion.div>
        </div>

        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 1, delay: 0.2, ease: [0.22, 1, 0.36, 1] }}
          className="h-[480px] lg:h-[560px]"
        >
          <Suspense fallback={
            <div className="w-full h-full flex items-center justify-center">
              <div className="w-16 h-16 rounded-full border border-purple-600/40 animate-pulse" />
            </div>
          }>
            <QuantumOrb />
          </Suspense>
        </motion.div>
      </div>
    </section>
  )
}