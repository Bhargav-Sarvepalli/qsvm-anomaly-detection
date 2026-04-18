import { useRef } from 'react'
import { motion, useInView } from 'framer-motion'

const STREAMLIT_URL = 'https://qsvm-anomaly-detection.streamlit.app'

export default function CTASection() {
  const ref = useRef(null)
  const inView = useInView(ref, { once: true, margin: '-80px' })

  return (
    <section className="py-20 relative">
      <div className="max-w-6xl mx-auto px-6">
        <motion.div
          ref={ref}
          initial={{ opacity: 0, y: 32 }}
          animate={inView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.7, ease: [0.22, 1, 0.36, 1] }}
          className="relative rounded-3xl overflow-hidden"
          style={{
            background: 'linear-gradient(135deg, rgba(124,58,237,0.2) 0%, rgba(59,130,246,0.1) 100%)',
            border: '1px solid rgba(139, 92, 246, 0.3)',
          }}
        >
          {/* Background orb */}
          <div
            className="absolute inset-0 pointer-events-none"
            style={{
              background: 'radial-gradient(ellipse at 70% 50%, rgba(124,58,237,0.15) 0%, transparent 60%)',
            }}
          />

          {/* Animated grid lines */}
          <div className="absolute inset-0 pointer-events-none opacity-10"
            style={{
              backgroundImage: `
                linear-gradient(rgba(139,92,246,0.3) 1px, transparent 1px),
                linear-gradient(90deg, rgba(139,92,246,0.3) 1px, transparent 1px)
              `,
              backgroundSize: '40px 40px',
            }}
          />

          <div className="relative z-10 px-10 py-14 flex flex-col lg:flex-row items-center gap-10">
            <div className="flex-1">
              <p className="text-purple-400 text-xs font-semibold tracking-widest uppercase mb-4">
                Live interactive demo
              </p>
              <h2 className="text-3xl lg:text-4xl font-bold text-white mb-4 leading-tight">
                Run the quantum model yourself.
              </h2>
              <p className="text-slate-300 leading-relaxed max-w-lg">
                Open the Streamlit app and classify network connections live.
                Use the built-in KDD Cup 99 sample data or upload your own CSV.
                See confusion matrices, ROC curves, and full metrics in real time.
              </p>

              <div className="mt-6 flex flex-wrap gap-5">
                {[
                  'Classical SVM — instant results',
                  'Quantum SVM — pre-computed full run',
                  'Side-by-side comparison',
                  'Upload your own dataset',
                ].map(feature => (
                  <div key={feature} className="flex items-center gap-2 text-sm text-slate-400">
                    <div className="w-1.5 h-1.5 rounded-full bg-purple-400" />
                    {feature}
                  </div>
                ))}
              </div>
            </div>

            <div className="flex flex-col items-center gap-4">
              <motion.a
                href={STREAMLIT_URL}
                target="_blank"
                rel="noopener noreferrer"
                whileHover={{ scale: 1.04 }}
                whileTap={{ scale: 0.97 }}
                className="group flex items-center gap-3 px-8 py-4 bg-purple-600 hover:bg-purple-500 text-white font-semibold rounded-2xl transition-colors duration-200 text-base shadow-lg"
                style={{ boxShadow: '0 0 40px rgba(124, 58, 237, 0.4)' }}
              >
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <polygon points="5,3 19,12 5,21"/>
                </svg>
                Open the model
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"
                  className="group-hover:translate-x-0.5 transition-transform">
                  <path d="M5 12h14M12 5l7 7-7 7"/>
                </svg>
              </motion.a>
              <p className="text-xs text-slate-600">
                Hosted on Streamlit Cloud · free · no signup
              </p>
            </div>
          </div>
        </motion.div>
      </div>
    </section>
  )
}
