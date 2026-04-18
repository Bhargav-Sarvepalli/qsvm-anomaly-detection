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

const otherProjects = [
  {
    name: 'NexTask',
    desc: 'AI-powered Kanban SaaS with real-time collaboration, Anthropic API, Three.js landing page',
    stack: 'React · TypeScript · Supabase · Anthropic API · Three.js',
    url: 'https://kanban-board-beige-seven.vercel.app',
  },
  {
    name: 'Pulsar Star Detection',
    desc: 'ML microservice platform achieving 99% accuracy and ROC-AUC 0.973 on telescope data',
    stack: 'FastAPI · LightGBM · Docker · AWS EC2 · Streamlit · SHAP',
    url: 'https://github.com/Bhargav-Sarvepalli/pulsar-star-prediction-service',
  },
]

export default function About() {
  return (
    <section id="about" className="py-28 relative border-t border-border">
      <div className="max-w-6xl mx-auto px-6">
        <div className="grid lg:grid-cols-2 gap-16">

          {/* Developer */}
          <div>
            <FadeIn>
              <p className="text-purple-400 text-xs font-semibold tracking-widest uppercase mb-8">Built by</p>

              <div className="flex items-center gap-5 mb-6">
                <img
                    src="/bhargav.png"
                    alt="Sree Sai Bhargav Sarvepalli"
                    className="w-20 h-20 rounded-full object-cover object-top border border-border"
                 />
                <div>
                  <h3 className="text-white font-bold text-xl leading-tight">
                    Sree Sai Bhargav<br />Sarvepalli
                  </h3>
                  <p className="text-slate-400 text-sm mt-1">
                    M.S. Computer Science · UMBC · GPA 3.74
                  </p>
                </div>
              </div>

              <p className="text-slate-400 text-sm leading-relaxed mb-8">
                Full-stack engineer and ML practitioner with experience across
                production web systems, machine learning pipelines, and quantum
                computing applications. Built this project to go beyond coursework —
                demonstrating quantum kernels as a viable, testable approach
                with a clear path to real hardware advantage.
              </p>

              <div className="flex flex-wrap gap-3">
                <a href="https://bhargav.tech" target="_blank" rel="noopener noreferrer"
                  className="flex items-center gap-2 px-4 py-2 gradient-border rounded-full text-sm text-white hover:border-purple-600/50 transition-colors">
                  bhargav.tech
                </a>
                <a href="https://github.com/Bhargav-Sarvepalli" target="_blank" rel="noopener noreferrer"
                  className="flex items-center gap-2 px-4 py-2 gradient-border rounded-full text-sm text-slate-300 hover:text-white transition-colors">
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
                    <path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0 0 24 12c0-6.63-5.37-12-12-12z"/>
                  </svg>
                  GitHub
                </a>
                <a href="https://linkedin.com/in/bhargav-sarvepalli" target="_blank" rel="noopener noreferrer"
                  className="flex items-center gap-2 px-4 py-2 gradient-border rounded-full text-sm text-slate-300 hover:text-white transition-colors">
                  LinkedIn
                </a>
              </div>
            </FadeIn>
          </div>

          {/* Other projects */}
          <div>
            <FadeIn delay={0.1}>
              <p className="text-purple-400 text-xs font-semibold tracking-widest uppercase mb-8">Other projects</p>
              <div className="space-y-4">
                {otherProjects.map(({ name, desc, stack, url }, i) => (
                  <FadeIn key={name} delay={0.08 * i}>
                    <a
                      href={url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="group block gradient-border rounded-xl p-5 hover:border-purple-600/40 transition-all"
                    >
                      <div className="flex items-start justify-between">
                        <h4 className="text-white font-semibold text-sm group-hover:text-purple-300 transition-colors">
                          {name}
                        </h4>
                        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor"
                          strokeWidth="2" className="text-slate-600 group-hover:text-purple-400 transition-colors mt-0.5 flex-shrink-0 ml-3">
                          <path d="M7 17L17 7M17 7H7M17 7v10"/>
                        </svg>
                      </div>
                      <p className="text-slate-500 text-xs leading-relaxed mt-1.5">{desc}</p>
                      <p className="text-slate-600 text-xs font-mono mt-2">{stack}</p>
                    </a>
                  </FadeIn>
                ))}
              </div>
            </FadeIn>
          </div>

        </div>
      </div>
    </section>
  )
}