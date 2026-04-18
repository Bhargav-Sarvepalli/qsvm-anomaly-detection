import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'

const STREAMLIT_URL = 'https://qsvm-anomaly-detection.streamlit.app'

function QSVMLogo() {
  return (
    <svg width="32" height="32" viewBox="0 0 32 32" fill="none" xmlns="http://www.w3.org/2000/svg">
      {/* Outer orbit ring */}
      <ellipse cx="16" cy="16" rx="14" ry="6" stroke="#8b5cf6" strokeWidth="1" opacity="0.6"
        transform="rotate(-30 16 16)" />
      {/* Inner orbit ring */}
      <ellipse cx="16" cy="16" rx="14" ry="6" stroke="#60a5fa" strokeWidth="0.8" opacity="0.4"
        transform="rotate(30 16 16)" />
      {/* Core circle */}
      <circle cx="16" cy="16" r="4.5" fill="#7c3aed" opacity="0.9" />
      <circle cx="16" cy="16" r="2.5" fill="#a78bfa" />
      {/* Orbiting qubit dots */}
      <circle cx="28" cy="13" r="1.5" fill="#8b5cf6" />
      <circle cx="5" cy="20" r="1.2" fill="#60a5fa" opacity="0.8" />
      <circle cx="20" cy="3" r="1" fill="#a78bfa" opacity="0.7" />
    </svg>
  )
}

export default function Navbar() {
  const [scrolled, setScrolled] = useState(false)
  const [menuOpen, setMenuOpen] = useState(false)

  useEffect(() => {
    const handler = () => setScrolled(window.scrollY > 30)
    window.addEventListener('scroll', handler)
    return () => window.removeEventListener('scroll', handler)
  }, [])

  const links = ['Problem', 'Solution', 'Results', 'Architecture', 'About']

  return (
    <motion.nav
      initial={{ opacity: 0, y: -16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      className={`fixed top-0 left-0 right-0 z-50 transition-all duration-300 ${
        scrolled ? 'bg-bg/90 backdrop-blur-md border-b border-border' : ''
      }`}
    >
      <div className="max-w-6xl mx-auto px-6 h-16 flex items-center justify-between">
        <div className="flex items-center gap-2.5">
          <QSVMLogo />
          <span className="font-semibold text-sm text-white tracking-tight">QSVM</span>
        </div>

        <div className="hidden md:flex items-center gap-8">
          {links.map(link => (
            <a
              key={link}
              href={`#${link.toLowerCase()}`}
              className="text-sm text-slate-400 hover:text-white transition-colors duration-200"
            >
              {link}
            </a>
          ))}
        </div>

        <div className="flex items-center gap-3">
          <a
            href="https://github.com/Bhargav-Sarvepalli/qsvm-anomaly-detection"
            target="_blank"
            rel="noopener noreferrer"
            className="hidden md:flex items-center gap-1.5 text-sm text-slate-400 hover:text-white transition-colors"
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
              <path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0 0 24 12c0-6.63-5.37-12-12-12z"/>
            </svg>
            GitHub
          </a>
          <a
            href={STREAMLIT_URL}
            target="_blank"
            rel="noopener noreferrer"
            className="px-4 py-1.5 bg-purple-600 hover:bg-purple-500 text-white text-sm font-medium rounded-full transition-colors duration-200"
          >
            Try the model
          </a>
          <button
            className="md:hidden text-slate-400 hover:text-white"
            onClick={() => setMenuOpen(!menuOpen)}
          >
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              {menuOpen
                ? <path d="M18 6L6 18M6 6l12 12"/>
                : <><path d="M3 12h18"/><path d="M3 6h18"/><path d="M3 18h18"/></>
              }
            </svg>
          </button>
        </div>
      </div>

      <AnimatePresence>
        {menuOpen && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="md:hidden bg-surface border-b border-border px-6 pb-4"
          >
            {links.map(link => (
              <a
                key={link}
                href={`#${link.toLowerCase()}`}
                className="block py-2.5 text-sm text-slate-400 hover:text-white transition-colors"
                onClick={() => setMenuOpen(false)}
              >
                {link}
              </a>
            ))}
          </motion.div>
        )}
      </AnimatePresence>
    </motion.nav>
  )
}