const STREAMLIT_URL = 'https://qsvm-anomaly-detection.streamlit.app'

function QSVMLogo() {
  return (
    <svg width="28" height="28" viewBox="0 0 32 32" fill="none" xmlns="http://www.w3.org/2000/svg">
      <ellipse cx="16" cy="16" rx="14" ry="6" stroke="#8b5cf6" strokeWidth="1" opacity="0.6"
        transform="rotate(-30 16 16)" />
      <ellipse cx="16" cy="16" rx="14" ry="6" stroke="#60a5fa" strokeWidth="0.8" opacity="0.4"
        transform="rotate(30 16 16)" />
      <circle cx="16" cy="16" r="4.5" fill="#7c3aed" opacity="0.9" />
      <circle cx="16" cy="16" r="2.5" fill="#a78bfa" />
      <circle cx="28" cy="13" r="1.5" fill="#8b5cf6" />
      <circle cx="5" cy="20" r="1.2" fill="#60a5fa" opacity="0.8" />
      <circle cx="20" cy="3" r="1" fill="#a78bfa" opacity="0.7" />
    </svg>
  )
}

export default function Footer() {
  return (
    <footer className="border-t border-border py-10">
      <div className="max-w-6xl mx-auto px-6 flex flex-col md:flex-row items-center justify-between gap-6">
        <div className="flex items-center gap-3">
          <QSVMLogo />
          <div>
            <span className="text-sm font-semibold text-white tracking-tight">QSVM</span>
            <span className="text-slate-600 text-sm"> · Network Intrusion Detection</span>
          </div>
        </div>

        <div className="flex items-center gap-6 text-xs text-slate-600">
          <a href={STREAMLIT_URL} target="_blank" rel="noopener noreferrer"
            className="hover:text-purple-400 transition-colors">
            Live app
          </a>
          <a href="https://github.com/Bhargav-Sarvepalli/qsvm-anomaly-detection"
            target="_blank" rel="noopener noreferrer"
            className="hover:text-purple-400 transition-colors">
            GitHub
          </a>
          <a href="https://bhargav.tech" target="_blank" rel="noopener noreferrer"
            className="hover:text-purple-400 transition-colors">
            bhargav.tech
          </a>
        </div>

        <p className="text-xs text-slate-700">
          Built by Sree Sai Bhargav Sarvepalli
        </p>
      </div>
    </footer>
  )
}