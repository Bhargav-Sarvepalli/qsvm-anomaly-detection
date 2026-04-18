const STREAMLIT_URL = 'https://qsvm-anomaly-detection.streamlit.app'

export default function Footer() {
  return (
    <footer className="border-t border-border py-10">
      <div className="max-w-6xl mx-auto px-6 flex flex-col md:flex-row items-center justify-between gap-6">
        <div className="flex items-center gap-3">
          <div className="w-6 h-6 rounded-full bg-purple-600 flex items-center justify-center">
            <span className="text-white text-xs font-bold">Q</span>
          </div>
          <span className="text-sm text-slate-500">
            QSVM · Network Intrusion Detection
          </span>
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
