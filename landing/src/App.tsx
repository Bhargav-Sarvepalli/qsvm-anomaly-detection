import { useEffect } from 'react'
import Lenis from 'lenis'
import Navbar from './components/Navbar'
import Hero from './components/Hero'
import Problem from './components/Problem'
import Solution from './components/Solution'
import Results from './components/Results'
import CTASection from './components/CTASection'
import Architecture from './components/Architecture'
import About from './components/About'
import Footer from './components/Footer'

export default function App() {
  useEffect(() => {
    const lenis = new Lenis({
      duration: 1.2,
      easing: (t) => Math.min(1, 1.001 - Math.pow(2, -10 * t)),
      smoothWheel: true,
    })

    function raf(time: number) {
      lenis.raf(time)
      requestAnimationFrame(raf)
    }
    requestAnimationFrame(raf)

    return () => lenis.destroy()
  }, [])

  return (
    <div className="noise-overlay">
      <Navbar />
      <main>
        <Hero />
        <Problem />
        <Solution />
        <Results />
        <CTASection />
        <Architecture />
        <About />
      </main>
      <Footer />
    </div>
  )
}
