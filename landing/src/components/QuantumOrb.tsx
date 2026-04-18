import { useRef, useMemo } from 'react'
import { Canvas, useFrame, useThree } from '@react-three/fiber'
import { Points, PointMaterial } from '@react-three/drei'
import * as THREE from 'three'

function ParticleField() {
  const ref = useRef<THREE.Points>(null!)
  const count = 2000

  const positions = useMemo(() => {
    const arr = new Float32Array(count * 3)
    for (let i = 0; i < count; i++) {
      const r = Math.random() * 4 + 1
      const theta = Math.random() * Math.PI * 2
      const phi = Math.acos(2 * Math.random() - 1)
      arr[i * 3]     = r * Math.sin(phi) * Math.cos(theta)
      arr[i * 3 + 1] = r * Math.sin(phi) * Math.sin(theta)
      arr[i * 3 + 2] = r * Math.cos(phi)
    }
    return arr
  }, [])

  useFrame((state) => {
    ref.current.rotation.y = state.clock.elapsedTime * 0.04
    ref.current.rotation.x = Math.sin(state.clock.elapsedTime * 0.02) * 0.1
  })

  return (
    <Points ref={ref} positions={positions} stride={3} frustumCulled={false}>
      <PointMaterial
        transparent
        color="#8b5cf6"
        size={0.015}
        sizeAttenuation
        depthWrite={false}
        opacity={0.6}
      />
    </Points>
  )
}

function QuantumRing({ radius, speed, color, tilt }: {
  radius: number; speed: number; color: string; tilt: [number, number, number]
}) {
  const ref = useRef<THREE.Mesh>(null!)

  useFrame((state) => {
    ref.current.rotation.z = state.clock.elapsedTime * speed
    ref.current.rotation.x = tilt[0] + Math.sin(state.clock.elapsedTime * 0.3) * 0.05
  })

  return (
    <mesh ref={ref} rotation={tilt}>
      <torusGeometry args={[radius, 0.006, 16, 120]} />
      <meshBasicMaterial color={color} transparent opacity={0.5} />
    </mesh>
  )
}

function CoreOrb() {
  const ref = useRef<THREE.Mesh>(null!)
  const { mouse } = useThree()

  useFrame((state) => {
    ref.current.rotation.y = state.clock.elapsedTime * 0.2
    ref.current.rotation.x = state.clock.elapsedTime * 0.1
    ref.current.position.x += (mouse.x * 0.3 - ref.current.position.x) * 0.05
    ref.current.position.y += (mouse.y * 0.2 - ref.current.position.y) * 0.05
  })

  return (
    <mesh ref={ref}>
      <icosahedronGeometry args={[0.7, 4]} />
      <meshStandardMaterial
        color="#1a0a3c"
        emissive="#6d28d9"
        emissiveIntensity={0.4}
        wireframe={false}
        metalness={0.8}
        roughness={0.2}
      />
    </mesh>
  )
}

function WireOrb() {
  const ref = useRef<THREE.Mesh>(null!)

  useFrame((state) => {
    ref.current.rotation.y = -state.clock.elapsedTime * 0.15
    ref.current.rotation.z = state.clock.elapsedTime * 0.08
  })

  return (
    <mesh ref={ref}>
      <icosahedronGeometry args={[0.85, 2]} />
      <meshBasicMaterial color="#7c3aed" wireframe transparent opacity={0.2} />
    </mesh>
  )
}

function Qubit({ position, phase }: { position: [number, number, number]; phase: number }) {
  const ref = useRef<THREE.Mesh>(null!)

  useFrame((state) => {
    const t = state.clock.elapsedTime + phase
    ref.current.position.x = position[0] + Math.sin(t * 0.8) * 0.12
    ref.current.position.y = position[1] + Math.cos(t * 0.6) * 0.12
    ref.current.position.z = position[2]
  })

  return (
    <mesh ref={ref} position={position}>
      <sphereGeometry args={[0.045, 8, 8]} />
      <meshBasicMaterial color="#a78bfa" />
    </mesh>
  )
}

function Scene() {
  const qubitPositions: [number, number, number][] = [
    [1.4, 0.3, 0], [-1.4, -0.3, 0],
    [0, 1.5, 0.2], [0, -1.5, -0.2],
    [1.0, -1.1, 0.3], [-1.0, 1.1, -0.3],
  ]

  return (
    <>
      <ambientLight intensity={0.3} />
      <pointLight position={[4, 4, 4]} color="#8b5cf6" intensity={2} />
      <pointLight position={[-4, -4, -4]} color="#3b82f6" intensity={1} />

      <ParticleField />
      <CoreOrb />
      <WireOrb />

      <QuantumRing radius={1.3} speed={0.3} color="#8b5cf6" tilt={[0.4, 0, 0]} />
      <QuantumRing radius={1.6} speed={-0.2} color="#3b82f6" tilt={[1.1, 0.3, 0]} />
      <QuantumRing radius={1.9} speed={0.15} color="#6d28d9" tilt={[0.7, 1.0, 0.2]} />

      {qubitPositions.map((pos, i) => (
        <Qubit key={i} position={pos} phase={i * 1.2} />
      ))}
    </>
  )
}

export default function QuantumOrb() {
  return (
    <div className="w-full h-full">
      <Canvas
        camera={{ position: [0, 0, 4.5], fov: 55 }}
        dpr={[1, 2]}
        gl={{ antialias: true, alpha: true }}
      >
        <Scene />
      </Canvas>
    </div>
  )
}
