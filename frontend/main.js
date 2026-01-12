import * as THREE from "https://cdn.jsdelivr.net/npm/three@0.160/build/three.module.js"

// =====================
// 1. BASIC SETUP (SCENE & CAMERA)
// =====================
const scene = new THREE.Scene()
scene.background = new THREE.Color(0x020205) // Hitam pekat sedikit biru
scene.fog = new THREE.FogExp2(0x020205, 0.02) // Kabut hitam untuk kedalaman

const camera = new THREE.PerspectiveCamera(
  60,
  window.innerWidth / window.innerHeight,
  0.1,
  1000
)
camera.position.set(0, 4, 10)

const renderer = new THREE.WebGLRenderer({ antialias: true })
renderer.setSize(window.innerWidth, window.innerHeight)
renderer.shadowMap.enabled = true // Aktifkan bayangan
renderer.shadowMap.type = THREE.PCFSoftShadowMap
document.body.appendChild(renderer.domElement)

// =====================
// 2. LIGHTING (SPACE ATMOSPHERE)
// =====================
// Cahaya ambien yang sangat redup (angkasa itu gelap)
scene.add(new THREE.AmbientLight(0x404040, 0.6))

// Cahaya Matahari (Keras dan menciptakan bayangan tajam)
const sunLight = new THREE.DirectionalLight(0xffffff, 1.5)
sunLight.position.set(20, 50, 20)
sunLight.castShadow = true
sunLight.shadow.mapSize.width = 2048
sunLight.shadow.mapSize.height = 2048
sunLight.shadow.camera.near = 0.5
sunLight.shadow.camera.far = 100
sunLight.shadow.camera.left = -30
sunLight.shadow.camera.right = 30
sunLight.shadow.camera.top = 30
sunLight.shadow.camera.bottom = -30
scene.add(sunLight)

// =====================
// 3. BACKGROUND (STARFIELD)
// =====================
function createStars() {
  const starGeo = new THREE.BufferGeometry()
  const starCount = 2000
  const posArray = new Float32Array(starCount * 3)

  for(let i = 0; i < starCount * 3; i++) {
    // Sebar bintang di area yang luas
    posArray[i] = (Math.random() - 0.5) * 200 
  }
  
  starGeo.setAttribute('position', new THREE.BufferAttribute(posArray, 3))
  const starMat = new THREE.PointsMaterial({
    size: 0.15,
    color: 0xffffff,
    transparent: true,
    opacity: 0.8
  })
  
  const stars = new THREE.Points(starGeo, starMat)
  scene.add(stars)
}
createStars()

// =====================
// 4. TERRAIN (LOW POLY MOON SURFACE)
// =====================
function createMoonSurface() {
  // Buat plane dengan segmen yang banyak agar bisa diubah bentuknya
  const geo = new THREE.PlaneGeometry(80, 80, 60, 60)
  
  // Manipulasi Vertex untuk membuat Kawah & Bukit
  const posAttribute = geo.attributes.position
  const vertex = new THREE.Vector3()

  for (let i = 0; i < posAttribute.count; i++) {
    vertex.fromBufferAttribute(posAttribute, i)
    
    // Perhitungan acak (Noise sederhana)
    // Gelombang besar + Gelombang kecil
    let zHeight = Math.sin(vertex.x * 0.2) * Math.cos(vertex.y * 0.2) * 1.5 
    zHeight += Math.random() * 0.4

    // PENTING: Ratakan area tengah (tempat landing) agar tidak aneh
    const distFromCenter = Math.sqrt(vertex.x * vertex.x + vertex.y * vertex.y)
    if (Math.abs(vertex.x) < 8 && Math.abs(vertex.y) < 5) {
      zHeight *= 0.1 // Datar di tengah
    }

    // Set ketinggian baru (Z dalam PlaneGeometry adalah 'tinggi' sebelum dirotasi)
    posAttribute.setZ(i, zHeight)
  }

  geo.computeVertexNormals() // Hitung ulang pencahayaan

  const mat = new THREE.MeshStandardMaterial({
    color: 0x555555, // Abu-abu bulan
    roughness: 0.9,
    metalness: 0.1,
    flatShading: true // Gaya Low Poly (kotak-kotak terlihat)
  })

  const mesh = new THREE.Mesh(geo, mat)
  mesh.rotation.x = -Math.PI / 2
  mesh.position.y = -0.6 // Turunkan sedikit agar roket pas di atasnya
  mesh.receiveShadow = true
  scene.add(mesh)
}
createMoonSurface()


// =====================
// 5. LANDER (FUTURISTIC ROCKET)
// =====================
const lander = new THREE.Group()

// Materials
const matBody = new THREE.MeshStandardMaterial({ color: 0xeeeeee, roughness: 0.3, metalness: 0.2 })
const matRed = new THREE.MeshStandardMaterial({ color: 0xd62828, roughness: 0.4 })
const matDark = new THREE.MeshStandardMaterial({ color: 0x1d3557, roughness: 0.7 })
const matWindow = new THREE.MeshStandardMaterial({ color: 0x4cc9f0, emissive: 0x111111, roughness: 0.1 })

// A. Body
const body = new THREE.Mesh(new THREE.CylinderGeometry(0.3, 0.35, 1.2, 32), matBody)
body.position.y = 0.2
body.castShadow = true
lander.add(body)

// B. Nose
const nose = new THREE.Mesh(new THREE.ConeGeometry(0.3, 0.5, 32), matRed)
nose.position.y = 1.05
nose.castShadow = true
lander.add(nose)

// C. Engine
const engine = new THREE.Mesh(new THREE.CylinderGeometry(0.25, 0.2, 0.3, 32), matDark)
engine.position.y = -0.45
lander.add(engine)

// D. Cockpit
const cockpit = new THREE.Mesh(new THREE.SphereGeometry(0.12, 32, 16), matWindow)
cockpit.position.set(0, 0.5, 0.28)
cockpit.scale.z = 0.6
lander.add(cockpit)

// E. Fins
const finGeo = new THREE.BoxGeometry(0.4, 0.5, 0.05)
for (let i = 0; i < 4; i++) {
  const finGroup = new THREE.Group()
  const fin = new THREE.Mesh(finGeo, matRed)
  fin.position.set(0.4, -0.1, 0)
  fin.rotation.z = Math.PI / 6
  fin.castShadow = true
  finGroup.add(fin)
  finGroup.rotation.y = (Math.PI / 2) * i
  lander.add(finGroup)
}
scene.add(lander)

// =====================
// 6. FLAMES & PARTICLES
// =====================
function createFlame(color = 0xffaa00, h = 0.5, w = 0.15) {
  const geo = new THREE.ConeGeometry(w, h, 16)
  const mat = new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.8 })
  return new THREE.Mesh(geo, mat)
}

const flameMain = createFlame(0xffaa00, 1.2, 0.25)
flameMain.position.set(0, -1.0, 0)
flameMain.rotation.x = Math.PI
flameMain.visible = false
lander.add(flameMain)

const flameLeft = createFlame(0x00ffff, 0.4, 0.08)
flameLeft.position.set(-0.35, 0.6, 0)
flameLeft.rotation.z = Math.PI / 2
flameLeft.visible = false
lander.add(flameLeft)

const flameRight = createFlame(0x00ffff, 0.4, 0.08)
flameRight.position.set(0.35, 0.6, 0)
flameRight.rotation.z = -Math.PI / 2
flameRight.visible = false
lander.add(flameRight)

// Cahaya Engine (Glow effect saat mesin nyala)
const engineLight = new THREE.PointLight(0xffaa00, 0, 5)
engineLight.position.set(0, -1, 0)
lander.add(engineLight)

// =====================
// 7. GAME LOGIC
// =====================
let episode = []
let frameIndex = 0
let playing = true

fetch("episode.json")
  .then(res => res.json())
  .then(data => {
    episode = data
    if(document.getElementById("scrubber")) 
      document.getElementById("scrubber").max = episode.length - 1
    animate()
  })

// UI Hooks
const btn = document.getElementById("playPause")
const scrub = document.getElementById("scrubber")
if(btn) btn.onclick = () => { playing = !playing; btn.innerText = playing ? "Pause" : "Play" }
if(scrub) scrub.oninput = () => { playing = false; btn.innerText = "Play"; frameIndex = Number(scrub.value); renderFrame(frameIndex) }

function updateFlames(action) {
  flameMain.visible = (action === 2)
  flameLeft.visible = (action === 1)
  flameRight.visible = (action === 3)

  // Efek flicker api + cahaya
  if (flameMain.visible) {
    flameMain.scale.set(1 + Math.random()*0.2, 1 + Math.random()*0.4, 1 + Math.random()*0.2)
    engineLight.intensity = 2 + Math.random() // Cahaya kedip-kedip
  } else {
    engineLight.intensity = 0
  }
}

function renderFrame(i) {
  const f = episode[i]
  if (!f) return
  
  if (f.state) {
    lander.position.set(f.state.x, f.state.y, 0)
    lander.rotation.z = f.state.angle
  } else {
    lander.position.set(f.x, f.y, 0)
    lander.rotation.z = f.angle
  }
  updateFlames(f.action ?? 0)
}

function updateCamera(action) {
  // Smooth Follow
  camera.position.x += (lander.position.x - camera.position.x) * 0.08
  
  // Dynamic Zoom & Height
  // Kalau tinggi > 5, kamera mundur biar kelihatan luas
  // Kalau dekat tanah, kamera mendekat
  const targetZ = lander.position.y < 3 ? 6 : 12
  const targetY = lander.position.y + 3
  
  camera.position.z += (targetZ - camera.position.z) * 0.05
  camera.position.y += (targetY - camera.position.y) * 0.05
  
  // Camera Shake (Getaran mesin)
  if (action === 2) {
    camera.position.x += (Math.random() - 0.5) * 0.05
    camera.position.y += (Math.random() - 0.5) * 0.05
  }
  
  camera.lookAt(lander.position.x, lander.position.y, 0)
}

function animate() {
  requestAnimationFrame(animate)

  if (episode.length && playing) {
    frameIndex++
    if (frameIndex >= episode.length) frameIndex = episode.length - 1
    if(scrub) scrub.value = frameIndex
  }
  
  if(episode.length) {
    renderFrame(frameIndex)
    updateCamera(episode[frameIndex]?.action ?? 0)
  }

  renderer.render(scene, camera)
}

window.addEventListener("resize", () => {
  camera.aspect = window.innerWidth / window.innerHeight
  camera.updateProjectionMatrix()
  renderer.setSize(window.innerWidth, window.innerHeight)
})