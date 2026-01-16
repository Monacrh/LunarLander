import * as THREE from "https://cdn.jsdelivr.net/npm/three@0.160/build/three.module.js";
import { EffectComposer } from "https://cdn.jsdelivr.net/npm/three@0.160/examples/jsm/postprocessing/EffectComposer.js";
import { RenderPass } from "https://cdn.jsdelivr.net/npm/three@0.160/examples/jsm/postprocessing/RenderPass.js";
import { UnrealBloomPass } from "https://cdn.jsdelivr.net/npm/three@0.160/examples/jsm/postprocessing/UnrealBloomPass.js";

// =====================
// 1. BASIC SETUP (SCENE, CAMERA, RENDERER)
// =====================
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x050508); // Sedikit lebih terang dari hitam pekat
scene.fog = new THREE.FogExp2(0x050508, 0.02);

const camera = new THREE.PerspectiveCamera(
  60,
  window.innerWidth / window.innerHeight,
  0.1,
  1000
);
camera.position.set(0, 4, 12);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(window.devicePixelRatio);
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.toneMapping = THREE.ReinhardToneMapping; // Agar warna bloom tidak 'burn'
document.body.appendChild(renderer.domElement);

// =====================
// 2. POST-PROCESSING (BLOOM / GLOW EFFECT)
// =====================
const renderScene = new RenderPass(scene, camera);

// Resolution, Strength, Radius, Threshold
const bloomPass = new UnrealBloomPass(
  new THREE.Vector2(window.innerWidth, window.innerHeight),
  1.5,
  0.4,
  0.85
);
bloomPass.strength = 1.2; // Kekuatan cahaya
bloomPass.radius = 0.5;   // Sebaran cahaya
bloomPass.threshold = 0.2; // Batas kecerahan objek yang akan glowing

const composer = new EffectComposer(renderer);
composer.addPass(renderScene);
composer.addPass(bloomPass);

// =====================
// 3. LIGHTING
// =====================
const ambientLight = new THREE.AmbientLight(0x404040, 0.5);
scene.add(ambientLight);

const sunLight = new THREE.DirectionalLight(0xffffff, 2);
sunLight.position.set(20, 50, 20);
sunLight.castShadow = true;
sunLight.shadow.mapSize.width = 2048;
sunLight.shadow.mapSize.height = 2048;
scene.add(sunLight);

// Cahaya biru dari bawah (pantulan atmosfer tipis/es)
const moonLight = new THREE.DirectionalLight(0x8888ff, 0.5);
moonLight.position.set(-5, -10, -5);
scene.add(moonLight);

// =====================
// 4. BACKGROUND (STARFIELD & TRAJECTORY)
// =====================
function createStars() {
  const starGeo = new THREE.BufferGeometry();
  const starCount = 3000;
  const posArray = new Float32Array(starCount * 3);
  const colorArray = new Float32Array(starCount * 3);

  for(let i = 0; i < starCount * 3; i+=3) {
    posArray[i] = (Math.random() - 0.5) * 300;
    posArray[i+1] = (Math.random() - 0.5) * 300;
    posArray[i+2] = (Math.random() - 0.5) * 100 - 50; // Jauh di belakang

    // Variasi warna bintang (putih, biru muda, kuning pucat)
    const starType = Math.random();
    let c = new THREE.Color();
    if(starType > 0.9) c.setHex(0xaaaaff); // Biru
    else if(starType > 0.7) c.setHex(0xffffee); // Kuning
    else c.setHex(0xffffff); // Putih

    colorArray[i] = c.r;
    colorArray[i+1] = c.g;
    colorArray[i+2] = c.b;
  }
  
  starGeo.setAttribute('position', new THREE.BufferAttribute(posArray, 3));
  starGeo.setAttribute('color', new THREE.BufferAttribute(colorArray, 3));

  const starMat = new THREE.PointsMaterial({
    size: 0.2,
    vertexColors: true,
    transparent: true,
    opacity: 0.8
  });
  
  const stars = new THREE.Points(starGeo, starMat);
  scene.add(stars);
}
createStars();

// Fungsi menggambar jalur lintasan (Trajectory)
function createTrajectory(episodeData) {
  const points = [];
  episodeData.forEach(step => {
    // Ambil posisi X, Y. Z=0.
    // Kita naikkan sedikit Z-nya (-0.1) agar ada di belakang roket tapi di depan background
    points.push(new THREE.Vector3(step.state.x, step.state.y, -0.1));
  });

  const geometry = new THREE.BufferGeometry().setFromPoints(points);
  const material = new THREE.LineBasicMaterial({ 
    color: 0x00ff44, // Hijau HUD
    opacity: 0.3, 
    transparent: true,
    linewidth: 2
  });
  
  const trajectoryLine = new THREE.Line(geometry, material);
  scene.add(trajectoryLine);
}

// =====================
// 5. TERRAIN
// =====================
function createMoonSurface() {
  const geo = new THREE.PlaneGeometry(100, 100, 80, 80);
  const posAttribute = geo.attributes.position;
  const vertex = new THREE.Vector3();

  for (let i = 0; i < posAttribute.count; i++) {
    vertex.fromBufferAttribute(posAttribute, i);
    let zHeight = Math.sin(vertex.x * 0.15) * Math.cos(vertex.y * 0.15) * 2; 
    zHeight += Math.random() * 0.5;

    // Landing pad area
    if (Math.abs(vertex.x) < 5 && Math.abs(vertex.y) < 3) {
      zHeight *= 0.05;
    }
    posAttribute.setZ(i, zHeight);
  }

  geo.computeVertexNormals();

  const mat = new THREE.MeshStandardMaterial({
    color: 0x333338,
    roughness: 0.8,
    metalness: 0.2,
    flatShading: true
  });

  const mesh = new THREE.Mesh(geo, mat);
  mesh.rotation.x = -Math.PI / 2;
  mesh.position.y = -0.6; 
  mesh.receiveShadow = true;
  scene.add(mesh);
  
  // Grid Helper di tanah untuk kesan simulasi
  const grid = new THREE.GridHelper(100, 50, 0x444444, 0x111111);
  grid.position.y = -1.45;
  scene.add(grid);
}
createMoonSurface();

// =====================
// 6. LANDER & PARTICLES
// =====================
const lander = new THREE.Group();

// Materials dengan Emissive untuk efek Glow
const matBody = new THREE.MeshStandardMaterial({ color: 0xeeeeee, roughness: 0.3, metalness: 0.5 });
const matRed = new THREE.MeshStandardMaterial({ color: 0xd62828, roughness: 0.4 });
const matDark = new THREE.MeshStandardMaterial({ color: 0x222222, roughness: 0.8 });
const matWindow = new THREE.MeshStandardMaterial({ 
  color: 0x00ffff, 
  emissive: 0x00aaaa, // Jendela bersinar
  emissiveIntensity: 0.8,
  roughness: 0.2 
});

// A. Body
const body = new THREE.Mesh(new THREE.CylinderGeometry(0.3, 0.4, 1.2, 16), matBody);
body.position.y = 0.2;
body.castShadow = true;
lander.add(body);

// B. Nose Cone
const nose = new THREE.Mesh(new THREE.ConeGeometry(0.3, 0.5, 16), matRed);
nose.position.y = 1.05;
nose.castShadow = true;
lander.add(nose);

// C. Engine Nozzle
const engine = new THREE.Mesh(new THREE.CylinderGeometry(0.25, 0.15, 0.3, 16), matDark);
engine.position.y = -0.45;
lander.add(engine);

// D. Cockpit
const cockpit = new THREE.Mesh(new THREE.SphereGeometry(0.15, 16, 16), matWindow);
cockpit.position.set(0, 0.5, 0.3);
cockpit.scale.z = 0.5;
lander.add(cockpit);

// E. Legs (Landing Gear) - Lebih detail sedikit
const legGeo = new THREE.BoxGeometry(0.1, 0.8, 0.1);
for (let i = 0; i < 4; i++) {
  const legGroup = new THREE.Group();
  const leg = new THREE.Mesh(legGeo, matBody);
  leg.position.set(0.4, -0.2, 0);
  leg.rotation.z = Math.PI / 4;
  leg.castShadow = true;
  
  const foot = new THREE.Mesh(new THREE.CylinderGeometry(0.15, 0.15, 0.05, 8), matDark);
  foot.position.set(0.65, -0.6, 0);
  foot.rotation.z = -Math.PI / 4; // Luruskan foot
  
  legGroup.add(leg);
  legGroup.add(foot);
  legGroup.rotation.y = (Math.PI / 2) * i + (Math.PI/4); // Rotate 45 derajat biar silang
  lander.add(legGroup);
}
scene.add(lander);

// --- PARTICLE SYSTEM UNTUK API ---
const particleCount = 200;
const particlesGeo = new THREE.BufferGeometry();
const particlePositions = new Float32Array(particleCount * 3);
const particleLifetimes = new Float32Array(particleCount); // Umur partikel
const particleVelocities = []; // Array of Vector3

// Inisialisasi partikel di tempat tersembunyi
for(let i=0; i<particleCount; i++) {
  particlePositions[i*3] = 0;     // x
  particlePositions[i*3+1] = -500; // y (hidden)
  particlePositions[i*3+2] = 0;     // z
  particleLifetimes[i] = 0;
  particleVelocities.push(new THREE.Vector3());
}

particlesGeo.setAttribute('position', new THREE.BufferAttribute(particlePositions, 3));
const particlesMat = new THREE.PointsMaterial({
  color: 0xffaa00,
  size: 0.3,
  transparent: true,
  opacity: 0.8,
  blending: THREE.AdditiveBlending, // Efek cahaya bertumpuk
  depthWrite: false
});
const particleSystem = new THREE.Points(particlesGeo, particlesMat);
scene.add(particleSystem);

// Fungsi Spawn Partikel
function spawnParticle(sourcePos, direction, spread, speed) {
  for(let i=0; i<particleCount; i++) {
    if(particleLifetimes[i] <= 0) { // Cari partikel mati
      particleLifetimes[i] = 1.0; // Reset umur (1.0 = 100%)
      
      // Posisi awal (relatif terhadap world, diambil dari posisi nozzle roket)
      particlePositions[i*3] = sourcePos.x;
      particlePositions[i*3+1] = sourcePos.y;
      particlePositions[i*3+2] = sourcePos.z;

      // Velocity acak
      const v = particleVelocities[i];
      v.copy(direction).multiplyScalar(speed);
      v.x += (Math.random() - 0.5) * spread;
      v.y += (Math.random() - 0.5) * spread;
      v.z += (Math.random() - 0.5) * spread;
      
      break; // Spawn satu per frame call cukup
    }
  }
}

function updateParticles() {
  const positions = particleSystem.geometry.attributes.position.array;

  for(let i=0; i<particleCount; i++) {
    if(particleLifetimes[i] > 0) {
      // Kurangi umur
      particleLifetimes[i] -= 0.02; // Kecepatan hilangnya api

      // Gerakkan partikel
      positions[i*3] += particleVelocities[i].x;
      positions[i*3+1] += particleVelocities[i].y;
      positions[i*3+2] += particleVelocities[i].z;

      // Reset jika mati
      if(particleLifetimes[i] <= 0) {
        positions[i*3+1] = -500; // Sembunyikan
      }
    }
  }
  particleSystem.geometry.attributes.position.needsUpdate = true;
}

// Light untuk engine glow
const engineLight = new THREE.PointLight(0xff6600, 0, 8);
engineLight.position.set(0, -0.5, 0);
lander.add(engineLight);


// =====================
// 7. GAME LOGIC
// =====================
let episode = [];
let frameIndex = 0;
let playing = true;

fetch("episode.json")
  .then(res => res.json())
  .then(data => {
    episode = data;
    createTrajectory(data); // Buat garis lintasan saat data dimuat
    if(document.getElementById("scrubber")) 
      document.getElementById("scrubber").max = episode.length - 1;
    animate();
  });

// UI Hooks
const btn = document.getElementById("playPause");
const scrub = document.getElementById("scrubber");
if(btn) btn.onclick = () => { playing = !playing; btn.innerText = playing ? "Pause" : "Play" };
if(scrub) scrub.oninput = () => { playing = false; btn.innerText = "Play"; frameIndex = Number(scrub.value); renderFrame(frameIndex); };

function handleEngineEffects(action) {
  const worldPos = new THREE.Vector3();
  
  // 1. Main Engine (Action 2)
  if (action === 2) {
    // Cari posisi nozzle di world space
    engine.getWorldPosition(worldPos);
    // Arah semburan (kebalikan dari arah atas roket)
    const downVec = new THREE.Vector3(0, -1, 0).applyQuaternion(lander.quaternion);
    
    // Spawn banyak partikel biar tebal
    spawnParticle(worldPos, downVec, 0.2, 0.15);
    spawnParticle(worldPos, downVec, 0.1, 0.2);

    engineLight.intensity = 3 + Math.random(); // Flicker light
  } 
  // 2. Side Engines (Action 1 & 3)
  else if (action === 1 || action === 3) {
    // Sederhanakan side thruster: spawn dari samping bodi
    lander.getWorldPosition(worldPos);
    // Geser sedikit ke kiri/kanan lokal
    const offset = action === 1 ? -0.3 : 0.3;
    const sideVec = new THREE.Vector3(offset, 0.5, 0).applyQuaternion(lander.quaternion);
    worldPos.add(sideVec);
    
    const thrustDir = new THREE.Vector3(action === 1 ? 1 : -1, 0, 0).applyQuaternion(lander.quaternion);
    spawnParticle(worldPos, thrustDir, 0.05, 0.1);
    
    engineLight.intensity = 0.5;
  } else {
    engineLight.intensity = 0;
  }
}

function renderFrame(i) {
  const f = episode[i];
  if (!f) return;
  
  // Update posisi roket
  if (f.state) {
    lander.position.set(f.state.x, f.state.y, 0);
    lander.rotation.z = f.state.angle;
  } else {
    lander.position.set(f.x, f.y, 0);
    lander.rotation.z = f.angle;
  }
  
  // Update efek visual
  handleEngineEffects(f.action ?? 0);
}

function updateCamera(action) {
  // Smooth Follow dengan sedikit "lead" (melihat ke arah gerakan)
  const targetX = lander.position.x;
  const targetY = lander.position.y + 2; // Kamera agak di atas roket
  
  camera.position.x += (targetX - camera.position.x) * 0.05;
  camera.position.y += (targetY - camera.position.y) * 0.05;
  
  // Zoom out jika roket tinggi
  const targetZ = lander.position.y < 5 ? 12 : 20;
  camera.position.z += (targetZ - camera.position.z) * 0.02;

  // Sedikit goyang kamera jika mesin utama nyala (Screen Shake)
  if (action === 2) {
    camera.position.x += (Math.random() - 0.5) * 0.02;
    camera.position.y += (Math.random() - 0.5) * 0.02;
  }
  
  camera.lookAt(lander.position.x, lander.position.y, 0);
}

function animate() {
  requestAnimationFrame(animate);

  if (episode.length && playing) {
    frameIndex++;
    if (frameIndex >= episode.length) frameIndex = episode.length - 1;
    if(scrub) scrub.value = frameIndex;
  }
  
  if(episode.length) {
    const currentAction = episode[frameIndex]?.action ?? 0;
    renderFrame(frameIndex);
    updateCamera(currentAction);
  }

  updateParticles();

  // PENTING: Gunakan composer.render(), bukan renderer.render() untuk efek bloom
  composer.render();
}

window.addEventListener("resize", () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
  composer.setSize(window.innerWidth, window.innerHeight); // Resize bloom composer
});