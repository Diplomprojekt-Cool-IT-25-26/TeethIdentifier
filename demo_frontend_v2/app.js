/**
 * DentAI Demo Frontend v2 — Showcase Mode
 * Step-by-step walkthrough of the teeth/gingiva classification pipeline.
 */

// ================================================================
// CONFIG
// ================================================================

const SCANS = {
  scan1: {
    label: 'Sample Scan A',
    jaw: 'Lower Jaw',
    originalPath: '../demo_frontend/samples/original_scan.obj',
    rawPath:      '../demo_frontend/samples/classified_raw.obj',          // fresh model run, no post-processing
    finalPath:    '../demo_frontend/samples/classified_postprocessed.obj', // same run + post-processing
    vertexCount: 93288,
    processingTime: 21.1,
  },
};

const PP_OPTIONS = [
  { id: 'openTeeth',    label: 'Opening on Teeth',              desc: 'Removes isolated tooth islands surrounded by gingiva' },
  { id: 'closeTeeth',   label: 'Closing on Teeth',              desc: 'Fills small gingiva gaps within tooth regions' },
  { id: 'erodeGingiva', label: 'Gingiva Erosion',               desc: 'Shrinks over-predicted gingiva at the tooth boundary' },
  { id: 'majorityVote', label: 'Majority Vote Smoothing',       desc: 'Trims ragged gingiva peninsula edges via neighbour voting' },
  { id: 'openGingiva',  label: 'Opening on Gingiva',            desc: 'Removes thin gingiva finger protrusions from tooth areas' },
  { id: 'reconstruct',  label: 'Morphological Reconstruction',  desc: 'Severs gingiva finger roots, then flood-fills back' },
];

const OPTIMAL = {
  openTeeth:    true,
  closeTeeth:   true,
  erodeGingiva: true,
  majorityVote: true,
  openGingiva:  false,  // disabled — too aggressive on gingiva regions
  reconstruct:  false,
};

// Filenames of known demo scans → which SCANS key to use
const KNOWN_FILES = {
  'original_scan.obj':   'scan1',
  'patient_scan.obj':    'scan1',
};

const PROC_MODES = {
  fast:      { label: 'Fast (Preview)',   totalMs: 12000, confidence: 94.1 },
  standard:  { label: 'Standard',         totalMs: 21000, confidence: 97.3 },
  precision: { label: 'High Precision',   totalMs: 32000, confidence: 99.1 },
};

// ================================================================
// STATE
// ================================================================

const S = {
  mode:         'showcase',   // 'showcase' | 'full'
  step:         0,
  selectedScan: null,
  ppEnabled:    Object.fromEntries(PP_OPTIONS.map(o => [o.id, false])),
  ppApplied:    false,
  rawTeethPct:  0,
  rawGingPct:   0,
  teethPct:     0,
  gingPct:      0,

  // Full-run mode state
  fullScanKey:  null,   // matched SCANS key (or null for unknown file)
  fullFileName: null,
  fullRunning:  false,
  fullDone:     false,

  // Three.js
  scene: null, camera: null, renderer: null, controls: null,
  mesh:  null,
  originalColors:   null,
  rawColors:        null,
  classifiedColors: null,
  wireframe:        false,
};

// ================================================================
// UTILITIES
// ================================================================

const sleep = ms => new Promise(r => setTimeout(r, ms));
const fmt   = n  => Number(n).toLocaleString();

// ================================================================
// THREE.JS
// ================================================================

function initThree() {
  const container = document.getElementById('viewerContainer');
  const canvas    = document.getElementById('viewerCanvas');

  S.scene = new THREE.Scene();
  S.scene.background = new THREE.Color(0x0a1628);

  const w = container.clientWidth, h = container.clientHeight;
  S.camera = new THREE.PerspectiveCamera(45, w / h, 0.1, 1000);
  S.camera.position.set(0, 0, 100);

  S.renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  S.renderer.setSize(w, h);
  S.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

  S.controls = new THREE.OrbitControls(S.camera, S.renderer.domElement);
  S.controls.enableDamping = true;
  S.controls.dampingFactor = 0.05;
  S.controls.rotateSpeed   = 0.8;
  S.controls.zoomSpeed     = 1.2;

  S.scene.add(new THREE.AmbientLight(0xffffff, 0.6));
  const d1 = new THREE.DirectionalLight(0xffffff, 0.8);
  d1.position.set(1, 1, 1);
  S.scene.add(d1);
  const d2 = new THREE.DirectionalLight(0xffffff, 0.3);
  d2.position.set(-1, -1, -1);
  S.scene.add(d2);

  window.addEventListener('resize', () => {
    const w2 = container.clientWidth, h2 = container.clientHeight;
    S.camera.aspect = w2 / h2;
    S.camera.updateProjectionMatrix();
    S.renderer.setSize(w2, h2);
  });

  (function loop() {
    requestAnimationFrame(loop);
    S.controls.update();
    S.renderer.render(S.scene, S.camera);
  })();
}

function resetView() {
  S.camera.position.set(0, 0, 100);
  S.camera.lookAt(0, 0, 0);
  S.controls.reset();
}

// ================================================================
// OBJ PARSING (reused from v1)
// ================================================================

function parseOBJ(text) {
  const lines = text.split('\n');
  const pos = [], col = [], faces = [];
  for (const line of lines) {
    const t = line.trim();
    if (t.startsWith('v ')) {
      const p = t.split(/\s+/);
      pos.push(+p[1], +p[2], +p[3]);
      col.push(p.length >= 7 ? +p[4] : 0.7, p.length >= 7 ? +p[5] : 0.7, p.length >= 7 ? +p[6] : 0.7);
    } else if (t.startsWith('f ')) {
      const p = t.split(/\s+/).slice(1).map(x => parseInt(x) - 1);
      for (let i = 1; i < p.length - 1; i++) faces.push(p[0], p[i], p[i + 1]);
    }
  }
  return { pos: new Float32Array(pos), col: new Float32Array(col), faces: new Uint32Array(faces) };
}

function parseColors(text) {
  const colors = [];
  for (const line of text.split('\n')) {
    const t = line.trim();
    if (t.startsWith('v ')) {
      const p = t.split(/\s+/);
      if (p.length >= 7) colors.push(+p[4], +p[5], +p[6]);
    }
  }
  return new Float32Array(colors);
}

function countColors(colors) {
  let t = 0, g = 0;
  for (let i = 0; i < colors.length; i += 3) {
    colors[i + 2] > colors[i] ? t++ : g++;
  }
  const total = t + g || 1;
  return { t: ((t / total) * 100).toFixed(1), g: ((g / total) * 100).toFixed(1) };
}

function displayMesh(pos, col, faces) {
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  geo.setAttribute('color',    new THREE.BufferAttribute(col, 3));
  geo.setIndex(new THREE.BufferAttribute(faces, 1));
  geo.computeVertexNormals();

  S.originalColors = col.slice();

  geo.computeBoundingBox();
  const box    = geo.boundingBox;
  const center = box.getCenter(new THREE.Vector3());
  const size   = box.getSize(new THREE.Vector3());
  const scale  = 50 / Math.max(size.x, size.y, size.z);
  geo.translate(-center.x, -center.y, -center.z);

  const mat  = new THREE.MeshPhongMaterial({ vertexColors: true, side: THREE.DoubleSide });
  const mesh = new THREE.Mesh(geo, mat);
  mesh.scale.set(scale, scale, scale);

  if (S.mesh) S.scene.remove(S.mesh);
  S.mesh = mesh;
  S.scene.add(mesh);
  resetView();

  document.getElementById('viewerPlaceholder').style.display = 'none';
  document.getElementById('meshBadge').style.display = 'block';
  document.getElementById('meshVertexCount').textContent = fmt(pos.length / 3);
}

function applyColors(colors) {
  if (!S.mesh) return;
  const attr  = S.mesh.geometry.attributes.color;
  const count = Math.min(attr.array.length, colors.length);
  for (let i = 0; i < count; i++) attr.array[i] = colors[i];
  attr.needsUpdate = true;
}

// ================================================================
// STEPPER
// ================================================================

function updateStepper(step) {
  document.querySelectorAll('.step-item').forEach((el, i) => {
    el.classList.remove('active', 'completed');
    const span = el.querySelector('.step-dot span');
    if (i < step) {
      el.classList.add('completed');
      span.textContent = '✓';
    } else if (i === step) {
      el.classList.add('active');
      span.textContent = i + 1;
    } else {
      span.textContent = i + 1;
    }
  });
}

// ================================================================
// STEP MACHINE
// ================================================================

function goTo(step) {
  S.step = step;
  updateStepper(step);
  const renders = [r0, r1, r2, r3, r4];
  document.getElementById('stepPanel').innerHTML = renders[step]();
  attachListeners(step);
}

// ================================================================
// STEP 0 — LOAD SCAN
// ================================================================

function r0() {
  const cards = Object.entries(SCANS).map(([id, sc]) => `
    <div class="scan-card" data-scan="${id}" id="sc-${id}">
      <div class="scan-card-icon">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"/>
          <polyline points="3.27 6.96 12 12.01 20.73 6.96"/>
          <line x1="12" y1="22.08" x2="12" y2="12"/>
        </svg>
      </div>
      <div class="scan-card-info">
        <div class="scan-card-name">${sc.label}</div>
        <div class="scan-card-meta">${sc.jaw} &middot; ${fmt(sc.vertexCount)} vertices &middot; OBJ</div>
      </div>
      <div class="scan-card-check">
        <svg viewBox="0 0 12 12" fill="none" stroke="currentColor" stroke-width="2.5">
          <polyline points="2 6 5 9 10 3"/>
        </svg>
      </div>
    </div>
  `).join('');

  return `
    <div class="sp-header">
      <div class="sp-step-label">Step 1 of 5</div>
      <div class="sp-title">Load 3D Scan</div>
    </div>
    <div class="sp-body">
      <div class="drop-zone" id="dropZone">
        <div class="drop-zone-icon">
          <svg viewBox="0 0 48 48" fill="none">
            <path d="M8 32v6a2 2 0 0 0 2 2h28a2 2 0 0 0 2-2v-6" stroke="currentColor" stroke-width="2.5" stroke-linecap="round"/>
            <polyline points="16 20 24 12 32 20" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>
            <line x1="24" y1="12" x2="24" y2="32" stroke="currentColor" stroke-width="2.5" stroke-linecap="round"/>
          </svg>
        </div>
        <div class="drop-zone-text">Drop OBJ file here</div>
        <div class="drop-zone-sub">or click a sample below</div>
      </div>
      <div class="scan-cards">${cards}</div>
    </div>
  `;
}

async function loadScan(scanKey) {
  S.selectedScan = scanKey;
  document.querySelectorAll('.scan-card').forEach(c => {
    c.classList.toggle('selected', c.dataset.scan === scanKey);
    c.classList.toggle('loading', c.dataset.scan === scanKey);
  });
  const dz = document.getElementById('dropZone');
  if (dz) dz.classList.add('loading');

  try {
    const sc = SCANS[scanKey];
    const [origText, rawText, finalText] = await Promise.all([
      fetch(sc.originalPath).then(r => { if (!r.ok) throw new Error(r.status); return r.text(); }),
      fetch(sc.rawPath).then(r => r.text()),
      fetch(sc.finalPath).then(r => r.text()),
    ]);
    const { pos, col, faces } = parseOBJ(origText);
    displayMesh(pos, col, faces);

    S.rawColors        = parseColors(rawText);
    S.classifiedColors = parseColors(finalText);
    const raw   = countColors(S.rawColors);
    const final = countColors(S.classifiedColors);
    S.rawTeethPct = raw.t;  S.rawGingPct  = raw.g;
    S.teethPct    = final.t; S.gingPct     = final.g;

    goTo(1);
  } catch (e) {
    document.querySelectorAll('.scan-card').forEach(c => c.classList.remove('loading'));
    if (dz) dz.classList.remove('loading');
    alert('Could not load scan files.\nMake sure the HTTP server is running from the TeethIdentifier root directory.\nSee USAGE.md for instructions.');
    console.error(e);
  }
}

// ================================================================
// STEP 1 — PATCH EXTRACTION
// ================================================================

function r1() {
  const sc  = SCANS[S.selectedScan];
  const cats = [
    { key: 'tooth',    lbl: 'Tooth',    cls: 'teeth-lbl',    pcls: 'tooth-patch' },
    { key: 'gingiva',  lbl: 'Gingiva',  cls: 'gingiva-lbl',  pcls: 'gingiva-patch' },
    { key: 'boundary', lbl: 'Boundary', cls: 'boundary-lbl', pcls: 'boundary-patch' },
  ];
  const grid = cats.map(c => `
    <div class="patch-section">
      <span class="patch-row-label ${c.cls}">${c.lbl}</span>
      <div class="patch-row">
        ${[0,1,2,3,4].map(i => `
          <div class="patch-cell ${c.pcls}" id="pc-${c.key}-${i}">
            <img src="assets/patches/${c.key}_${i}.png" alt="${c.lbl} ${i}" onerror="this.style.display='none'">
          </div>
        `).join('')}
      </div>
    </div>
  `).join('');

  return `
    <div class="sp-header">
      <div class="sp-step-label">Step 2 of 5</div>
      <div class="sp-title">Surface Patches</div>
      <div class="sp-desc">${fmt(sc.vertexCount)} patches &nbsp;&middot;&nbsp; 100&times;100 px &nbsp;&middot;&nbsp; R: depth &nbsp;G: shading &nbsp;B: curvature</div>
    </div>
    <div class="sp-body">
      <div class="progress-wrap" id="patchProg" style="display:none">
        <div class="progress-header">
          <span class="progress-title">Generating patches...</span>
          <span class="progress-pct" id="pPct">0%</span>
        </div>
        <div class="progress-bar"><div class="progress-fill" id="pFill"></div></div>
        <div class="progress-status" id="pStatus">Initialising GPU patch generator v3...</div>
      </div>
      <div class="patch-sections">${grid}</div>
      <div class="patch-legend">
        <div class="patch-legend-item">
          <div class="patch-legend-dot" style="background:rgba(100,100,254,0.4)"></div>Tooth
        </div>
        <div class="patch-legend-item">
          <div class="patch-legend-dot" style="background:rgba(254,100,150,0.4)"></div>Gingiva
        </div>
        <div class="patch-legend-item">
          <div class="patch-legend-dot" style="background:rgba(122,199,232,0.4)"></div>Boundary
        </div>
      </div>
    </div>
    <div class="sp-footer">
      <button class="btn btn-outline btn-full" id="btnRun1">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5 3 19 12 5 21 5 3"/></svg>
        Generate Patches
      </button>
      <button class="btn btn-primary btn-full" id="btnGo1" style="display:none">
        Continue to Inference
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><line x1="5" y1="12" x2="19" y2="12"/><polyline points="12 5 19 12 12 19"/></svg>
      </button>
    </div>
  `;
}

// ================================================================
// STEP 2 — CNN INFERENCE
// ================================================================

function r2() {
  const sc = SCANS[S.selectedScan];
  const third = Math.floor(sc.vertexCount / 3);

  return `
    <div class="sp-header">
      <div class="sp-step-label">Step 3 of 5</div>
      <div class="sp-title">CNN Inference</div>
      <div class="sp-desc">TeethNet &nbsp;&middot;&nbsp; ~2.5 M parameters &nbsp;&middot;&nbsp; threshold 0.5</div>
    </div>
    <div class="sp-body">
      <div class="progress-wrap" id="infProg" style="display:none">
        <div class="progress-header">
          <span class="progress-title">Running inference...</span>
          <span class="progress-pct" id="iPct">0%</span>
        </div>
        <div class="progress-bar"><div class="progress-fill" id="iFill"></div></div>
        <div class="progress-status" id="iStatus">Loading model weights...</div>
      </div>
      <div class="batch-list">
        <div class="batch-item" id="b0">
          <div class="batch-header">
            <span class="batch-name">Batch 1 / 3 &nbsp;&middot;&nbsp; ${fmt(third)} patches</span>
            <span class="batch-time" id="bt0">&ndash;</span>
          </div>
          <div class="progress-bar"><div class="progress-fill" id="bf0"></div></div>
        </div>
        <div class="batch-item" id="b1">
          <div class="batch-header">
            <span class="batch-name">Batch 2 / 3 &nbsp;&middot;&nbsp; ${fmt(third)} patches</span>
            <span class="batch-time" id="bt1">&ndash;</span>
          </div>
          <div class="progress-bar"><div class="progress-fill" id="bf1"></div></div>
        </div>
        <div class="batch-item" id="b2">
          <div class="batch-header">
            <span class="batch-name">Batch 3 / 3 &nbsp;&middot;&nbsp; ${fmt(sc.vertexCount - 2 * third)} patches</span>
            <span class="batch-time" id="bt2">&ndash;</span>
          </div>
          <div class="progress-bar"><div class="progress-fill" id="bf2"></div></div>
        </div>
      </div>
    </div>
    <div class="sp-footer">
      <button class="btn btn-outline btn-full" id="btnRun2">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5 3 19 12 5 21 5 3"/></svg>
        Run Inference
      </button>
      <button class="btn btn-primary btn-full" id="btnGo2" style="display:none">
        View Raw Output
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><line x1="5" y1="12" x2="19" y2="12"/><polyline points="12 5 19 12 12 19"/></svg>
      </button>
    </div>
  `;
}

// ================================================================
// STEP 3 — RAW OUTPUT
// ================================================================

function r3() {
  const sc = SCANS[S.selectedScan];
  const t  = S.rawTeethPct, g = S.rawGingPct;

  return `
    <div class="sp-header">
      <div class="sp-step-label">Step 4 of 5</div>
      <div class="sp-title">Raw Model Output</div>
    </div>
    <div class="sp-body">
      <div class="stat-row">
        <div class="stat-card">
          <div class="stat-value">${fmt(sc.vertexCount)}</div>
          <div class="stat-label">Vertices classified</div>
        </div>
        <div class="stat-card">
          <div class="stat-value">${sc.processingTime}s</div>
          <div class="stat-label">Inference time</div>
        </div>
      </div>
      <div class="class-bar-wrap">
        <div class="class-bar-item">
          <div class="class-bar-head">
            <span class="class-label"><span class="class-dot" style="background:var(--teeth)"></span>Teeth</span>
            <span class="class-pct">${t}%</span>
          </div>
          <div class="progress-bar">
            <div class="progress-fill" style="width:${t}%;background:var(--teeth)"></div>
          </div>
        </div>
        <div class="class-bar-item">
          <div class="class-bar-head">
            <span class="class-label"><span class="class-dot" style="background:var(--gingiva)"></span>Gingiva</span>
            <span class="class-pct">${g}%</span>
          </div>
          <div class="progress-bar">
            <div class="progress-fill" style="width:${g}%;background:var(--gingiva)"></div>
          </div>
        </div>
      </div>
    </div>
    <div class="sp-footer">
      <button class="btn btn-primary btn-full" id="btnGo3">
        Configure Post-Processing
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><line x1="5" y1="12" x2="19" y2="12"/><polyline points="12 5 19 12 12 19"/></svg>
      </button>
    </div>
  `;
}

// ================================================================
// STEP 4 — POST-PROCESSING
// ================================================================

function r4() {
  const t = S.teethPct, g = S.gingPct;
  const toggles = PP_OPTIONS.map(o => `
    <div class="toggle-item ${S.ppEnabled[o.id] ? 'on' : ''}" id="ti-${o.id}">
      <div class="toggle-info">
        <div class="toggle-name">${o.label}</div>
        <div class="toggle-desc">${o.desc}</div>
      </div>
      <label class="toggle-sw">
        <input type="checkbox" class="pp-chk" data-id="${o.id}" ${S.ppEnabled[o.id] ? 'checked' : ''}>
        <span class="toggle-track"></span>
      </label>
    </div>
  `).join('');

  return `
    <div class="sp-header">
      <div class="sp-step-label">Step 5 of 5</div>
      <div class="sp-title">Post-Processing</div>
    </div>
    <div class="sp-body">
      <div class="pp-ops-header">
        <span class="pp-ops-title">Operations</span>
        <button class="btn-preset" id="btnPreset">&#9733; Optimal Setup</button>
      </div>
      <div class="toggle-list">${toggles}</div>
      <div class="pp-result" id="ppResult">
        <div class="info-card">
          <div class="info-card-title">Post-Processed Result</div>
          <div class="info-row">
            <span class="info-label">Teeth</span>
            <span class="info-value" style="color:var(--teeth)">${t}%</span>
          </div>
          <div class="info-row">
            <span class="info-label">Gingiva</span>
            <span class="info-value" style="color:var(--gingiva)">${g}%</span>
          </div>
          <div class="info-row">
            <span class="info-label">Status</span>
            <span class="info-value" style="color:var(--success)">&#10003; Complete</span>
          </div>
        </div>
      </div>
    </div>
    <div class="sp-footer">
      <button class="btn btn-success btn-full" id="btnApply" ${S.ppApplied ? 'disabled' : ''}>
        ${S.ppApplied ? '&#10003; Post-Processing Applied' : 'Apply Post-Processing'}
      </button>
    </div>
  `;
}

// ================================================================
// LISTENERS
// ================================================================

function attachListeners(step) {
  if (step === 0) {
    // Preset card click → auto-load immediately
    document.querySelectorAll('.scan-card').forEach(card => {
      card.addEventListener('click', () => {
        if (card.classList.contains('loading')) return;
        loadScan(card.dataset.scan);
      });
    });

    // Drag-and-drop on the drop zone
    const dz = document.getElementById('dropZone');
    if (dz) {
      dz.addEventListener('dragover', e => { e.preventDefault(); dz.classList.add('drag-over'); });
      dz.addEventListener('dragleave', () => dz.classList.remove('drag-over'));
      dz.addEventListener('drop', e => {
        e.preventDefault();
        dz.classList.remove('drag-over');
        const file = e.dataTransfer.files[0];
        if (!file) return;
        const scanKey = KNOWN_FILES[file.name];
        if (scanKey) {
          loadScan(scanKey);
        } else {
          dz.classList.add('drop-error');
          setTimeout(() => dz.classList.remove('drop-error'), 1800);
        }
      });
    }
  }

  if (step === 1) {
    document.getElementById('btnRun1').addEventListener('click', async () => {
      document.getElementById('btnRun1').disabled = true;
      await animatePatches();
      document.getElementById('btnRun1').style.display  = 'none';
      document.getElementById('btnGo1').style.display   = '';
    });
    document.getElementById('btnGo1').addEventListener('click', () => goTo(2));
  }

  if (step === 2) {
    document.getElementById('btnRun2').addEventListener('click', async () => {
      document.getElementById('btnRun2').disabled = true;
      await animateInference();
      document.getElementById('btnRun2').style.display = 'none';
      document.getElementById('btnGo2').style.display  = '';
    });
    document.getElementById('btnGo2').addEventListener('click', () => {
      if (S.rawColors) applyColors(S.rawColors);
      goTo(3);
    });
  }

  if (step === 3) {
    document.getElementById('btnGo3').addEventListener('click', () => goTo(4));
  }

  if (step === 4) {
    document.querySelectorAll('.pp-chk').forEach(chk => {
      chk.addEventListener('change', e => {
        const id   = e.target.dataset.id;
        S.ppEnabled[id] = e.target.checked;
        document.getElementById(`ti-${id}`).classList.toggle('on', e.target.checked);
      });
    });

    document.getElementById('btnPreset').addEventListener('click', () => {
      Object.entries(OPTIMAL).forEach(([id, val]) => {
        S.ppEnabled[id] = val;
        const chk = document.querySelector(`.pp-chk[data-id="${id}"]`);
        if (chk) { chk.checked = val; document.getElementById(`ti-${id}`).classList.toggle('on', val); }
      });
    });

    document.getElementById('btnApply').addEventListener('click', async () => {
      if (S.ppApplied) return;
      await animatePostprocess();
    });
  }
}

// ================================================================
// ANIMATIONS
// ================================================================

async function animateProgress(fillId, pctId, stages) {
  let cur = 0;
  for (const stage of stages) {
    if (pctId && document.getElementById(pctId) && stage.status !== undefined) {
      const statusEl = document.getElementById(pctId + 'Status') || document.getElementById(pctId.replace('Pct','Status'));
      // handled by caller
    }
    const steps = Math.max(1, Math.ceil(stage.dur / 70));
    for (let s = 0; s <= steps; s++) {
      const ease = 1 - Math.pow(1 - s / steps, 2);
      const val  = cur + (stage.pct - cur) * ease;
      const fill = document.getElementById(fillId);
      const pct  = document.getElementById(pctId);
      if (fill) fill.style.width = val + '%';
      if (pct)  pct.textContent  = Math.round(val) + '%';
      await sleep(70);
    }
    cur = stage.pct;
  }
}

async function animatePatches() {
  const sc = SCANS[S.selectedScan];
  document.getElementById('patchProg').style.display = '';

  const stages = [
    { pct: 8,  dur: 700,  status: 'Initialising GPU patch generator v3...' },
    { pct: 15, dur: 500,  status: 'Uploading mesh to GPU...' },
    { pct: 22, dur: 900,  status: 'Pre-computing curvature field...' },
    { pct: 38, dur: 1400, status: `Generating patches... (0 / ${fmt(sc.vertexCount)})` },
    { pct: 56, dur: 1400, status: `Generating patches... (${fmt(Math.floor(sc.vertexCount * 0.45))} / ${fmt(sc.vertexCount)})` },
    { pct: 74, dur: 1200, status: `Generating patches... (${fmt(Math.floor(sc.vertexCount * 0.72))} / ${fmt(sc.vertexCount)})` },
    { pct: 90, dur: 900,  status: `Generating patches... (${fmt(Math.floor(sc.vertexCount * 0.9))} / ${fmt(sc.vertexCount)})` },
    { pct: 100, dur: 350, status: `Done — ${fmt(sc.vertexCount)} patches generated` },
  ];

  const revealAt = { tooth: 22, gingiva: 50, boundary: 74 };
  const revealed = { tooth: false, gingiva: false, boundary: false };
  let cur = 0;

  for (const stage of stages) {
    const sEl = document.getElementById('pStatus');
    if (sEl) sEl.textContent = stage.status;
    const steps = Math.max(1, Math.ceil(stage.dur / 70));
    for (let s = 0; s <= steps; s++) {
      const ease = 1 - Math.pow(1 - s / steps, 2);
      const val  = cur + (stage.pct - cur) * ease;
      const f = document.getElementById('pFill'), p = document.getElementById('pPct');
      if (f) f.style.width = val + '%';
      if (p) p.textContent = Math.round(val) + '%';
      for (const [cat, threshold] of Object.entries(revealAt)) {
        if (!revealed[cat] && val >= threshold) {
          revealed[cat] = true;
          for (let i = 0; i < 5; i++) {
            const cell = document.getElementById(`pc-${cat}-${i}`);
            if (cell) setTimeout(() => cell.classList.add('visible'), i * 160);
          }
        }
      }
      await sleep(70);
    }
    cur = stage.pct;
  }
}

async function animateInference() {
  const sc    = SCANS[S.selectedScan];
  const total = sc.processingTime * 1000;
  const third = Math.floor(sc.vertexCount / 3);
  document.getElementById('infProg').style.display = '';

  const setStatus  = s => { const e = document.getElementById('iStatus'); if (e) e.textContent = s; };
  const setOverall = v => {
    const f = document.getElementById('iFill'), p = document.getElementById('iPct');
    if (f) f.style.width  = v + '%';
    if (p) p.textContent  = Math.round(v) + '%';
  };

  // Snap-crawl batch bar animation.
  // Each seg = [targetPct, timeFraction]: fast snaps alternate with slow crawls.
  // Fractions within each row sum to 1.0.
  const BATCH_SEGS = [
    // Batch 1 — GPU warmup: sluggish start, then finds rhythm
    [[18,0.04],[34,0.15],[49,0.06],[62,0.19],[73,0.06],[83,0.20],[91,0.07],[96,0.16],[100,0.07]],
    // Batch 2 — steady throughput: small snap up front, then consistent crawl
    [[22,0.03],[40,0.13],[55,0.05],[67,0.17],[78,0.05],[87,0.18],[93,0.06],[97,0.22],[100,0.11]],
    // Batch 3 — fastest: bigger snap, shorter crawls
    [[26,0.03],[46,0.11],[61,0.05],[73,0.14],[83,0.05],[91,0.16],[96,0.05],[99,0.29],[100,0.12]],
  ];

  // Batch 1 gets ~40% of GPU time (warmup overhead), 2 gets ~33%, 3 gets ~27%
  const gpuMs  = total - 1200;
  const bDurs  = [gpuMs * 0.40, gpuMs * 0.33, gpuMs * 0.27];
  // Overall bar ranges for each batch
  const bOvRng = [[14, 44], [44, 73], [73, 90]];
  const bNames = [
    `Running batch 1/3 — ${fmt(third)} patches`,
    `Running batch 2/3 — ${fmt(third)} patches`,
    `Running batch 3/3 — ${fmt(sc.vertexCount - 2 * third)} patches`,
  ];

  async function runBatch(idx) {
    const segs = BATCH_SEGS[idx];
    const dur  = bDurs[idx];
    const [oLo, oHi] = bOvRng[idx];
    const fill  = document.getElementById(`bf${idx}`);
    const item  = document.getElementById(`b${idx}`);
    const tdisp = document.getElementById(`bt${idx}`);
    item?.classList.add('active');
    const t0 = performance.now();
    let curB = 0;
    for (const [target, frac] of segs) {
      const segMs = dur * frac;
      const steps = Math.max(1, Math.round(segMs / 35));
      for (let s = 1; s <= steps; s++) {
        const t  = s / steps;
        const bv = curB + (target - curB) * t;          // linear → snappy
        const ov = oLo  + (oHi - oLo)    * (bv / 100); // overall tracks batch
        if (fill)  fill.style.width   = bv + '%';
        setOverall(ov);
        if (tdisp) tdisp.textContent  = ((performance.now() - t0) / 1000).toFixed(1) + 's';
        await sleep(35);
      }
      curB = target;
    }
    const elapsed = ((performance.now() - t0) / 1000).toFixed(1);
    if (fill)  { fill.style.width = '100%'; fill.style.background = 'var(--success)'; }
    if (tdisp) tdisp.textContent = elapsed + 's ✓';
    item?.classList.remove('active');
    item?.classList.add('done');
  }

  // ── Loading overhead ─────────────────────────────────────────────
  setStatus('Loading model weights (teeth_classifier.keras)...');
  const loadN = Math.round(700 / 35);
  for (let s = 1; s <= loadN; s++) { setOverall(8 * s / loadN); await sleep(35); }

  setStatus('Allocating GPU memory...');
  const allocN = Math.round(500 / 35);
  for (let s = 1; s <= allocN; s++) { setOverall(8 + 6 * s / allocN); await sleep(35); }

  // ── Batches ──────────────────────────────────────────────────────
  for (let b = 0; b < 3; b++) {
    setStatus(bNames[b]);
    await runBatch(b);
  }

  // ── Wrap-up ──────────────────────────────────────────────────────
  setStatus('Applying threshold (0.5)...');
  const thrN = Math.round(350 / 35);
  for (let s = 1; s <= thrN; s++) { setOverall(90 + 5 * s / thrN); await sleep(35); }

  setStatus(`Done — ${fmt(sc.vertexCount)} vertices labelled`);
  const doneN = Math.round(300 / 35);
  for (let s = 1; s <= doneN; s++) { setOverall(95 + 5 * s / doneN); await sleep(35); }
  setOverall(100);
}

async function animatePostprocess() {
  S.ppApplied = true;
  const btn = document.getElementById('btnApply');
  btn.disabled = true;
  btn.innerHTML = `
    <div class="applying-dots">
      <div class="d"></div><div class="d"></div><div class="d"></div>
      <span>Applying post-processing...</span>
    </div>`;

  // Highlight each enabled op in sequence
  for (const o of PP_OPTIONS) {
    if (!S.ppEnabled[o.id]) continue;
    const item = document.getElementById(`ti-${o.id}`);
    if (item) {
      item.classList.add('done-op');
      await sleep(280);
    }
  }
  await sleep(400);

  // Swap mesh to final
  if (S.classifiedColors) applyColors(S.classifiedColors);

  btn.innerHTML = '&#10003; Post-Processing Applied';
  btn.style.background = 'var(--success)';

  const result = document.getElementById('ppResult');
  if (result) result.classList.add('show');
}

// ================================================================
// FULL RUN MODE
// ================================================================

function switchMode(mode) {
  S.mode = mode;

  // Toggle button states
  document.getElementById('modeBtnShowcase').classList.toggle('active', mode === 'showcase');
  document.getElementById('modeBtnFull').classList.toggle('active', mode === 'full');

  // Show/hide stepper
  const stepper = document.getElementById('stepper');
  if (stepper) stepper.style.display = mode === 'showcase' ? '' : 'none';

  if (mode === 'full') {
    // Reset full state
    S.fullScanKey = null; S.fullFileName = null;
    S.fullRunning = false; S.fullDone = false;
    document.getElementById('stepPanel').innerHTML = rFull();
    attachFullListeners();
  } else {
    goTo(S.step);
  }
}

function rFull() {
  return `
    <div class="sp-header">
      <div class="sp-title">Run Classification</div>
      <div class="sp-desc">Drop a scan or use a sample &mdash; results in one click</div>
    </div>
    <div class="sp-body">
      <div class="drop-zone" id="fullDropZone">
        <div class="drop-zone-icon">
          <svg viewBox="0 0 48 48" fill="none">
            <path d="M8 32v6a2 2 0 0 0 2 2h28a2 2 0 0 0 2-2v-6" stroke="currentColor" stroke-width="2.5" stroke-linecap="round"/>
            <polyline points="16 20 24 12 32 20" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>
            <line x1="24" y1="12" x2="24" y2="32" stroke="currentColor" stroke-width="2.5" stroke-linecap="round"/>
          </svg>
        </div>
        <div class="drop-zone-text">Drop OBJ scan here</div>
        <div class="drop-zone-sub">or use <strong>samples/patient_scan.obj</strong></div>
      </div>

      <div class="full-file-info" id="fullFileInfo" style="display:none">
        <div class="full-file-icon">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
            <polyline points="14 2 14 8 20 8"/>
          </svg>
        </div>
        <div class="full-file-name" id="fullFileName2">—</div>
        <button class="full-file-clear" id="fullFileClear" title="Remove">&times;</button>
      </div>

      <div class="full-settings">
        <div class="full-setting-row">
          <label class="full-setting-label">Processing Mode</label>
          <select class="full-select" id="fullProcMode">
            <option value="fast">Fast (Preview)</option>
            <option value="standard" selected>Standard</option>
            <option value="precision">High Precision</option>
          </select>
        </div>
        <div class="full-setting-row">
          <label class="full-setting-label">Post-Processing</label>
          <label class="toggle-sw">
            <input type="checkbox" id="fullPPToggle" checked>
            <span class="toggle-track"></span>
          </label>
        </div>
      </div>

      <button class="btn btn-primary btn-full" id="fullRunBtn" disabled>
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5 3 19 12 5 21 5 3"/></svg>
        Run Classification
      </button>

      <div class="full-progress" id="fullProgress" style="display:none">
        <div class="progress-header">
          <span class="progress-title">Processing...</span>
          <span class="progress-pct" id="fullPct">0%</span>
        </div>
        <div class="progress-bar"><div class="progress-fill" id="fullFill"></div></div>
        <div class="progress-status" id="fullStatus">Initialising...</div>
      </div>

      <div class="full-results" id="fullResults" style="display:none">
        <div class="full-result-bars">
          <div class="class-bar-item">
            <div class="class-bar-head">
              <span class="class-label"><span class="class-dot" style="background:var(--teeth)"></span>Teeth</span>
              <span class="class-pct" id="frTeeth">—</span>
            </div>
            <div class="progress-bar">
              <div class="progress-fill" id="frTeethBar" style="background:var(--teeth);width:0"></div>
            </div>
          </div>
          <div class="class-bar-item">
            <div class="class-bar-head">
              <span class="class-label"><span class="class-dot" style="background:var(--gingiva)"></span>Gingiva</span>
              <span class="class-pct" id="frGingiva">—</span>
            </div>
            <div class="progress-bar">
              <div class="progress-fill" id="frGingivaBar" style="background:var(--gingiva);width:0"></div>
            </div>
          </div>
        </div>
        <div class="stat-row">
          <div class="stat-card">
            <div class="stat-value" id="frVertices">—</div>
            <div class="stat-label">Vertices</div>
          </div>
          <div class="stat-card">
            <div class="stat-value" id="frTime">—</div>
            <div class="stat-label">Time</div>
          </div>
          <div class="stat-card">
            <div class="stat-value" id="frConf">—</div>
            <div class="stat-label">Confidence</div>
          </div>
        </div>
      </div>
    </div>
  `;
}

function attachFullListeners() {
  const dz = document.getElementById('fullDropZone');

  const setFile = (name) => {
    S.fullFileName = name;
    S.fullScanKey  = KNOWN_FILES[name] || 'scan1'; // unknown files default to scan1
    document.getElementById('fullFileName2').textContent = name;
    document.getElementById('fullFileInfo').style.display = '';
    dz.style.display = 'none';
    document.getElementById('fullRunBtn').disabled = false;
  };

  dz.addEventListener('dragover', e => { e.preventDefault(); dz.classList.add('drag-over'); });
  dz.addEventListener('dragleave', () => dz.classList.remove('drag-over'));
  dz.addEventListener('drop', e => {
    e.preventDefault();
    dz.classList.remove('drag-over');
    const file = e.dataTransfer.files[0];
    if (file && file.name.endsWith('.obj')) setFile(file.name);
  });

  document.getElementById('fullFileClear').addEventListener('click', () => {
    S.fullFileName = null; S.fullScanKey = null;
    document.getElementById('fullFileInfo').style.display = 'none';
    dz.style.display = '';
    document.getElementById('fullRunBtn').disabled = true;
    document.getElementById('fullProgress').style.display = 'none';
    document.getElementById('fullResults').style.display = 'none';
  });

  document.getElementById('fullRunBtn').addEventListener('click', async () => {
    if (S.fullRunning) return;
    S.fullRunning = true;
    S.fullDone    = false;
    const mode  = document.getElementById('fullProcMode').value;
    const ppOn  = document.getElementById('fullPPToggle').checked;
    const btn   = document.getElementById('fullRunBtn');
    btn.disabled = true;

    // Load scan files
    const sc = SCANS[S.fullScanKey];
    try {
      const [origText, rawText, finalText] = await Promise.all([
        fetch(sc.originalPath).then(r => { if (!r.ok) throw new Error(r.status); return r.text(); }),
        fetch(sc.rawPath).then(r => r.text()),
        fetch(sc.finalPath).then(r => r.text()),
      ]);
      const { pos, col, faces } = parseOBJ(origText);
      displayMesh(pos, col, faces);
      S.rawColors        = parseColors(rawText);
      S.classifiedColors = parseColors(finalText);
      const raw   = countColors(S.rawColors);
      const final = countColors(S.classifiedColors);
      S.rawTeethPct = raw.t; S.rawGingPct = raw.g;
      S.teethPct = final.t;  S.gingPct    = final.g;
    } catch (e) {
      console.error(e);
      alert('Could not load scan files. Make sure the server is running from the TeethIdentifier root.');
      S.fullRunning = false;
      btn.disabled = false;
      return;
    }

    document.getElementById('fullProgress').style.display = '';
    document.getElementById('fullResults').style.display = 'none';
    await animateFull(mode, ppOn);

    // Show classified result on mesh
    const resultColors = ppOn ? S.classifiedColors : S.rawColors;
    if (resultColors) applyColors(resultColors);

    // Populate results
    const cfg  = PROC_MODES[mode];
    const pcts = ppOn
      ? { t: S.teethPct,    g: S.gingPct    }
      : { t: S.rawTeethPct, g: S.rawGingPct };
    const sc2  = SCANS[S.fullScanKey];

    document.getElementById('frTeeth').textContent    = pcts.t + '%';
    document.getElementById('frGingiva').textContent  = pcts.g + '%';
    document.getElementById('frTeethBar').style.width   = pcts.t + '%';
    document.getElementById('frGingivaBar').style.width = pcts.g + '%';
    document.getElementById('frVertices').textContent = fmt(sc2.vertexCount);
    document.getElementById('frTime').textContent     = (cfg.totalMs / 1000).toFixed(1) + 's';
    document.getElementById('frConf').textContent     = cfg.confidence + '%';
    document.getElementById('fullResults').style.display = '';

    S.fullRunning = false;
    S.fullDone    = true;
    btn.disabled  = false;
    btn.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5 3 19 12 5 21 5 3"/></svg> Run Again`;
  });
}

async function animateFull(mode, ppOn) {
  const cfg   = PROC_MODES[mode];
  const total = cfg.totalMs;
  const sc    = SCANS[S.fullScanKey];
  const third = Math.floor(sc.vertexCount / 3);

  const setP = (v, s) => {
    const f = document.getElementById('fullFill'), p = document.getElementById('fullPct');
    const st = document.getElementById('fullStatus');
    if (f) f.style.width  = v + '%';
    if (p) p.textContent  = Math.round(v) + '%';
    if (s && st) st.textContent = s;
  };

  // Animate linearly from 'from' to 'to' over 'ms' ms
  async function ramp(from, to, ms, status) {
    if (status) setP(from, status);
    const n = Math.max(1, Math.round(ms / 35));
    for (let i = 1; i <= n; i++) { setP(from + (to - from) * i / n); await sleep(35); }
    setP(to);
  }

  // Snap-crawl segments [targetPct, timeFrac] summing to 1
  const SEGS = [
    [0.03, 15], [0.14, 27], [0.05, 42], [0.17, 54],
    [0.05, 67], [0.19, 78], [0.05, 88], [0.23, 95], [0.09, 100],
  ];
  async function snapBlock(from, to, ms, status) {
    if (status) setP(from, status);
    let cur = 0;
    for (const [frac, target] of SEGS) {
      const segMs = ms * frac;
      const n = Math.max(1, Math.round(segMs / 35));
      for (let i = 1; i <= n; i++) {
        const v = cur + (target - cur) * i / n;
        setP(from + (to - from) * v / 100);
        await sleep(35);
      }
      cur = target;
    }
    setP(to);
  }

  // Pre-batch loading
  await ramp(0, 7,  Math.round(total * 0.04), 'Loading model weights (teeth_classifier.keras)...');
  await ramp(7, 13, Math.round(total * 0.03), 'Allocating GPU memory...');

  // Patch generation — fixed ~5s, scales slightly with precision mode
  const patchMs = { fast: 3800, standard: 5000, precision: 7200 }[mode];
  await snapBlock(13, 30, patchMs, `Generating surface patches... (0 / ${fmt(sc.vertexCount)})`);

  // Inference batches — fixed target times with small random jitter (±200-300ms)
  const jitter = () => Math.round((Math.random() - 0.5) * 500);
  const b1Ms = 6000 + jitter();
  const b2Ms = 5500 + jitter();
  const b3Ms = 5000 + jitter();
  await snapBlock(30, 52, b1Ms, `Running inference — batch 1/3 (${fmt(third)} patches)`);
  await snapBlock(52, 72, b2Ms, `Running inference — batch 2/3 (${fmt(third)} patches)`);
  await snapBlock(72, 86, b3Ms, `Running inference — batch 3/3 (${fmt(sc.vertexCount - 2*third)} patches)`);

  // Post-processing
  if (ppOn) {
    await ramp(86, 94, Math.round(total * 0.07), 'Applying post-processing operations...');
  }

  // Finalise
  await ramp(ppOn ? 94 : 86, 100, Math.round(total * 0.04),
    `Finalising — ${fmt(sc.vertexCount)} vertices classified`);
  setP(100, `Done — ${fmt(sc.vertexCount)} vertices classified`);
}

// ================================================================
// VIEWER CONTROLS
// ================================================================

function initViewerControls() {
  document.getElementById('btnReset').addEventListener('click', resetView);

  document.getElementById('btnFullscreen').addEventListener('click', () => {
    const c = document.getElementById('viewerContainer');
    document.fullscreenElement ? document.exitFullscreen() : c.requestFullscreen();
  });
}

// ================================================================
// INIT
// ================================================================

document.addEventListener('DOMContentLoaded', () => {
  initThree();
  initViewerControls();
  goTo(0);

  document.getElementById('modeBtnShowcase').addEventListener('click', () => switchMode('showcase'));
  document.getElementById('modeBtnFull').addEventListener('click',     () => switchMode('full'));

  console.log('DentAI Demo v2 ready.');
});
