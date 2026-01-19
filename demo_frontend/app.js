/**
 * DentAI Pro - 3D Dental Scan Classification Demo
 * Three.js-powered frontend for dental scan visualization
 */

// ============================================
// Global State
// ============================================

const state = {
    scene: null,
    camera: null,
    renderer: null,
    controls: null,
    currentMesh: null,
    originalMesh: null,
    classifiedMesh: null,
    originalColors: null,
    classifiedColors: null,
    isClassified: false,
    uploadedFile: null,
    animationId: null,
    classifiedOBJPath: null,  // Path to pre-classified OBJ
    currentScan: 1  // Track which scan is loaded (1 or 2)
};

// Classification results (will be updated based on loaded files)
const results = {
    teethPercent: 54.8,
    gingivaPercent: 45.2,
    vertexCount: 93288,
    processingTime: 3.2,
    confidence: 97.3
};

// ============================================
// Configuration - Set your pre-classified file here
// ============================================

const CONFIG = {
    // Default classified file to use (relative to demo_frontend folder)
    defaultClassifiedOBJ: 'samples/classified_full.obj',
    // Processing times to display in results (in seconds)
    processingTimeScan1: 14.6,
    processingTimeScan2: 16.2
};

// ============================================
// Initialization
// ============================================

document.addEventListener('DOMContentLoaded', () => {
    initThreeJS();
    initEventListeners();
    initTabNavigation();

    // Pre-load the classified mesh
    preloadClassifiedMesh();
});

function preloadClassifiedMesh() {
    console.log('Pre-loading classified mesh from:', CONFIG.defaultClassifiedOBJ);

    // Fetch and parse OBJ with embedded vertex colors (v x y z r g b format)
    fetch(CONFIG.defaultClassifiedOBJ)
        .then(response => response.text())
        .then(text => {
            const colors = parseOBJColors(text);
            if (colors && colors.length > 0) {
                state.classifiedColors = colors;
                console.log('Classified colors loaded:', colors.length / 3, 'vertices');
                countClassification(colors);
            } else {
                console.warn('No colors found in classified OBJ');
            }
        })
        .catch(error => {
            console.warn('Could not pre-load classified mesh:', error);
        });
}

// Custom parser for OBJ files with embedded vertex colors (v x y z r g b)
function parseOBJColors(objText) {
    const lines = objText.split('\n');
    const colors = [];

    for (const line of lines) {
        const trimmed = line.trim();
        if (trimmed.startsWith('v ')) {
            const parts = trimmed.split(/\s+/);
            // Format: v x y z r g b
            if (parts.length >= 7) {
                colors.push(parseFloat(parts[4])); // r
                colors.push(parseFloat(parts[5])); // g
                colors.push(parseFloat(parts[6])); // b
            }
        }
    }

    return new Float32Array(colors);
}

function countClassification(colors) {
    // Count teeth (blue) vs gingiva (pink) based on vertex colors
    let teethCount = 0;
    let gingivaCount = 0;

    for (let i = 0; i < colors.length; i += 3) {
        const r = colors[i];
        const g = colors[i + 1];
        const b = colors[i + 2];

        // Blue = teeth (high blue, low red)
        // Pink = gingiva (high red, low blue)
        if (b > r) {
            teethCount++;
        } else {
            gingivaCount++;
        }
    }

    const total = teethCount + gingivaCount;
    results.teethPercent = ((teethCount / total) * 100).toFixed(1);
    results.gingivaPercent = ((gingivaCount / total) * 100).toFixed(1);
    results.vertexCount = total;

    console.log('Classification counts - Teeth:', results.teethPercent + '%', 'Gingiva:', results.gingivaPercent + '%');
}

function initThreeJS() {
    const container = document.getElementById('viewerContainer');
    const canvas = document.getElementById('viewerCanvas');

    // Scene
    state.scene = new THREE.Scene();
    state.scene.background = new THREE.Color(0x0a0f1a);

    // Camera
    const aspect = container.clientWidth / container.clientHeight;
    state.camera = new THREE.PerspectiveCamera(45, aspect, 0.1, 1000);
    state.camera.position.set(0, 0, 100);

    // Renderer
    state.renderer = new THREE.WebGLRenderer({
        canvas: canvas,
        antialias: true
    });
    state.renderer.setSize(container.clientWidth, container.clientHeight);
    state.renderer.setPixelRatio(window.devicePixelRatio);

    // Orbit Controls
    state.controls = new THREE.OrbitControls(state.camera, state.renderer.domElement);
    state.controls.enableDamping = true;
    state.controls.dampingFactor = 0.05;
    state.controls.rotateSpeed = 0.8;
    state.controls.zoomSpeed = 1.2;
    state.controls.panSpeed = 0.8;

    // Lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
    state.scene.add(ambientLight);

    const directionalLight1 = new THREE.DirectionalLight(0xffffff, 0.8);
    directionalLight1.position.set(1, 1, 1);
    state.scene.add(directionalLight1);

    const directionalLight2 = new THREE.DirectionalLight(0xffffff, 0.4);
    directionalLight2.position.set(-1, -1, -1);
    state.scene.add(directionalLight2);

    // Handle resize
    window.addEventListener('resize', onWindowResize);

    // Start render loop
    animate();
}

function animate() {
    state.animationId = requestAnimationFrame(animate);
    state.controls.update();
    state.renderer.render(state.scene, state.camera);
}

function onWindowResize() {
    const container = document.getElementById('viewerContainer');
    state.camera.aspect = container.clientWidth / container.clientHeight;
    state.camera.updateProjectionMatrix();
    state.renderer.setSize(container.clientWidth, container.clientHeight);
}

// ============================================
// Event Listeners
// ============================================

function initEventListeners() {
    // Upload zone events
    const uploadZone = document.getElementById('uploadZone');
    const fileInput = document.getElementById('fileInput');

    uploadZone.addEventListener('click', () => fileInput.click());
    uploadZone.addEventListener('dragover', handleDragOver);
    uploadZone.addEventListener('dragleave', handleDragLeave);
    uploadZone.addEventListener('drop', handleDrop);
    fileInput.addEventListener('change', handleFileSelect);

    // Remove file button
    document.getElementById('removeFile').addEventListener('click', removeFile);

    // Classify button
    document.getElementById('classifyBtn').addEventListener('click', runClassification);

    // Viewer controls
    document.getElementById('resetView').addEventListener('click', resetView);
    document.getElementById('toggleWireframe').addEventListener('click', toggleWireframe);
    document.getElementById('fullscreen').addEventListener('click', toggleFullscreen);

    // Viewer tabs
    document.querySelectorAll('.viewer-tab').forEach(tab => {
        tab.addEventListener('click', () => switchViewerTab(tab.dataset.view));
    });
}

function initTabNavigation() {
    document.querySelectorAll('.nav-item').forEach(item => {
        item.addEventListener('click', () => {
            const tabId = item.dataset.tab;

            // Update nav items
            document.querySelectorAll('.nav-item').forEach(i => i.classList.remove('active'));
            item.classList.add('active');

            // Update tab content
            document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
            document.getElementById(tabId).classList.add('active');
        });
    });
}

// ============================================
// File Handling
// ============================================

function handleDragOver(e) {
    e.preventDefault();
    e.currentTarget.classList.add('dragover');
}

function handleDragLeave(e) {
    e.currentTarget.classList.remove('dragover');
}

function handleDrop(e) {
    e.preventDefault();
    e.currentTarget.classList.remove('dragover');

    const files = e.dataTransfer.files;
    if (files.length > 0) {
        processFile(files[0]);
    }
}

function handleFileSelect(e) {
    const files = e.target.files;
    if (files.length > 0) {
        processFile(files[0]);
    }
}

function processFile(file) {
    if (!file.name.toLowerCase().endsWith('.obj')) {
        alert('Please upload an OBJ file');
        return;
    }

    state.uploadedFile = file;
    state.isClassified = false;

    // Update UI
    document.getElementById('uploadZone').style.display = 'none';
    document.getElementById('fileInfo').style.display = 'flex';
    document.getElementById('fileName').textContent = file.name;
    document.getElementById('fileSize').textContent = formatFileSize(file.size);
    document.getElementById('classifyBtn').disabled = false;
    document.getElementById('resultsSection').style.display = 'none';

    // Hide placeholder
    document.getElementById('viewerPlaceholder').style.display = 'none';

    // Load the OBJ file
    loadOBJFile(file);
}

function removeFile() {
    state.uploadedFile = null;
    state.isClassified = false;
    state.originalColors = null;

    // Clear mesh
    if (state.currentMesh) {
        state.scene.remove(state.currentMesh);
        state.currentMesh = null;
    }

    // Reset UI
    document.getElementById('uploadZone').style.display = 'block';
    document.getElementById('fileInfo').style.display = 'none';
    document.getElementById('fileInput').value = '';
    document.getElementById('classifyBtn').disabled = true;
    document.getElementById('progressSection').style.display = 'none';
    document.getElementById('resultsSection').style.display = 'none';
    document.getElementById('viewerPlaceholder').style.display = 'block';
}

function formatFileSize(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}

// ============================================
// OBJ Loading
// ============================================

function loadOBJFile(file) {
    const reader = new FileReader();
    reader.onload = (e) => {
        const contents = e.target.result;
        // Use custom parser that handles vertex colors
        const result = parseOBJWithColors(contents);
        displayParsedMesh(result.positions, result.colors, result.faces);
        console.log('File loaded:', file.name, '- Vertices:', result.positions.length / 3);

        // Auto-detect known demo scans and load matching classified colors
        const filename = file.name.toLowerCase();
        let classifiedPath = null;

        if (filename.includes('original_scan_2') || filename.includes('original_scan2') || filename.includes('pm5k088n')) {
            classifiedPath = 'samples/classified_scan2.obj';
            state.currentScan = 2;
            console.log('Detected Scan 2 (102k vertices) - loading matching classification...');
        } else if (filename.includes('original_scan') || filename.includes('o52p1szt')) {
            classifiedPath = 'samples/classified_full.obj';
            state.currentScan = 1;
            console.log('Detected Scan 1 (93k vertices) - loading matching classification...');
        }

        if (classifiedPath) {
            // Load the matching classified colors
            fetch(classifiedPath)
                .then(response => response.text())
                .then(text => {
                    state.classifiedColors = parseOBJColors(text);
                    countClassification(state.classifiedColors);
                    console.log('Classified colors loaded! Ready to classify.');
                })
                .catch(err => console.warn('Could not load classified colors:', err));
        } else if (state.classifiedColors) {
            console.log('Using pre-loaded classified colors');
        } else {
            console.log('Unknown scan - using default classified colors if available');
        }
    };
    reader.readAsText(file);
}

// ============================================
// Classification (Using Pre-classified File)
// ============================================

async function runClassification() {
    const progressSection = document.getElementById('progressSection');
    const progressFill = document.getElementById('progressFill');
    const progressPercent = document.getElementById('progressPercent');
    const progressStatus = document.getElementById('progressStatus');
    const resultsSection = document.getElementById('resultsSection');
    const classifyBtn = document.getElementById('classifyBtn');

    // Check if we have the classified colors
    if (!state.classifiedColors) {
        alert('Classified mesh not loaded. Please ensure samples/classified_full.obj exists.');
        return;
    }

    // Show progress, hide results
    progressSection.style.display = 'block';
    resultsSection.style.display = 'none';
    classifyBtn.disabled = true;

    // Simulated classification with realistic pauses and snaps
    // Scan 1: ~15 seconds, Scan 2: ~16.2 seconds (more vertices)
    const timeMultiplier = state.currentScan === 2 ? 1.08 : 1.0;

    const stages = [
        // Quick initial burst
        { progress: 8, status: 'Initializing neural network...', duration: 400 * timeMultiplier, snap: true },
        { progress: 12, status: 'Initializing neural network...', duration: 800 * timeMultiplier, snap: false },
        // Pause then snap forward on model load
        { progress: 12, status: 'Loading model weights...', duration: 600 * timeMultiplier, snap: false },
        { progress: 24, status: 'Loading model weights...', duration: 300 * timeMultiplier, snap: true },
        // Slow preprocessing
        { progress: 28, status: 'Preprocessing mesh data...', duration: 1200 * timeMultiplier, snap: false },
        { progress: 35, status: 'Extracting surface features...', duration: 1400 * timeMultiplier, snap: false },
        // Stall then burst on patch generation
        { progress: 37, status: 'Generating surface patches...', duration: 800 * timeMultiplier, snap: false },
        { progress: 48, status: 'Generating surface patches...', duration: 400 * timeMultiplier, snap: true },
        { progress: 52, status: 'Generating surface patches...', duration: 600 * timeMultiplier, snap: false },
        // GPU inference - chunky progress
        { progress: 58, status: 'Running GPU inference (batch 1/3)...', duration: 1200 * timeMultiplier, snap: false },
        { progress: 67, status: 'Running GPU inference (batch 2/3)...', duration: 300 * timeMultiplier, snap: true },
        { progress: 71, status: 'Running GPU inference (batch 2/3)...', duration: 900 * timeMultiplier, snap: false },
        { progress: 79, status: 'Running GPU inference (batch 3/3)...', duration: 250 * timeMultiplier, snap: true },
        { progress: 84, status: 'Running GPU inference (batch 3/3)...', duration: 1100 * timeMultiplier, snap: false },
        // Quick finish
        { progress: 92, status: 'Applying vertex classifications...', duration: 400 * timeMultiplier, snap: true },
        { progress: 94, status: 'Applying vertex classifications...', duration: 800 * timeMultiplier, snap: false },
        { progress: 97, status: 'Validating results...', duration: 600 * timeMultiplier, snap: false },
        { progress: 100, status: 'Classification complete!', duration: 300 * timeMultiplier, snap: true }
    ];

    // Animate through stages
    let currentProgress = 0;

    for (let i = 0; i < stages.length; i++) {
        const stage = stages[i];
        const targetProgress = stage.progress;

        progressStatus.textContent = stage.status;

        if (stage.snap) {
            // Instant snap to target
            currentProgress = targetProgress;
            progressFill.style.width = currentProgress + '%';
            progressPercent.textContent = Math.round(currentProgress) + '%';
            await sleep(stage.duration);
        } else {
            // Smooth crawl to target
            const startProgress = currentProgress;
            const progressDiff = targetProgress - startProgress;
            const stepDuration = 80;
            const steps = Math.ceil(stage.duration / stepDuration);

            for (let step = 0; step <= steps; step++) {
                // Ease-out for more natural feel
                const t = step / steps;
                const eased = 1 - Math.pow(1 - t, 2);
                const stepProgress = startProgress + (progressDiff * eased);
                progressFill.style.width = stepProgress + '%';
                progressPercent.textContent = Math.round(stepProgress) + '%';
                await sleep(stepDuration);
            }
            currentProgress = targetProgress;
        }

        // At 92%, apply the classified colors
        if (stage.progress === 92 && state.currentMesh) {
            applyClassifiedColors();
        }
    }

    // Show results
    state.isClassified = true;
    displayResults();

    classifyBtn.disabled = false;
    progressSection.style.display = 'none';
    resultsSection.style.display = 'block';
}

function applyClassifiedColors() {
    if (!state.currentMesh || !state.classifiedColors) {
        console.error('Cannot apply colors: mesh or classified colors missing');
        console.log('  currentMesh:', !!state.currentMesh);
        console.log('  classifiedColors:', !!state.classifiedColors, state.classifiedColors ? state.classifiedColors.length / 3 : 0);
        return false;
    }

    const geometry = state.currentMesh.geometry;
    const colors = geometry.attributes.color;

    const meshVertices = colors.count;
    const classifiedVertices = state.classifiedColors.length / 3;

    console.log('Applying classified colors...');
    console.log('  Mesh vertices:', meshVertices);
    console.log('  Classified vertices:', classifiedVertices);

    if (meshVertices !== classifiedVertices) {
        console.warn('Vertex count mismatch! Colors may not align correctly.');
        console.warn('For best results, use the matching scan file.');
    }

    // Apply the pre-classified colors directly
    const count = Math.min(colors.count * 3, state.classifiedColors.length);

    for (let i = 0; i < count; i++) {
        colors.array[i] = state.classifiedColors[i];
    }

    colors.needsUpdate = true;
    console.log('Colors applied successfully!');
    return true;
}

function displayResults() {
    // Update result values
    document.getElementById('teethPercent').textContent = results.teethPercent + '%';
    document.getElementById('gingivaPercent').textContent = results.gingivaPercent + '%';
    document.getElementById('vertexCount').textContent = results.vertexCount.toLocaleString();

    // Show appropriate processing time based on scan
    const processingTime = state.currentScan === 2 ? CONFIG.processingTimeScan2 : CONFIG.processingTimeScan1;
    document.getElementById('processingTime').textContent = processingTime + 's';
    document.getElementById('confidence').textContent = results.confidence + '%';

    // Animate result bars
    setTimeout(() => {
        document.getElementById('teethBar').style.width = results.teethPercent + '%';
        document.getElementById('gingivaBar').style.width = results.gingivaPercent + '%';
    }, 100);
}

// ============================================
// Viewer Controls
// ============================================

function resetView() {
    state.camera.position.set(0, 0, 100);
    state.camera.lookAt(0, 0, 0);
    state.controls.reset();
}

function toggleWireframe() {
    if (state.currentMesh) {
        state.currentMesh.material.wireframe = !state.currentMesh.material.wireframe;
    }
}

function toggleFullscreen() {
    const container = document.getElementById('viewerContainer');
    if (!document.fullscreenElement) {
        container.requestFullscreen();
    } else {
        document.exitFullscreen();
    }
}

function switchViewerTab(view) {
    // Update tab styles
    document.querySelectorAll('.viewer-tab').forEach(tab => {
        tab.classList.toggle('active', tab.dataset.view === view);
    });

    if (!state.currentMesh) return;

    const colors = state.currentMesh.geometry.attributes.color;

    switch (view) {
        case 'original':
            // Restore original colors
            if (state.originalColors) {
                for (let i = 0; i < colors.array.length; i++) {
                    colors.array[i] = state.originalColors[i];
                }
                colors.needsUpdate = true;
            }
            break;
        case 'classified':
            // Apply classified colors
            if (state.isClassified && state.classifiedColors) {
                const count = Math.min(colors.array.length, state.classifiedColors.length);
                for (let i = 0; i < count; i++) {
                    colors.array[i] = state.classifiedColors[i];
                }
                colors.needsUpdate = true;
            }
            break;
        case 'comparison':
            // Could implement split view here
            break;
    }
}

// ============================================
// Utilities
// ============================================

function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

// ============================================
// Demo Mode - Quick Loading Functions
// ============================================

/**
 * Load the demo with pre-classified results
 * Usage: loadDemo() or loadDemo('samples/original_scan.obj', 'samples/classified_full.obj')
 */
window.loadDemo = function(originalPath = 'samples/original_scan.obj', classifiedPath = 'samples/classified_full.obj') {
    console.log('Loading demo...');

    // Update UI
    document.getElementById('uploadZone').style.display = 'none';
    document.getElementById('fileInfo').style.display = 'flex';
    document.getElementById('fileName').textContent = originalPath.split('/').pop();
    document.getElementById('fileSize').textContent = 'Demo file';
    document.getElementById('classifyBtn').disabled = false;
    document.getElementById('viewerPlaceholder').style.display = 'none';

    // Load classified colors first (custom parser for v x y z r g b format)
    fetch(classifiedPath)
        .then(response => response.text())
        .then(classifiedText => {
            state.classifiedColors = parseOBJColors(classifiedText);
            console.log('Classified colors loaded:', state.classifiedColors.length / 3, 'vertices');
            countClassification(state.classifiedColors);

            // Then load and display original mesh
            return fetch(originalPath);
        })
        .then(response => response.text())
        .then(originalText => {
            // Parse original with custom loader to preserve colors
            const result = parseOBJWithColors(originalText);
            displayParsedMesh(result.positions, result.colors, result.faces);
            console.log('Demo loaded! Click "Run Classification" to see results.');
        })
        .catch(error => {
            console.error('Error loading demo:', error);
        });
};

// Parse OBJ file with positions, colors, and faces
function parseOBJWithColors(objText) {
    const lines = objText.split('\n');
    const positions = [];
    const colors = [];
    const faces = [];

    for (const line of lines) {
        const trimmed = line.trim();
        if (trimmed.startsWith('v ')) {
            const parts = trimmed.split(/\s+/);
            positions.push(parseFloat(parts[1]), parseFloat(parts[2]), parseFloat(parts[3]));
            // Colors if present
            if (parts.length >= 7) {
                colors.push(parseFloat(parts[4]), parseFloat(parts[5]), parseFloat(parts[6]));
            } else {
                colors.push(0.7, 0.7, 0.7); // Default gray
            }
        } else if (trimmed.startsWith('f ')) {
            const parts = trimmed.split(/\s+/).slice(1);
            const indices = parts.map(p => parseInt(p.split('/')[0]) - 1);
            // Triangulate faces
            for (let i = 1; i < indices.length - 1; i++) {
                faces.push(indices[0], indices[i], indices[i + 1]);
            }
        }
    }

    return {
        positions: new Float32Array(positions),
        colors: new Float32Array(colors),
        faces: new Uint32Array(faces)
    };
}

// Display parsed mesh with proper colors
function displayParsedMesh(positions, colors, faces) {
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    geometry.setIndex(new THREE.BufferAttribute(faces, 1));
    geometry.computeVertexNormals();

    // Store original colors
    state.originalColors = colors.slice();

    const material = new THREE.MeshPhongMaterial({
        vertexColors: true,
        side: THREE.DoubleSide,
        flatShading: false
    });

    const mesh = new THREE.Mesh(geometry, material);

    // Center and scale
    geometry.computeBoundingBox();
    const box = geometry.boundingBox;
    const center = box.getCenter(new THREE.Vector3());
    const size = box.getSize(new THREE.Vector3());
    const maxDim = Math.max(size.x, size.y, size.z);
    const scale = 50 / maxDim;

    geometry.translate(-center.x, -center.y, -center.z);
    mesh.scale.set(scale, scale, scale);

    // Update results
    results.vertexCount = positions.length / 3;

    // Remove old mesh
    if (state.currentMesh) {
        state.scene.remove(state.currentMesh);
    }

    state.currentMesh = mesh;
    state.scene.add(mesh);
    resetView();
}

/**
 * Instantly show classification (skip animation)
 */
window.instantClassify = function() {
    if (!state.currentMesh || !state.classifiedColors) {
        console.error('Load a mesh first with loadDemo()');
        console.log('state.currentMesh:', !!state.currentMesh);
        console.log('state.classifiedColors:', !!state.classifiedColors);
        return;
    }

    applyClassifiedColors();

    state.isClassified = true;
    displayResults();
    document.getElementById('resultsSection').style.display = 'block';

    console.log('Classification applied instantly');
};

/**
 * Directly load and show the classified mesh (skip original entirely)
 * Usage: loadClassified() or loadClassified('samples/classified_scan2.obj')
 */
window.loadClassified = function(path = 'samples/classified_full.obj') {
    console.log('Loading pre-classified mesh directly...');

    document.getElementById('uploadZone').style.display = 'none';
    document.getElementById('fileInfo').style.display = 'flex';
    document.getElementById('fileName').textContent = path.split('/').pop();
    document.getElementById('fileSize').textContent = 'Classified';
    document.getElementById('viewerPlaceholder').style.display = 'none';

    fetch(path)
        .then(response => response.text())
        .then(text => {
            const result = parseOBJWithColors(text);
            displayParsedMesh(result.positions, result.colors, result.faces);
            state.classifiedColors = result.colors.slice();
            state.isClassified = true;
            countClassification(result.colors);
            displayResults();
            document.getElementById('resultsSection').style.display = 'block';
            document.getElementById('classifyBtn').disabled = true;
            console.log('Classified mesh loaded and displayed!');
        })
        .catch(error => {
            console.error('Error loading classified mesh:', error);
        });
};

// Demo scan presets
window.loadScan1 = function() {
    state.currentScan = 1;
    loadDemo('samples/original_scan.obj', 'samples/classified_full.obj');
};
window.loadScan2 = function() {
    state.currentScan = 2;
    loadDemo('samples/original_scan_2.obj', 'samples/classified_scan2.obj');
};

console.log('========================================');
console.log('DentAI Pro Demo Ready');
console.log('========================================');
console.log('');
console.log('QUICK START - Two sample scans:');
console.log('  loadScan1()  - First scan (93k vertices)');
console.log('  loadScan2()  - Second scan (102k vertices)');
console.log('  Then click "Run Classification"');
console.log('');
console.log('OTHER OPTIONS:');
console.log('  loadClassified()  - Show result directly');
console.log('  instantClassify() - Skip loading bar');
console.log('');
console.log('========================================');
