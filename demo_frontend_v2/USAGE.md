# DentAI Demo Frontend v2 — Usage Guide

## First-time setup

### 1. Generate patch images (one-time)

From the **TeethIdentifier root directory**, run:

```
venv\Scripts\python.exe demo_frontend_v2/generate_patches.py
```

This takes ~10 seconds with GPU and produces 15 PNG images in `demo_frontend_v2/assets/patches/`.
You only need to run this once. The images are already there if you've already done this.

### 2. Start a local HTTP server

The demo uses `fetch()` to load OBJ files, so it **must** be served over HTTP — opening `index.html` directly will not work.

From the **TeethIdentifier root directory**, run:

```
python -m http.server 8080
```

### 3. Open in browser

```
http://localhost:8080/demo_frontend_v2/
```

Use **Chrome or Edge** for best compatibility with the Three.js viewer.

---

## Showcase walkthrough

The demo has 5 steps. At each step, the right panel shows relevant information.
You advance manually using the buttons — nothing progresses on its own.

---

### Step 1 — Load Scan

Click either **Sample Scan A** (upper jaw, 93k vertices) or **Sample Scan B** (lower jaw, 102k vertices).

Click **Load Scan**. The mesh appears in the 3D viewer as a grey object.

> **What to say:** "This is a raw 3D dental scan in OBJ format. It contains ~93,000 vertices — 
> the model will assign a label to every single one."

---

### Step 2 — Extract Patches

Click **Generate Patches**. The progress bar runs for ~10 seconds.

As it advances, real patch images fade in row by row:
- Top row: tooth surface patches (curved, high blue channel)
- Middle row: gingiva patches (flatter, more uniform)
- Bottom row: boundary patches (mixed signal)

> **What to say:** "For each vertex, we render a 100×100 image top-down from above the surface.
> Red encodes depth, green encodes shading, blue encodes curvature.
> High blue = strongly curved = tooth cusp. This is literally what the neural network sees."

Click **Continue to Inference**.

---

### Step 3 — CNN Inference

Click **Run Inference**. The animation takes ~15 seconds (matches real inference time).

- The architecture diagram lights up block by block as the network processes data
- 3 batch progress bars fill sequentially
- The overall progress bar tracks inference across all vertices

> **What to say:** "TeethNet is a convolutional neural network with ~2.5 million parameters.
> Input patches go through 3 Conv blocks, 2 Dense layers, and a sigmoid output.
> Any vertex scoring above 0.5 is classified as tooth. Below 0.5 is gingiva."

Click **View Raw Output**.

---

### Step 4 — Raw Model Output

The 3D mesh switches colour — **blue = teeth, pink = gingiva**.

The right panel shows raw statistics (vertex count, inference time, class percentages).

Rotate the mesh in the viewer and point out boundary artifacts.

> **What to say:** "These are direct model predictions, no cleanup yet.
> You can see rough boundaries, small isolated blobs, and areas where gingiva 
> creeps into tooth regions. This is normal — the CNN classifies each vertex 
> independently, without any awareness of the surrounding mesh structure."

Click **Configure Post-Processing**.

---

### Step 5 — Post-Processing

The right panel shows 6 toggle switches, each corresponding to a real morphological operation.

Click **★ Optimal Setup** to enable the recommended combination found during evaluation:
- Opening on Teeth ✓
- Closing on Teeth ✓
- Gingiva Erosion ✓
- Majority Vote Smoothing ✓
- Opening on Gingiva ✓
- Morphological Reconstruction ✗ (aggressive, off by default)

You can explain each toggle individually as you enable them.

Click **Apply Post-Processing**. Each enabled operation highlights briefly, then the mesh 
transitions to the clean post-processed result.

> **What to say:** "Post-processing works on the mesh adjacency graph — the connectivity
> between vertices. Opening removes isolated blobs, closing fills holes, erosion 
> shrinks the gingiva border, majority vote straightens jagged edges.
> These are all standard morphological operations, just applied to a graph instead of pixels."

---

## Viewer controls

| Action | Control |
|---|---|
| Rotate | Left-click + drag |
| Zoom | Scroll wheel |
| Pan | Right-click + drag |
| Reset view | Click the reset button (top right of viewer) |
| Toggle wireframe | Click the **Wireframe** tab |
| Show post-processed | Click the **Classified** tab (available after step 1) |
| Fullscreen | Click the fullscreen button |

---

## Notes

- The 3D animations are pre-scripted to match real timings measured from actual inference runs.
  Nothing communicates with any backend — it is entirely frontend.
- If patch images don't appear in Step 2, re-run `generate_patches.py`.
- If OBJ files fail to load, check that the HTTP server is running from the **TeethIdentifier root** (not from inside `demo_frontend_v2/`).
- Scan A uses freshly generated files (`classified_raw.obj` and `classified_postprocessed.obj`) — 6,246 vertices differ between them (6.7%), giving a visible before/after on the boundary regions.
- Scan B does not have a separate raw/post-processed distinction — the mesh transition in Step 5 will animate but the mesh itself won't change visually. If you want the same for Scan B, re-run `generate_samples.py` pointing at `original_scan_2.obj`.
