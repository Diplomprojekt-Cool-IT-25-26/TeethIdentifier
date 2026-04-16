"""
Morphological post-processing for mesh vertex label predictions.

Operations work on the vertex adjacency graph derived from mesh faces:
  - Erosion:  shrinks a class by flipping border vertices to the other class
  - Dilation: expands a class by converting adjacent vertices to that class
  - Opening:  erosion then dilation — removes small isolated blobs
  - Closing:  dilation then erosion — fills small holes within a region
"""

import numpy as np
from scipy.sparse import csr_matrix


def build_edge_array(mesh):
    """
    Build directed edge array from mesh faces.

    Returns:
        np.ndarray of shape (M, 2): each row is [src_vertex, dst_vertex]
    """
    faces = mesh.faces
    edges = np.vstack([
        faces[:, [0, 1]], faces[:, [1, 0]],
        faces[:, [1, 2]], faces[:, [2, 1]],
        faces[:, [0, 2]], faces[:, [2, 0]],
    ])
    return edges


def erode(labels, edges, target_class=1, iterations=1):
    """
    Shrink target_class: any border vertex of target_class that has a
    neighbour of the other class gets flipped.
    """
    result = labels.copy().astype(np.int32)
    src, dst = edges[:, 0], edges[:, 1]
    for _ in range(iterations):
        border = (result[src] == target_class) & (result[dst] != target_class)
        result[np.unique(src[border])] = 1 - target_class
    return result


def dilate(labels, edges, target_class=1, iterations=1):
    """
    Expand target_class: any vertex adjacent to target_class gets flipped to it.
    """
    result = labels.copy().astype(np.int32)
    src, dst = edges[:, 0], edges[:, 1]
    for _ in range(iterations):
        border = (result[src] != target_class) & (result[dst] == target_class)
        result[np.unique(src[border])] = target_class
    return result


def opening(labels, edges, target_class=1, iterations=1):
    """Remove small isolated target_class blobs (erosion then dilation)."""
    return dilate(erode(labels, edges, target_class, iterations), edges, target_class, iterations)


def closing(labels, edges, target_class=1, iterations=1):
    """Fill small holes inside target_class regions (dilation then erosion)."""
    return erode(dilate(labels, edges, target_class, iterations), edges, target_class, iterations)


def remove_small_components(labels, edges, min_size=500):
    """
    Find all connected components per class and flip any that are smaller
    than min_size to the opposing class.

    Removes isolated gingiva blobs sitting on teeth and vice versa.
    """
    from collections import deque

    n = len(labels)

    # Build adjacency list
    adj = [[] for _ in range(n)]
    for src, dst in edges:
        adj[src].append(dst)

    result = labels.copy().astype(np.int32)
    visited = np.zeros(n, dtype=bool)

    for start in range(n):
        if visited[start]:
            continue
        # BFS to find full connected component
        component = []
        queue = deque([start])
        visited[start] = True
        cls = result[start]
        while queue:
            v = queue.popleft()
            component.append(v)
            for nb in adj[v]:
                if not visited[nb] and result[nb] == cls:
                    visited[nb] = True
                    queue.append(nb)

        if len(component) < min_size:
            result[component] = 1 - cls

    return result


def smooth_boundary(labels, edges, iterations=10):
    """
    Smooth the boundary by treating labels as a float field and applying
    graph diffusion in the boundary region only.

    Each boundary vertex's value is replaced by the average of its
    neighbours' values, then re-thresholded at 0.5. Interior vertices
    (clearly tooth or gingiva) are left untouched.
    Repeated iterations progressively straighten and smooth the boundary curve.
    """
    n = len(labels)

    # Build sparse adjacency matrix for efficient neighbour averaging
    rows, cols = edges[:, 0], edges[:, 1]
    A = csr_matrix((np.ones(len(edges)), (rows, cols)), shape=(n, n))
    degree = np.array(A.sum(axis=1)).flatten()
    degree[degree == 0] = 1.0

    field = labels.astype(np.float32)

    for _ in range(iterations):
        # Identify current boundary vertices
        src, dst = edges[:, 0], edges[:, 1]
        boundary_mask = np.zeros(n, dtype=bool)
        boundary_mask[np.unique(src[np.round(field[src]) != np.round(field[dst])])] = True

        # Average neighbour values
        smoothed = A.dot(field) / degree

        # Only update boundary vertices
        new_field = field.copy()
        new_field[boundary_mask] = smoothed[boundary_mask]
        field = new_field

    return (field >= 0.5).astype(np.int32)


def majority_vote(labels, edges, iterations=5):
    """
    Smooth the boundary by repeatedly flipping border vertices
    where the majority of neighbours disagree.

    Unlike erosion/dilation (any neighbour triggers a flip), this only
    flips a vertex when strictly more than half its neighbours have the
    other class — so interior regions stay stable and only the ragged
    boundary gets straightened.
    """
    # Build adjacency list from edge array for efficient neighbour lookup
    n = len(labels)
    from collections import defaultdict
    adj = defaultdict(list)
    for src, dst in edges:
        adj[src].append(dst)

    result = labels.copy().astype(np.int32)
    for it in range(iterations):
        new_labels = result.copy()
        # Only process border vertices (at least one differing neighbour)
        src, dst = edges[:, 0], edges[:, 1]
        border_mask = result[src] != result[dst]
        border_vertices = np.unique(src[border_mask])

        for v in border_vertices:
            neighbours = adj[v]
            if not neighbours:
                continue
            neighbour_sum = np.sum(result[neighbours])
            majority_class = 1 if neighbour_sum > len(neighbours) / 2 else 0
            new_labels[v] = majority_class

        changed = int(np.sum(new_labels != result))
        result = new_labels
        print(f"  iteration {it+1}: {changed:,} vertices flipped")
        if changed == 0:
            break

    return result


def reconstruct_gingiva(labels, edges, erode_iters=4):
    """
    Remove gingiva fingers via morphological reconstruction:
      1. Erode gingiva heavily to sever narrow fingers at their necks
      2. Keep only the largest surviving gingiva component (the core body)
      3. Flood-fill outward from that seed, constrained to the pre-erosion
         gingiva mask — the main body grows back exactly, severed fingers
         can never be reached so they stay gone.

    Args:
        labels:      np.ndarray (N,) of int, 0=gingiva / 1=tooth
        edges:       edge array from build_edge_array
        erode_iters: how aggressively to erode before reconstruction;
                     must be >= half the finger neck width in vertices

    Returns:
        np.ndarray: labels with gingiva fingers removed
    """
    from collections import deque

    n = len(labels)
    original_gingiva = (labels == 0)  # mask we constrain flood-fill to

    # Step 1: erode gingiva to sever narrow necks
    eroded = erode(labels, edges, target_class=0, iterations=erode_iters)

    # Step 2: find largest gingiva component in eroded result as the seed
    src, dst = edges[:, 0], edges[:, 1]
    adj = [[] for _ in range(n)]
    for s, d in zip(src, dst):
        adj[s].append(d)

    visited = np.zeros(n, dtype=bool)
    components = []
    for start in range(n):
        if visited[start] or eroded[start] != 0:
            continue
        comp = []
        queue = deque([start])
        visited[start] = True
        while queue:
            v = queue.popleft()
            comp.append(v)
            for nb in adj[v]:
                if not visited[nb] and eroded[nb] == 0:
                    visited[nb] = True
                    queue.append(nb)
        components.append(comp)

    if not components:
        return labels.copy()

    seed_verts = max(components, key=len)

    # Step 3: constrained flood-fill — expand seed back into original gingiva mask
    result = labels.copy()
    result[:] = 1  # start all-tooth, flood fill will reclaim gingiva
    in_queue = np.zeros(n, dtype=bool)
    queue = deque(seed_verts)
    for v in seed_verts:
        result[v] = 0
        in_queue[v] = True

    while queue:
        v = queue.popleft()
        for nb in adj[v]:
            if not in_queue[nb] and original_gingiva[nb]:
                result[nb] = 0
                in_queue[nb] = True
                queue.append(nb)

    return result


def postprocess(labels, mesh, open_iters=2, close_iters=2, erode_gingiva_iters=1, majority_vote_iters=0, open_gingiva_iters=0, reconstruct_iters=0):
    """
    Full post-processing pipeline applied after model predictions:
      1. Opening on teeth   — removes small isolated tooth islands in gingiva
      2. Closing on teeth   — fills small gingiva gaps inside tooth regions
      3. Erosion on gingiva — shrinks over-predicted gingiva at the border

    Args:
        labels:               np.ndarray (N,) of int, 0=gingiva / 1=tooth
        mesh:                 trimesh.Trimesh
        open_iters:           opening iterations (0 to skip)
        close_iters:          closing iterations (0 to skip)
        erode_gingiva_iters:  gingiva erosion iterations (0 to skip)
        majority_vote_iters:  majority vote iterations (0 to skip) — trims gingiva peninsulas
        open_gingiva_iters:   opening on gingiva (0 to skip) — erode then dilate gingiva to remove fingers
        reconstruct_iters:    morphological reconstruction erosion depth (0 to skip) — severs fingers then flood-fills back

    Returns:
        np.ndarray: post-processed labels, same shape as input
    """
    print("\n[Post-processing] Building mesh edge array...")
    edges = build_edge_array(mesh)

    result = labels.copy()

    if open_iters > 0:
        print(f"[Post-processing] Opening on teeth ({open_iters} iter) - removes isolated tooth islands...")
        result = opening(result, edges, target_class=1, iterations=open_iters)

    if close_iters > 0:
        print(f"[Post-processing] Closing on teeth ({close_iters} iter) - fills gingiva gaps in tooth regions...")
        result = closing(result, edges, target_class=1, iterations=close_iters)

    if erode_gingiva_iters > 0:
        print(f"[Post-processing] Eroding gingiva ({erode_gingiva_iters} iter) - shrinks over-predicted gingiva border...")
        result = erode(result, edges, target_class=0, iterations=erode_gingiva_iters)

    print(f"[Post-processing] Connected component filter (min 500 vertices) - removes isolated blobs...")
    result = remove_small_components(result, edges, min_size=500)

    print(f"[Post-processing] Boundary smoothing (10 iter) - diffusion-based curve straightening...")
    result = smooth_boundary(result, edges, iterations=10)

    if majority_vote_iters > 0:
        print(f"[Post-processing] Majority vote ({majority_vote_iters} iter) - trims gingiva peninsulas...")
        result = majority_vote(result, edges, iterations=majority_vote_iters)

    if open_gingiva_iters > 0:
        print(f"[Post-processing] Opening on gingiva ({open_gingiva_iters} iter) - removes gingiva fingers...")
        result = opening(result, edges, target_class=0, iterations=open_gingiva_iters)

    if reconstruct_iters > 0:
        print(f"[Post-processing] Morphological reconstruction (erode {reconstruct_iters} iter) - severs fingers then flood-fills back...")
        result = reconstruct_gingiva(result, edges, erode_iters=reconstruct_iters)

    n_changed = int(np.sum(result != labels))
    print(f"[Post-processing] Done. Changed {n_changed:,} / {len(labels):,} vertices ({n_changed/len(labels)*100:.1f}%)")

    return result
