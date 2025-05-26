import numpy as np
import heapq
from scipy.spatial import KDTree


class Vertex:
    def __init__(self, position, index):
        self.position = np.array(position, dtype=np.float64)
        self.index = index
        self.quadric = np.zeros((4, 4))
        self.valid = True


class EdgeCollapse:
    def __init__(self, v1, v2, error, target_position):
        self.v1 = v1
        self.v2 = v2
        self.error = error
        self.target_position = target_position

    def __lt__(self, other):
        return self.error < other.error


def compute_face_quadric(v0, v1, v2):
    # Compute plane: ax + by + cz + d = 0
    normal = np.cross(v1 - v0, v2 - v0)
    if np.linalg.norm(normal) == 0:
        # Degenerate case: points are collinear, return zero quadric
        return np.zeros((4, 4))
    normal = normal / np.linalg.norm(normal)
    a, b, c = normal
    d = -np.dot(normal, v0)
    plane = np.array([a, b, c, d])
    return np.outer(plane, plane)


def compute_edge_collapse(v1, v2):
    Q = v1.quadric + v2.quadric
    Q_bar = Q.copy()
    Q_bar[3] = [0, 0, 0, 1]

    try:
        v_opt = np.linalg.solve(Q_bar, [0, 0, 0, 1])[:3]
    except np.linalg.LinAlgError:
        # Fall back: choose v1.position, v2.position, or midpoint, whichever gives least error
        candidates = [v1.position, v2.position, (v1.position + v2.position) / 2]
        errors = [(np.append(pos, 1.0) @ Q @ np.append(pos, 1.0), pos) for pos in candidates]
        v_opt = min(errors, key=lambda x: x[0])[1]

    v_opt_hom = np.append(v_opt, 1.0)
    error = v_opt_hom @ Q @ v_opt_hom
    return EdgeCollapse(v1, v2, error, v_opt)


def update_faces(faces, v_from, v_to):
    new_faces = []
    for f in faces:
        if v_from in f:
            f = [v_to if vi == v_from else vi for vi in f]
        # Skip degenerate faces (e.g. two or more equal vertices)
        if len(set(f)) == 3:
            new_faces.append(f)
    return new_faces


def get_vertex_neighbors(faces, v_index):
    neighbors = set()
    for f in faces:
        if v_index in f:
            neighbors.update(f)
    neighbors.discard(v_index)
    return neighbors


def extract_mesh(vertices, faces):
    # Reindex vertices
    index_map = {}
    new_vertices = []
    for i, v in enumerate(vertices):
        if v.valid:
            index_map[v.index] = len(new_vertices)
            new_vertices.append(v.position)
    new_faces = []
    for f in faces:
        if all(vertices[vi].valid for vi in f):
            new_faces.append([index_map[vi] for vi in f])
    return np.array(new_vertices), np.array(new_faces)


def simplify_mesh(vertices, faces, proximity_threshold=0.0, target_vertex_count=0, target_face_count=0, target_error=float("inf")):
    V = [Vertex(v, i) for i, v in enumerate(vertices)]
    F = faces.copy()

    # Step 1: Compute initial quadrics
    for f in F:
        q = compute_face_quadric(V[f[0]].position, V[f[1]].position, V[f[2]].position)
        for idx in f:
            V[idx].quadric += q

    # Step 2: Select all valid pairs
    edges = set()
    for f in F:
        edges.update([(min(f[i], f[j]), max(f[i], f[j])) for i in range(3) for j in range(i + 1, 3)])

    # If proximity threshold is set, use KDTree to find pairs within the threshold
    if proximity_threshold > 0.0:
        positions = np.array([v.position for v in V])
        tree = KDTree(positions)
        for i, v in enumerate(V):
            indices = tree.query_ball_point(v.position, proximity_threshold)
            for j in indices:
                if i < j and (i, j) not in edges:
                    edges.add((i, j))

    # Step 3 and 4: Compute edge collapses and use a min-heap
    heap = []
    for v1, v2 in edges:
        collapse = compute_edge_collapse(V[v1], V[v2])
        heapq.heappush(heap, collapse)

    # Step 5: Simplify
    while len([v for v in V if v.valid]) > target_vertex_count and len(F) > target_face_count and heap:
        collapse = heapq.heappop(heap)
        if collapse.error > target_error:
            break
        v1, v2 = collapse.v1, collapse.v2

        if not v1.valid or not v2.valid:
            continue

        # Collapse v2 into v1
        new_pos = collapse.target_position
        new_vertex = Vertex(new_pos, v1.index)
        new_vertex.quadric = v1.quadric + v2.quadric
        V[v1.index] = new_vertex
        V[v2.index].valid = False

        # Update faces: replace v2 with v1
        F = update_faces(F, v2.index, v1.index)

        # Remove old edges from heap
        heap = [ec for ec in heap if ec.v1.index not in (v1.index, v2.index) and ec.v2.index not in (v1.index, v2.index)]
        heapq.heapify(heap)

        # Recompute collapses for affected edges
        neighbors = get_vertex_neighbors(F, v1.index)
        for n in neighbors:
            if V[n].valid:
                ec = compute_edge_collapse(V[v1.index], V[n])
                heapq.heappush(heap, ec)

    return extract_mesh(V, F)


if __name__ == "__main__":
    import open3d as o3d

    mesh = o3d.io.read_triangle_mesh("surface_simplification_QEM/data/laptop.obj")
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)

    vertices, faces = simplify_mesh(vertices, faces, target_error=0.039)

    simplified_mesh = o3d.geometry.TriangleMesh()
    simplified_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    simplified_mesh.triangles = o3d.utility.Vector3iVector(faces)
    o3d.io.write_triangle_mesh("surface_simplification_QEM/data/laptop_simplified.obj", simplified_mesh)
