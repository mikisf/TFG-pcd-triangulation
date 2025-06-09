import numpy as np

tet_edges = [
    (0, 1),
    (1, 2),
    (2, 0),
    (0, 3),
    (1, 3),
    (2, 3),
]

MARCHING_TETRAHEDRA_LOT = {
    0: [],
    1: [(3, 2, 0)],
    2: [(1, 4, 0)],
    3: [(4, 2, 1), (4, 3, 2)],
    4: [(2, 5, 1)],
    5: [(0, 3, 5), (0, 5, 1)],
    6: [(4, 0, 5), (5, 0, 2)],
    7: [(5, 4, 3)],
    8: [(3, 4, 5)],
    9: [(2, 0, 4), (4, 5, 2)],
    10: [(0, 1, 5), (0, 5, 3)],
    11: [(2, 1, 5)],
    12: [(2, 3, 4), (2, 4, 1)],
    13: [(1, 0, 4)],
    14: [(0, 2, 3)],
    15: [],
}


def interpolate(p1, p2, valp1, valp2, threshold):
    if abs(threshold - valp1) < 1e-5:
        return p1
    if abs(threshold - valp2) < 1e-5:
        return p2
    if abs(valp1 - valp2) < 1e-5:
        return p1
    t = (threshold - valp1) / (valp2 - valp1)
    return p1 + t * (p2 - p1)


def marching_tetrahedra(points, tetrahedra, scalars, threshold):
    """
        0
       /|\ 
      / | \ 
  e3 /  |  \ e0
    /   e2  \ 
   /    |    \ 
  3--e4-|-----1
   \    |    /
 e5 \   |   / e1
     \  |  /
      \ | /
       \|/
        2

    """

    vertices = []
    faces = []

    for tet in tetrahedra:
        # Orient tetrahedra in a consistent manner
        p0, p1, p2, p3 = [np.array(points[i]) for i in tet]
        mat = np.column_stack((p1 - p0, p2 - p0, p3 - p0))
        if np.linalg.det(mat) < 0:
            tet = (tet[0], tet[3], tet[2], tet[1])

        tet_points = [points[i] for i in tet]
        tet_values = [scalars[i] for i in tet]

        # Compute 4-bit tet_index: 1 = below threshold
        tet_index = 0
        for i, v in enumerate(tet_values):
            if v < threshold:
                tet_index |= 1 << i

        triangles = MARCHING_TETRAHEDRA_LOT.get(tet_index, [])
        edge_vertices = {}

        for e_idx, (a, b) in enumerate(tet_edges):
            va, vb = tet_values[a], tet_values[b]
            if (tet_index >> a & 1) != (tet_index >> b & 1):  # Check if edge is intersected by the isosurface
                p1, p2 = np.array(tet_points[a]), np.array(tet_points[b])
                edge_vertices[e_idx] = interpolate(p1, p2, va, vb, threshold)

        for tri in triangles:
            v = []
            for edge_idx in tri:
                v.append(edge_vertices[edge_idx])
            start_idx = len(vertices)
            vertices.extend(v)
            faces.append([start_idx, start_idx + 1, start_idx + 2])

    return np.array(vertices), np.array(faces)


def create_obj(vertices, faces, file_path):
    """
    Creates a .obj file from vertices, faces, and calculated normals.

    Args:
        vertices (ndarray): Array of vertices (Nx3).
        faces (ndarray): Array of faces (Mx3).
        file_path (str): Path to save the .obj file. Defaults to 'output_with_normals.obj'.
    """

    with open(file_path, "w") as obj_file:
        # Write vertices
        for vertex in vertices:
            obj_file.write(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")

        # Write faces, referencing normals
        for face in faces:
            # Faces reference both vertex and normal indices
            face_line = " ".join([f"{v + 1}//{v + 1}" for v in face])  # OBJ format uses 1-based indexing
            obj_file.write(f"f {face_line}\n")

    print(f"OBJ file with normals created at {file_path}")


if __name__ == "__main__":
    import pyvista as pv

    multiblock = pv.read("marching_tetrahedra/data/disk_out_ref.ex2")
    mesh = multiblock["Element Blocks"][0]
    points = mesh.points
    points = [tuple(point) for point in points]
    scalars = mesh["Temp"]

    # Compute the Delaunay tetrahedralization
    from scipy.spatial import Delaunay

    scipy_delaunay = Delaunay(points)
    tetrahedra = [tuple(sorted(tetra)) for tetra in scipy_delaunay.simplices]

    threshold = 450
    vertices, faces = marching_tetrahedra(points, tetrahedra, scalars, threshold)

    create_obj(vertices, faces, "marching_tetrahedra/data/isosurface.obj")
