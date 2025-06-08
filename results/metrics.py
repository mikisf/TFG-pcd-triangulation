import open3d as o3d
import numpy as np
import psutil
import os
from scipy.spatial import cKDTree


def point_to_mesh_errors(pcd, mesh):
    pcd_points = np.asarray(pcd.points)
    tmesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene = o3d.t.geometry.RaycastingScene()
    _ = scene.add_triangles(tmesh)
    queries = o3d.core.Tensor(pcd_points, dtype=o3d.core.Dtype.Float32)
    dists = scene.compute_distance(queries).numpy()
    return dists


def hausdorff_distance(mesh, pcd, sample_density=1.0):
    # Step 1: from PCD → Mesh surface
    dists_pcd_to_mesh = point_to_mesh_errors(pcd, mesh)
    hausdorff_1 = np.max(dists_pcd_to_mesh)

    # Step 2: from Mesh surface → PCD using mesh sampling
    sampled_points = mesh.sample_points_uniformly(number_of_points=int(len(pcd.points) * sample_density))
    tree = cKDTree(np.asarray(pcd.points))
    dists_mesh_to_pcd, _ = tree.query(np.asarray(sampled_points.points))
    hausdorff_2 = np.max(dists_mesh_to_pcd)

    # Symmetric Hausdorff distance
    return max(hausdorff_1, hausdorff_2)


def triangle_aspect_ratios(mesh):
    triangles = np.asarray(mesh.triangles)
    vertices = np.asarray(mesh.vertices)
    ratios = []

    for tri in triangles:
        a, b, c = vertices[tri]
        ab, bc, ca = np.linalg.norm(a - b), np.linalg.norm(b - c), np.linalg.norm(c - a)
        sides = [ab, bc, ca]
        s = sum(sides) / 2
        area = np.sqrt(s * (s - ab) * (s - bc) * (s - ca))
        if area > 1e-10:
            inradius = 2 * area / sum(sides)
            circumradius = ab * bc * ca / (4 * area)
            ratio = circumradius / (2 * inradius)
            ratios.append(ratio)
    return np.array(ratios)


import numpy as np


def angle_quality_stats(mesh):
    q_values = []
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)

    for tri in triangles:
        a, b, c = vertices[tri[0]], vertices[tri[1]], vertices[tri[2]]

        # Compute side lengths
        A = np.linalg.norm(b - c)
        B = np.linalg.norm(c - a)
        C = np.linalg.norm(a - b)

        # Compute semiperimeter
        s = (A + B + C) / 2.0

        # Compute triangle area using Heron's formula
        area_sq = s * (s - A) * (s - B) * (s - C)
        if area_sq <= 0:
            continue  # Degenerate triangle, skip

        area = np.sqrt(area_sq)

        # Compute inradius and circumradius
        r = area / s
        R = (A * B * C) / (4.0 * area)

        # Angular quality metric: ratio of inradius to circumradius
        q = r / R
        q_values.append(q)

    q_values = np.array(q_values)
    mean_q = np.mean(q_values)
    var_q = np.var(q_values)
    return mean_q, var_q


def triangle_area_stats(mesh):
    triangles = np.asarray(mesh.triangles)
    vertices = np.asarray(mesh.vertices)
    areas = []

    for tri in triangles:
        a, b, c = vertices[tri]
        area = 0.5 * np.linalg.norm(np.cross(b - a, c - a))
        areas.append(area)

    areas = np.array(areas)
    return np.mean(areas), np.var(areas)


def vertex_valence_distribution(mesh):
    triangles = np.asarray(mesh.triangles)
    valences = [0] * len(mesh.vertices)
    for tri in triangles:
        for i in tri:
            valences[i] += 1
    return np.mean(valences)


def edge_length_stats(mesh):
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles).astype(np.int64)

    edges = set()
    for tri in triangles:
        edges.update(
            [
                tuple(sorted((tri[0], tri[1]))),
                tuple(sorted((tri[1], tri[2]))),
                tuple(sorted((tri[2], tri[0]))),
            ]
        )

    ratios = []  # for h_min / l_max

    for tri in triangles:
        a, b, c = vertices[tri[0]], vertices[tri[1]], vertices[tri[2]]

        A = np.linalg.norm(b - c)
        B = np.linalg.norm(c - a)
        C = np.linalg.norm(a - b)

        s = (A + B + C) / 2.0
        area_sq = s * (s - A) * (s - B) * (s - C)
        if area_sq <= 0:
            continue  # Degenerate triangle, skip
        area = np.sqrt(area_sq)

        hA = 2 * area / A
        hB = 2 * area / B
        hC = 2 * area / C

        h_min = min(hA, hB, hC)
        l_max = max(A, B, C)

        ratio = h_min / l_max
        ratios.append(ratio)

    ratios = np.array(ratios)
    mean_ratio = np.mean(ratios)
    var_ratio = np.var(ratios)

    return mean_ratio, var_ratio


def count_holes(mesh):
    import networkx as nx

    triangles = np.asarray(mesh.triangles)

    edge_count = {}

    for tri in triangles:
        edges = [
            tuple(sorted((tri[0], tri[1]))),
            tuple(sorted((tri[1], tri[2]))),
            tuple(sorted((tri[2], tri[0]))),
        ]
        for edge in edges:
            edge_count[edge] = edge_count.get(edge, 0) + 1

    # Boundary edges appear only once
    boundary_edges = [edge for edge, count in edge_count.items() if count == 1]

    # Build a graph of boundary edges
    G = nx.Graph()
    G.add_edges_from(boundary_edges)

    # Count connected components (each = one hole)
    holes = list(nx.connected_components(G))
    return len(holes)


def memory_usage_gb():
    return psutil.Process(os.getpid()).memory_info().rss / (1024**3)


def analyze_surface(mesh_path, pcd_path=None):
    print(f"\nAnalyzing Mesh: {mesh_path}")
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    if pcd_path:
        pcd = o3d.io.read_point_cloud(pcd_path)

    # Accuracy
    if pcd_path:
        dists = point_to_mesh_errors(pcd, mesh)
        mean_err = np.mean(dists)
        rms_err = np.sqrt(np.mean(dists**2))
        hausdorff = np.max(dists)

    # Mesh quality
    aspect_ratios = triangle_aspect_ratios(mesh)
    mean_q, var_q = angle_quality_stats(mesh)
    area_mean, area_var = triangle_area_stats(mesh)
    mean_ratio, var_ratio = edge_length_stats(mesh)
    valence = vertex_valence_distribution(mesh)

    # Topological
    holes = count_holes(mesh)

    # Complexity
    num_vertices = len(mesh.vertices)
    num_faces = len(mesh.triangles)

    if pcd_path:
        print(f"-> Error mitjà punt-a-superfície: {1000 * mean_err:.4f} mm")
        print(f"-> RMS: {1000 * rms_err:.4f} mm")
        print(f"-> Distància de Hausdorff: {1000 * hausdorff:.4f} mm")

    print(f"-> Relació d’aspecte (mitjana): {np.mean(aspect_ratios):.2f}")
    print(f"-> Qualitat angular mitjana: {mean_q:.3f}, variància: {var_q:.5f}")
    print(f"-> Àrea (mitjana ± var): {area_mean:.4f} ± {area_var:.4f}")
    print(f"-> Quocient h_min/l_max (mitjana ± variància): {mean_ratio:.4f} ± {var_ratio:.6f}")
    print(f"-> València mitjana: {valence:.2f}")
    print(f"-> Nombre de forats: {holes}")
    print(f"-> Vèrtexs: {num_vertices}")
    print(f"-> Cares: {num_faces}")


ws = "results/outputs/"
analyze_surface(ws + "marching_cubes.obj", ws + "sphere.ply")
