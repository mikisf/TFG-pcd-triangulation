import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))  # Add the parent directory to sys.path
from marching_cubes.marching_cubes import marching_cubes
from marching_tetrahedra.marching_tetrahedra import marching_tetrahedra
from poisson.poisson_3d import poisson_3d

import numpy as np
import open3d as o3d
from scipy.spatial import Delaunay

# Parameters
grid_size = 32
center = np.array([16, 16, 16])
radius = 8


# ---------- 1. Marching Cubes ----------
def marching_cubes_sphere():
    print("Generating Marching Cubes...")

    grid = np.zeros((grid_size, grid_size, grid_size), dtype=np.float32)

    # Fill SDF: negative inside the sphere, positive outside
    for x in range(grid_size):
        for y in range(grid_size):
            for z in range(grid_size):
                p = np.array([x, y, z])
                d = np.linalg.norm(p - center)
                grid[x, y, z] = radius - d  # Positive inside

    # Extract isosurface
    verts, faces = marching_cubes(grid, 0)

    # Convert to mesh and save
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    o3d.io.write_triangle_mesh("results/outputs/marching_cubes.obj", mesh)


# ---------- 2. Marching Tetrahedra ----------
def marching_tetrahedra_sphere():
    print("Generating Marching Tetrahedra...")

    # Random points in cube
    N = 32**3
    points = np.random.uniform(0, grid_size, (N, 3))

    # Signed distance field (SDF)
    sdf = radius - np.linalg.norm(points - center, axis=1)

    # Tetrahedralization
    delaunay = Delaunay(points)
    tetras = delaunay.simplices

    # Extract isosurface
    verts, faces = marching_tetrahedra(points, tetras, sdf, 0)

    # Convert to mesh and save
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    o3d.io.write_triangle_mesh("results/outputs/marching_tetrahedra.obj", mesh)


# ---------- 3. Poisson Surface Reconstruction ----------
def poisson_sphere():
    print("Generating Poisson Reconstruction...")

    N = 32**3
    # Uniform spherical surface sampling
    theta = 2 * np.pi * np.random.rand(N)
    phi = np.arccos(2 * np.random.rand(N) - 1)
    x = center[0] + radius * np.sin(phi) * np.cos(theta)
    y = center[1] + radius * np.sin(phi) * np.sin(theta)
    z = center[2] + radius * np.cos(phi)
    points = np.vstack((x, y, z)).T

    # Normals: from outward to center
    normals = center - points
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)

    # Extract surface
    verts, faces = poisson_3d(points, normals, grid_size=32)

    # Convert to mesh and save
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    o3d.io.write_triangle_mesh("results/outputs/poisson.obj", mesh)


marching_cubes_sphere()
marching_tetrahedra_sphere()
poisson_sphere()

print("All meshes saved in ./outputs")
