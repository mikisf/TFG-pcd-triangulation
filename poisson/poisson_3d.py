import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
from skimage.measure import marching_cubes


def generate_sphere_points(n_points, radius=0.4, center=(0.5, 0.5, 0.5)):
    phi = np.random.uniform(0, np.pi, n_points)
    theta = np.random.uniform(0, 2 * np.pi, n_points)

    x = center[0] + radius * np.sin(phi) * np.cos(theta)
    y = center[1] + radius * np.sin(phi) * np.sin(theta)
    z = center[2] + radius * np.cos(phi)

    points = np.stack([x, y, z], axis=-1)
    normals = -(points - center)  # inward normals
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    return points, normals


def create_vector_field(points, normals, grid_size):
    Vx = np.zeros((grid_size, grid_size, grid_size))
    Vy = np.zeros_like(Vx)
    Vz = np.zeros_like(Vx)
    count = np.zeros_like(Vx)

    for p, n in zip(points, normals):
        ix = int(p[0] * (grid_size - 1))
        iy = int(p[1] * (grid_size - 1))
        iz = int(p[2] * (grid_size - 1))
        ix, iy, iz = np.clip([ix, iy, iz], 0, grid_size - 1)
        Vx[ix, iy, iz] += n[0]
        Vy[ix, iy, iz] += n[1]
        Vz[ix, iy, iz] += n[2]
        count[ix, iy, iz] += 1

    count[count == 0] = 1  # avoid division by zero
    Vx /= count
    Vy /= count
    Vz /= count
    return Vx, Vy, Vz


def compute_divergence(Vx, Vy, Vz):
    h = 1.0 / (Vx.shape[0] - 1)

    dVx_dx = np.gradient(Vx, h, axis=0)
    dVy_dy = np.gradient(Vy, h, axis=1)
    dVz_dz = np.gradient(Vz, h, axis=2)

    div = dVx_dx + dVy_dy + dVz_dz
    return div


def solve_poisson(div):
    n = div.shape[0]
    A = lil_matrix((n**3, n**3))
    b = div.flatten()

    def index(i, j, k):
        return i * n * n + j * n + k

    for i in range(n):
        for j in range(n):
            for k in range(n):
                idx = index(i, j, k)
                A[idx, idx] = -6
                for di, dj, dk in [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]:
                    ni, nj, nk = i + di, j + dj, k + dk
                    if 0 <= ni < n and 0 <= nj < n and 0 <= nk < n:
                        A[idx, index(ni, nj, nk)] = 1

    A = A.tocsr()  # convert to CSR format for faster solving
    chi = spsolve(A, b)
    chi = chi.reshape((n, n, n))

    chi = (chi - chi.min()) / (chi.max() - chi.min())  # normalize to [0, 1]
    return chi


def poisson_3d(points, normals, grid_size):
    # Normalize
    min_coords = points.min(axis=0)
    max_coords = points.max(axis=0)
    points = (points - min_coords) / (max_coords - min_coords) * 0.9 + 0.05

    Vx, Vy, Vz = create_vector_field(points, normals, grid_size)
    div = compute_divergence(Vx, Vy, Vz)
    chi = solve_poisson(div)
    verts, faces, _, _ = marching_cubes(chi, level=0.5)

    # Denormalize vertices
    verts = verts / grid_size  # verts are in voxel units, convert to [0,1] range
    verts = (verts - 0.05) / 0.9  # invert [0.05, 0.95] scaling
    verts = verts * (max_coords - min_coords) + min_coords  # back to original space

    for face in faces:
        face[0], face[2] = face[2], face[0]  # flip the face orientation so that the normals point outwards

    return verts, faces


if __name__ == "__main__":
    import open3d as o3d

    pcd = o3d.io.read_point_cloud("poisson/data/stanford-bunny.ply")
    points = np.asarray(pcd.points)
    normals = -np.asarray(pcd.normals)  # inward normals

    verts, faces = poisson_3d(points, normals, grid_size=32)

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    o3d.io.write_triangle_mesh("poisson/data/stanford-bunny.obj", mesh)
