import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
from skimage.measure import find_contours

np.set_printoptions(suppress=True, precision=8)

chi = None
Vx = None
Vy = None


def generate_circle_points(n_points, radius=0.3, center=(0.5, 0.5)):
    angles = np.linspace(0, 2 * np.pi, n_points)
    points = np.stack([center[0] + radius * np.cos(angles), center[1] + radius * np.sin(angles)], axis=-1)
    normals = -np.stack([np.cos(angles), np.sin(angles)], axis=-1)  # inward normals
    return points, normals


def create_vector_field(points, normals, grid_size):
    global Vx, Vy
    Vx = np.zeros((grid_size, grid_size))
    Vy = np.zeros_like(Vx)
    count = np.zeros_like(Vx)

    for p, n in zip(points, normals):
        ix = int(p[0] * grid_size)
        iy = int(p[1] * grid_size)
        ix = np.clip(ix, 0, grid_size - 1)
        iy = np.clip(iy, 0, grid_size - 1)
        Vx[ix, iy] += n[0]
        Vy[ix, iy] += n[1]
        count[ix, iy] += 1

    count[count == 0] = 1  # avoid division by zero
    Vx /= count
    Vy /= count
    return Vx, Vy


def compute_divergence(Vx, Vy):
    h = 1.0 / (Vx.shape[0] - 1)

    dVx_dx = np.gradient(Vx, h, axis=0)
    dVy_dy = np.gradient(Vy, h, axis=1)

    div = dVx_dx + dVy_dy
    return div


def solve_poisson(div):
    global chi
    n = div.shape[0]
    A = lil_matrix((n**2, n**2))
    b = div.flatten()

    def index(i, j):
        return i * n + j

    for i in range(n):
        for j in range(n):
            idx = index(i, j)
            A[idx, idx] = -4
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ni, nj = i + di, j + dj
                if 0 <= ni < n and 0 <= nj < n:
                    A[idx, index(ni, nj)] = 1

    A = A.tocsr()  # convert to CSR format for faster solving
    chi = spsolve(A, b)
    chi = chi.reshape((n, n))

    chi = (chi - chi.min()) / (chi.max() - chi.min())  # normalize to [0, 1]
    return chi


def poisson_2d(points, normals, grid_size):
    Vx, Vy = create_vector_field(points, normals, grid_size)
    div = compute_divergence(Vx, Vy)
    chi = solve_poisson(div)
    contours = find_contours(chi, level=0.5)
    return contours


if __name__ == "__main__":
    points, normals = generate_circle_points(100)
    contours = poisson_2d(points, normals, grid_size=64)

    # Plot the result (potential field) and the points
    plt.figure(figsize=(6, 6))
    plt.imshow(chi, cmap="hot", interpolation="nearest")
    plt.colorbar()
    plt.title("Solution to Poisson Equation (Potential Field $\chi$)")

    # Plot the points on the grid
    points_grid_x = points[:, 0] * (Vx.shape[0] - 1)
    points_grid_y = points[:, 1] * (Vy.shape[1] - 1)
    plt.scatter(points_grid_x, points_grid_y, color="blue", s=30, marker="x", label="Circle Points")

    # Plot the contours from Marching Squares
    for contour in contours:
        plt.plot(contour[:, 1], contour[:, 0], color="green", lw=2, label="Isoline (Level 0.5)")

    plt.legend()

    plt.show()
