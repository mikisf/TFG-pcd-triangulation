import matplotlib.pyplot as plt
import math
import numpy as np


def circumcircle(p1, p2, p3):
    """
    Compute the circumcenter and squared radius of the circle
    passing through points p1, p2, and p3.
    """
    ax, ay = p1
    bx, by = p2
    cx, cy = p3
    d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(d) < 1e-12:
        return None, None  # Points are collinear
    ux = ((ax**2 + ay**2) * (by - cy) + (bx**2 + by**2) * (cy - ay) + (cx**2 + cy**2) * (ay - by)) / d
    uy = ((ax**2 + ay**2) * (cx - bx) + (bx**2 + by**2) * (ax - cx) + (cx**2 + cy**2) * (bx - ax)) / d
    center = (ux, uy)
    r2 = (ax - ux) ** 2 + (ay - uy) ** 2
    return center, r2


def in_circumcircle(p, p1, p2, p3):
    """
    Check if point p lies inside the circumcircle of triangle (p1, p2, p3).
    A small tolerance is used to avoid numerical issues.
    """
    center, r2 = circumcircle(p1, p2, p3)
    if center is None:
        return False
    dx = p[0] - center[0]
    dy = p[1] - center[1]
    return (dx * dx + dy * dy) < r2 - 1e-9


def angle(A, B, C):
    BA = (A[0] - B[0], A[1] - B[1])
    BC = (C[0] - B[0], C[1] - B[1])
    dot_product = BA[0] * BC[0] + BA[1] * BC[1]
    cross_product = BA[0] * BC[1] - BA[1] * BC[0]
    angle_rad = math.atan2(cross_product, dot_product)
    if angle_rad < 0:
        angle_rad += 2 * math.pi
    return angle_rad


def in_circumsphere(p, p1, p2, p3, p4):
    """
    Determines if point `p` is inside the circumsphere of the tetrahedron (p1, p2, p3, p4).
    Returns True if inside, False otherwise.
    """

    def to_homogeneous(q):
        return [q[0], q[1], q[2], np.dot(q, q), 1.0]

    M = np.array([to_homogeneous(p1), to_homogeneous(p2), to_homogeneous(p3), to_homogeneous(p4), to_homogeneous(p)])
    orient = np.linalg.det(M[:4, [0, 1, 2, 4]])  # determinant of orientation of tetrahedron
    det = np.linalg.det(M)
    return det * orient > 0


def plot_triangulation(points, triangles, title="Delaunay Triangulation"):
    """
    Plot the triangulation using matplotlib.
    """
    plt.figure(figsize=(6, 6))
    # Plot each triangle
    for tri in triangles:
        triangle_points = [points[i] for i in tri]
        # Close the triangle by repeating the first point
        triangle_points.append(triangle_points[0])
        xs, ys = zip(*triangle_points)
        plt.plot(xs, ys, "b-", lw=1)
    # Plot the points
    xs, ys = zip(*points)
    plt.plot(xs, ys, "ro")
    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.show()


def plot_3d_triangulation(points, faces, title="3D Delaunay Triangulation"):
    """
    Plot the 3D triangulation using matplotlib.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    for face in faces:
        verts = np.array([points[i] for i in face])
        ax.add_collection3d(Poly3DCollection([verts], facecolors="cyan", linewidths=1, edgecolors="k", alpha=0.3))
    points = np.array(points)
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c="r")
    ax.set_title(title)
    plt.show()
