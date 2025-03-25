import matplotlib.pyplot as plt

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
    ux = ((ax**2 + ay**2) * (by - cy) +
          (bx**2 + by**2) * (cy - ay) +
          (cx**2 + cy**2) * (ay - by)) / d
    uy = ((ax**2 + ay**2) * (cx - bx) +
          (bx**2 + by**2) * (ax - cx) +
          (cx**2 + cy**2) * (bx - ax)) / d
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
        plt.plot(xs, ys, 'b-', lw=1)
    # Plot the points
    xs, ys = zip(*points)
    plt.plot(xs, ys, 'ro')
    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()
