import math
import matplotlib.pyplot as plt

def circumcircle(p1, p2, p3):
    """Compute the circumcenter and squared radius of the circle passing through points p1, p2, and p3."""
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
    """Check if point p lies inside the circumcircle of triangle (p1, p2, p3)."""
    center, r2 = circumcircle(p1, p2, p3)
    if center is None:
        return False
    dx = p[0] - center[0]
    dy = p[1] - center[1]
    return (dx * dx + dy * dy) < r2 - 1e-9

def make_edge(i, j):
    """Return an order-independent edge represented as a tuple of indices."""
    return tuple(sorted([i, j]))

def delaunay_triangulation(points):
    """Compute the Delaunay triangulation using Bowyer-Watson with a symbolic super-triangle at infinity."""
    original_points = points.copy()
    
    # Step 1: Define symbolic infinite vertices
    inf_idx1 = len(points)
    inf_idx2 = len(points) + 1
    inf_idx3 = len(points) + 2
    points.extend([(-float('inf'), -float('inf')), (0, float('inf')), (float('inf'), 0)])

    # Initial triangulation with the symbolic infinite points
    triangulation = [(inf_idx1, inf_idx2, inf_idx3)]
    
    # Step 2: Incrementally insert each real point
    for idx, point in enumerate(original_points):
        bad_triangles = []
        for triangle in triangulation:
            i, j, k = triangle
            if in_circumcircle(point, points[i], points[j], points[k]):
                bad_triangles.append(triangle)

        # Step 3: Find the boundary of the hole
        polygon = set()
        for triangle in bad_triangles:
            edges = [make_edge(triangle[0], triangle[1]),
                     make_edge(triangle[1], triangle[2]),
                     make_edge(triangle[2], triangle[0])]
            for edge in edges:
                shared = False
                for other in bad_triangles:
                    if other == triangle:
                        continue
                    other_edges = [make_edge(other[0], other[1]),
                                   make_edge(other[1], other[2]),
                                   make_edge(other[2], other[0])]
                    if edge in other_edges:
                        shared = True
                        break
                if not shared:
                    polygon.add(edge)

        # Remove bad triangles
        triangulation = [t for t in triangulation if t not in bad_triangles]

        # Step 4: Re-triangulate the polygonal hole
        for edge in polygon:
            new_triangle = (edge[0], edge[1], idx)
            triangulation.append(new_triangle)

    # Step 5: Remove any triangle containing a symbolic infinite vertex
    final_triangles = [t for t in triangulation if inf_idx1 not in t and inf_idx2 not in t and inf_idx3 not in t]

    return final_triangles, original_points

def plot_triangulation(points, triangles, title="Delaunay Triangulation"):
    """Plot the triangulation using matplotlib."""
    plt.figure(figsize=(6, 6))
    for tri in triangles:
        triangle_points = [points[i] for i in tri]
        triangle_points.append(triangle_points[0])  # Close the triangle
        xs, ys = zip(*triangle_points)
        plt.plot(xs, ys, 'b-', lw=1)
    xs, ys = zip(*points)
    plt.plot(xs, ys, 'ro')
    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

# Example usage
if __name__ == "__main__":
    sample_points = [(0.5, 0.5), (0.8, 0.4), (0.3, 0.7), (0.4, 0.3)]
    triangles, orig_points = delaunay_triangulation(sample_points)
    
    print("Delaunay Triangulation (triangles as indices):")
    for tri in triangles:
        print(tri, "->", [orig_points[i] for i in tri])

    plot_triangulation(orig_points, triangles, title="Delaunay Triangulation (Symbolic Super-Triangle)")
