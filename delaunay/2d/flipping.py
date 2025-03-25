import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))) # Add the parent directory to sys.path
from utils import plot_triangulation, in_circumcircle

def edge_flip(triangles, points):
    """
    Iterate over all edges shared by two triangles.
    If the Delaunay condition is violated (the vertex opposite the edge in one triangle
    lies inside the circumcircle of the other), flip the edge.
    """
    # Build a mapping from each (sorted) edge to the list of triangles that contain it.
    edge_to_tri = {}
    for ti, tri in enumerate(triangles):
        for i in range(3):
            a, b = tri[i], tri[(i+1) % 3]
            edge = tuple(sorted((a, b)))
            edge_to_tri.setdefault(edge, []).append(ti)

    # Check each shared edge for legality.
    for edge, tris in edge_to_tri.items():
        if len(tris) == 2:
            t1_idx, t2_idx = tris
            t1 = triangles[t1_idx]
            t2 = triangles[t2_idx]
            a, b = edge
            # Find the vertices opposite the common edge in each triangle.
            c = next(x for x in t1 if x not in edge)
            d = next(x for x in t2 if x not in edge)
            # Check Delaunay condition: if d lies inside circumcircle of triangle (a, b, c)

            if in_circumcircle(points[d], points[a], points[b], points[c]):
                new_triangles = triangles.copy()
                new_triangles.remove(t1)
                new_triangles.remove(t2)
                new_triangles.append((c, d, a))
                new_triangles.append((c, d, b))
                return new_triangles, True
            # Alternatively, check if c is inside circumcircle of triangle (a, b, d)
            elif in_circumcircle(points[c], points[a], points[b], points[d]):
                new_triangles = triangles.copy()
                new_triangles.remove(t1)
                new_triangles.remove(t2)
                new_triangles.append((c, d, a))
                new_triangles.append((c, d, b))
                return new_triangles, True
    return triangles, False

def delaunay_triangulation(points, initial_triangles):
    """
    Starting from an initial triangulation, repeatedly apply edge flips until
    the Delaunay condition holds for every shared edge.
    """
    triangles = initial_triangles
    changed = True
    while changed:
        triangles, changed = edge_flip(triangles, points)
    return triangles

if __name__ == '__main__':
    points = [
        (0, 0),    # Point 0
        (1, 0),    # Point 1
        (0.9, 1),  # Point 2
        (0, 1),    # Point 3
        (0.5, 0.5), # Point 4
        (1.2, 0.2), # Point 5
        (-0.3, 0.7), # Point 6
        (0.3, -0.2),  # Point 7
        (1.5, 1.5),  # Point 8
        (-0.5, -0.5),  # Point 9
        (0.8, 0.3),  # Point 10
        (1.1, 1.2),  # Point 11
        (-0.6, 1.3),  # Point 12
        (0.2, 1.1),  # Point 13
        (1.4, -0.4),  # Point 14
        (-0.4, 0.2),  # Point 15
        (0.7, 1.4),  # Point 16
        (-1, -1),    # Point 17
        (1.8, 0.5),  # Point 18
        (0.3, 0.8)   # Point 19
    ]
    
    # An initial triangulation that covers all points (may not be Delaunay)
    initial_triangles = [
        (17, 9, 12),  # Triangle 0
        (9, 15, 12),  # Triangle 1
        (15, 6, 12),  # Triangle 2
        (3, 13, 12),  # Triangle 3
        (12, 13, 16),  # Triangle 4
        (12, 16, 8),  # Triangle 5
        (16, 11, 8),  # Triangle 6
        (8, 11, 18),  # Triangle 7
        (13, 11, 16),  # Triangle 8
        (13, 2, 11),  # Triangle 9
        (2, 18, 11),  # Triangle 10
        (3, 19, 13),  # Triangle 11
        (13, 19, 2),  # Triangle 12
        (2, 4, 18),  # Triangle 13
        (1, 5, 10),  # Triangle 14
        (14, 18, 5),  # Triangle 15
        (1, 14, 5),  # Triangle 16
        (14, 9, 17),  # Triangle 17
        (7, 9, 14),  # Triangle 18
        (0, 9, 7),  # Triangle 19
        (0, 15, 9),  # Triangle 20
        (0, 6, 15),  # Triangle 21
        (12, 4, 3),  # Triangle 22
        (3, 4, 19),  # Triangle 23
        (6, 10, 4),  # Triangle 24
        (6, 1, 10),  # Triangle 25
        (6, 14, 1),  # Triangle 26
        (0, 7, 14),  # Triangle 27
        (10, 5, 18),  # Triangle 28
        (6, 0, 14),  # Triangle 29
        (4, 10, 18),  # Triangle 30
        (4, 2, 19),  # Triangle 31
        (6, 4, 12),  # Triangle 32
    ]

    plot_triangulation(points, initial_triangles, title="Initial Triangulation")
    
    final_triangles = delaunay_triangulation(points, initial_triangles)
    plot_triangulation(points, final_triangles, title="Algorithm Delaunay Triangulation")
    
    from scipy.spatial import Delaunay
    import numpy as np
    points_np = np.array(points)
    scipy_delaunay = Delaunay(points_np)
    scipy_triangles = [tuple(sorted(tri)) for tri in scipy_delaunay.simplices]
    plot_triangulation(points, scipy_triangles, title="Scipy Delaunay Triangulation")

    final_triangles = sorted([tuple(sorted(t)) for t in final_triangles])
    scipy_triangles = sorted([tuple(sorted(t)) for t in scipy_triangles])
    print("Final triangles:", final_triangles)
    print("Scipy triangles:", scipy_triangles)
    print("Triangles match:", final_triangles == scipy_triangles)
    
    
