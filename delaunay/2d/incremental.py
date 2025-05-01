import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))) # Add the parent directory to sys.path
from utils import plot_triangulation, in_circumcircle

def delaunay_triangulation(points):
    """
    Compute the Delaunay triangulation for a set of points (list of (x, y) tuples)
    using the Bowyer-Watson algorithm. Triangles are represented as tuples of indices.
    
    Returns:
        - triangles: list of triangles (each a tuple of indices referring to original_points)
    """
    original_points = points.copy()  # Keep the original list for output
    pts = points.copy()  # Work on a copy so we can add super-triangle vertices

    # Step 1: Create a "super-triangle" that encloses all the points.
    p1 = (-10_000_000, -10_000_000)
    p2 = (0, 10_000_000)
    p3 = (10_000_000, 0)
    
    # Append super-triangle vertices to pts.
    super_idx1 = len(pts)
    pts.append(p1)
    super_idx2 = len(pts)
    pts.append(p2)
    super_idx3 = len(pts)
    pts.append(p3)
    
    # The initial triangulation contains only the super-triangle.
    triangulation = [(super_idx1, super_idx2, super_idx3)]
    
    # Step 2: Add each point (by index) from the original points.
    # Note: indices for original points remain 0 ... len(original_points)-1.
    for idx, point in enumerate(original_points):
        bad_triangles = []
        # Find all triangles whose circumcircles contain the point.
        for triangle in triangulation:
            i, j, k = triangle
            if in_circumcircle(point, pts[i], pts[j], pts[k]):
                bad_triangles.append(triangle)

        # Step 3: Find the boundary (polygon) of the hole.
        polygon = set()
        for triangle in bad_triangles:
            edges = [tuple(sorted((triangle[0], triangle[1]))),
                     tuple(sorted((triangle[1], triangle[2]))),
                     tuple(sorted((triangle[2], triangle[0])))]
            for edge in edges:
                shared = False
                for other in bad_triangles:
                    if other == triangle:
                        continue
                    other_edges = [tuple(sorted((other[0], other[1]))),
                                   tuple(sorted((other[1], other[2]))),
                                   tuple(sorted((other[2], other[0])))]
                    if edge in other_edges:
                        shared = True
                        break
                if not shared:
                    polygon.add(edge)

        # Remove the bad triangles from the triangulation.
        triangulation = [t for t in triangulation if t not in bad_triangles]
        
        # Step 4: Re-triangulate the polygonal hole.
        for edge in polygon:
            new_triangle = (edge[0], edge[1], idx)
            triangulation.append(new_triangle)
                    
    # Step 5: Remove triangles that reference any of the super-triangle vertices.
    final_triangles = [t for t in triangulation if (super_idx1 not in t and 
                                                    super_idx2 not in t and 
                                                    super_idx3 not in t)]
    
    return final_triangles

# Example usage:
if __name__ == "__main__":
    # Generate random sample points within a unit square (0, 0) to (1, 1).
    import random
    points = [(random.random(), random.random()) for _ in range(20)]
    
    triangles = delaunay_triangulation(points)
    plot_triangulation(points, triangles, title="Delaunay Triangulation")

    from scipy.spatial import Delaunay
    import numpy as np
    points_np = np.array(points)
    scipy_delaunay = Delaunay(points_np)
    scipy_triangles = [tuple(sorted(tri)) for tri in scipy_delaunay.simplices]
    plot_triangulation(points, scipy_triangles, title="Scipy Delaunay Triangulation")

    triangles = sorted([tuple(sorted(t)) for t in triangles])
    scipy_triangles = sorted([tuple(sorted(t)) for t in scipy_triangles])
    print("Final triangles:", triangles)
    print("Scipy triangles:", scipy_triangles)
    print("Triangles match:", triangles == scipy_triangles)
