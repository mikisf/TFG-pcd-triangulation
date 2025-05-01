import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))) # Add the parent directory to sys.path
from utils import in_circumsphere, plot_3d_triangulation

def delaunay_tetrahedralization(points):
    """
    Compute the Delaunay triangulation for a set of points (list of (x, y, z) tuples)
    using the Bowyer-Watson algorithm. Tetrahedra are represented as tuples of indices.
    
    Returns:
      - tetrahedra: list of tetrahedra (each a tuple of indices referring to original_points)
    """
    original_points = points.copy()  # Keep the original list for output
    pts = points.copy()  # Work on a copy so we can add super-tetrahedron vertices

    # Step 1: Create a "super-tetrahedron" that encloses all the points.
    p1 = (-10_000, -10_000, -10_000)
    p2 = (10_000, 0, 0)
    p3 = (0, 10_000, 0)
    p4 = (0, 0, 10_000)
    
    # Append super-tetrahedron vertices to pts.
    super_idx1 = len(pts)
    pts.append(p1)
    super_idx2 = len(pts)
    pts.append(p2)
    super_idx3 = len(pts)
    pts.append(p3)
    super_idx4 = len(pts)
    pts.append(p4)
    
    # The initial tetrahedralization contains only the super-tetrahedron.
    tetrahedralization = [(super_idx1, super_idx2, super_idx3, super_idx4)]
    
    # Step 2: Add each point (by index) from the original points.
    # Note: indices for original points remain 0 ... len(original_points)-1.
    for idx, point in enumerate(original_points):
        bad_tetrahedra = []
        # Find all tetrahedra whose circumspheres contain the point.
        for tetrahedron in tetrahedralization:
            i, j, k, l = tetrahedron
            if in_circumsphere(point, pts[i], pts[j], pts[k], pts[l]):
                bad_tetrahedra.append(tetrahedron)

        # Step 3: Find the boundary of the hole.
        boundary = set()
        for tetrahedron in bad_tetrahedra:
            faces = [
                tuple(sorted((tetrahedron[0], tetrahedron[1], tetrahedron[2]))),
                tuple(sorted((tetrahedron[0], tetrahedron[1], tetrahedron[3]))),
                tuple(sorted((tetrahedron[0], tetrahedron[2], tetrahedron[3]))),
                tuple(sorted((tetrahedron[1], tetrahedron[2], tetrahedron[3])))
            ]
            for face in faces:
                shared = False
                for other in bad_tetrahedra:
                    if other == tetrahedron:
                        continue
                    other_faces = [
                        tuple(sorted((other[0], other[1], other[2]))),
                        tuple(sorted((other[0], other[1], other[3]))),
                        tuple(sorted((other[0], other[2], other[3]))),
                        tuple(sorted((other[1], other[2], other[3])))
                    ]
                    if face in other_faces:
                        shared = True
                        break
                if not shared:
                    boundary.add(face)
        
        # Remove the bad tetrahedra from the tetrahedralization.
        tetrahedralization = [t for t in tetrahedralization if t not in bad_tetrahedra]

        # Step 4: Re-tetrahedralize the hole.
        for face in boundary:
            new_tetrahedron = (face[0], face[1], face[2], idx)
            tetrahedralization.append(new_tetrahedron)
    
    # Step 5: Remove triangles that reference any of the super-triangle vertices.
    final_tetrahedra = [t for t in tetrahedralization if (super_idx1 not in t and 
                                                          super_idx2 not in t and 
                                                          super_idx3 not in t and
                                                          super_idx4 not in t)]
    
    return final_tetrahedra

# Example usage:
if __name__ == "__main__":
    # Generate random sample points within a unit square (0, 0, 0) to (10, 10, 10).
    import random
    seed = random.randint(0, 500)
    random.seed(seed)
    print("Seed:", seed)
    points = [(random.random(), random.random(), random.random()) for _ in range(20)]

    delaunay_tetrahedra = delaunay_tetrahedralization(points)
    
    from scipy.spatial import Delaunay
    scipy_delaunay = Delaunay(points)
    scipy_tetras = [tuple(sorted(tetra)) for tetra in scipy_delaunay.simplices]

    print("Tetraedra match:", sorted(delaunay_tetrahedra) == sorted(scipy_tetras))
    if sorted(delaunay_tetrahedra) != sorted(scipy_tetras): # Because the super tetrahedron is not big enough
        print("Tetrahedra do not match!")
        print("Difference:", set(sorted(scipy_tetras)) - set(sorted(delaunay_tetrahedra)))
    
    # Visualize the tetrahedra
    faces = []
    for tetra in delaunay_tetrahedra:
        faces.extend([
            (tetra[0], tetra[1], tetra[2]),
            (tetra[0], tetra[1], tetra[3]),
            (tetra[0], tetra[2], tetra[3]),
            (tetra[1], tetra[2], tetra[3])
        ])
    plot_3d_triangulation(points, faces, title="Delaunay Tetrahedralization")

    # Visualize the surface reconstruction
    face_count = {}
    for tetra in delaunay_tetrahedra:
        faces = [
            (tetra[0], tetra[1], tetra[2]),
            (tetra[0], tetra[1], tetra[3]),
            (tetra[0], tetra[2], tetra[3]),
            (tetra[1], tetra[2], tetra[3])
        ]
        for face in faces:
            face_count[face] = face_count.get(face, 0) + 1
    
    faces = [face for face, count in face_count.items() if count == 1]

    plot_3d_triangulation(points, faces, title="3D Delaunay Triangulation")
