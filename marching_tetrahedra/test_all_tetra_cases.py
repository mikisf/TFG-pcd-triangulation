from marching_tetrahedra import marching_tetrahedra, tet_edges

if __name__ == "__main__":
    points = [
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ]
    tetrahedra = [
        [0, 1, 2, 3],
    ]

    for i in range(16):
        binary_str = bin(i)[2:].zfill(4)
        scalars = [int(bit) for bit in binary_str]
        print("Scalars:", scalars)
        threshold = 0.5
        vertices, faces = marching_tetrahedra(points, tetrahedra, scalars, threshold)

        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        import matplotlib.pyplot as plt

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        # Paint red the points with scalar < threshold
        below_threshold = [i for i, val in enumerate(scalars) if val < threshold]
        if below_threshold:
            ax.scatter(*zip(*[points[i] for i in below_threshold]), color="r", s=100, label="Below Threshold")
        above_threshold = [i for i in range(len(points)) if i not in below_threshold]
        if above_threshold:
            ax.scatter(*zip(*[points[i] for i in above_threshold]), color="b", s=100, label="Above Threshold")

        # Plot the faces
        for face in faces:
            tri = [vertices[i] for i in face]
            poly3d = [[tri[0], tri[1], tri[2]]]
            ax.add_collection3d(Poly3DCollection(poly3d, facecolors="cyan", linewidths=1, edgecolors="r", alpha=0.25))

            # Calculate and plot the normal vector
            v1 = [tri[1][i] - tri[0][i] for i in range(3)]
            v2 = [tri[2][i] - tri[0][i] for i in range(3)]
            normal = [
                v1[1] * v2[2] - v1[2] * v2[1],
                v1[2] * v2[0] - v1[0] * v2[2],
                v1[0] * v2[1] - v1[1] * v2[0],
            ]
            # Normalize the normal vector
            length = sum(n**2 for n in normal) ** 0.5
            normal = [n / length for n in normal]
            # Plot the normal vector
            centroid = [sum(coord) / 3 for coord in zip(*tri)]
            ax.quiver(centroid[0], centroid[1], centroid[2], normal[0], normal[1], normal[2], length=0.2, color="g", label="Normal")

        # Plot the edges of the tetrahedra
        for tet in tetrahedra:
            for edge in tet_edges:
                p1 = points[tet[edge[0]]]
                p2 = points[tet[edge[1]]]
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color="k")

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        plt.show()
