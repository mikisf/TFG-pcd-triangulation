import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))  # Add the parent directory to sys.path
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

    all_scalars = [
        [0, 0, 0, 0],
        [1, 0, 0, 0],
        [1, 0, 0, 1],
    ]

    for i, scalars in enumerate(all_scalars, start=1):
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

        # Plot the edges of the tetrahedra
        for tet in tetrahedra:
            for edge in tet_edges:
                p1 = points[tet[edge[0]]]
                p2 = points[tet[edge[1]]]
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color="k")

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        ax.view_init(elev=20, azim=75)
        plt.axis('off')
        plt.xlim(-0., 1.)
        plt.ylim(-0., 1.)
        ax.set_zlim(-0., 1.)
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

        plt.savefig(f"marching tetrahedra/images/MT{i}.png")
        plt.close()
        #plt.show()
