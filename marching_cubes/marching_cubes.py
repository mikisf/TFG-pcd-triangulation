import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))  # Add the parent directory to sys.path
from edgeTable import edgeTable

import numpy as np
from triTable import triTable
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from utils.create_obj import create_obj


def interpolate_vertex(threshold, p1, p2, valp1, valp2):
    if abs(threshold - valp1) < 0.00001:
        return p1
    if abs(threshold - valp2) < 0.00001:
        return p2
    if abs(valp1 - valp2) < 0.00001:
        return p1
    t = (threshold - valp1) / (valp2 - valp1)
    return p1 + t * (p2 - p1)


def marching_cubes(grid, threshold):
    vertices = []
    faces = []
    nx, ny, nz = grid.shape

    for x in range(nx - 1):
        for y in range(ny - 1):
            for z in range(nz - 1):
                # fmt: off
                """
                z
                ^
                
                4---------[4]--------5
               /|                   /|
            [7] |                [5] |
            /   |                /   |
           7---------[6]--------6    |
           |    |               |    |
           |   [8]              |   [9]
           |    |               |    |
           |    0---------[0]---|----1  > y
          [11]  /             [10]  /
           | [3]                | [1]
           | /                  | /
           3---------[2]--------2

         x
                """
                # fmt: on

                p0 = np.array([x, y, z])
                p1 = np.array([x, y + 1, z])
                p2 = np.array([x + 1, y + 1, z])
                p3 = np.array([x + 1, y, z])
                p4 = np.array([x, y, z + 1])
                p5 = np.array([x, y + 1, z + 1])
                p6 = np.array([x + 1, y + 1, z + 1])
                p7 = np.array([x + 1, y, z + 1])

                v0 = grid[p0[0]][p0[1]][p0[2]]
                v1 = grid[p1[0]][p1[1]][p1[2]]
                v2 = grid[p2[0]][p2[1]][p2[2]]
                v3 = grid[p3[0]][p3[1]][p3[2]]
                v4 = grid[p4[0]][p4[1]][p4[2]]
                v5 = grid[p5[0]][p5[1]][p5[2]]
                v6 = grid[p6[0]][p6[1]][p6[2]]
                v7 = grid[p7[0]][p7[1]][p7[2]]

                cube_index = 0
                cube_index |= 1 if v0 < threshold else 0
                cube_index |= 2 if v1 < threshold else 0
                cube_index |= 4 if v2 < threshold else 0
                cube_index |= 8 if v3 < threshold else 0
                cube_index |= 16 if v4 < threshold else 0
                cube_index |= 32 if v5 < threshold else 0
                cube_index |= 64 if v6 < threshold else 0
                cube_index |= 128 if v7 < threshold else 0

                if edgeTable[cube_index] == 0:
                    continue

                vertlist = [None] * 12
                vertlist[0] = interpolate_vertex(threshold, p0, p1, v0, v1) if edgeTable[cube_index] & 1 else None
                vertlist[1] = interpolate_vertex(threshold, p1, p2, v1, v2) if edgeTable[cube_index] & 2 else None
                vertlist[2] = interpolate_vertex(threshold, p2, p3, v2, v3) if edgeTable[cube_index] & 4 else None
                vertlist[3] = interpolate_vertex(threshold, p3, p0, v3, v0) if edgeTable[cube_index] & 8 else None
                vertlist[4] = interpolate_vertex(threshold, p4, p5, v4, v5) if edgeTable[cube_index] & 16 else None
                vertlist[5] = interpolate_vertex(threshold, p5, p6, v5, v6) if edgeTable[cube_index] & 32 else None
                vertlist[6] = interpolate_vertex(threshold, p6, p7, v6, v7) if edgeTable[cube_index] & 64 else None
                vertlist[7] = interpolate_vertex(threshold, p7, p4, v7, v4) if edgeTable[cube_index] & 128 else None
                vertlist[8] = interpolate_vertex(threshold, p0, p4, v0, v4) if edgeTable[cube_index] & 256 else None
                vertlist[9] = interpolate_vertex(threshold, p1, p5, v1, v5) if edgeTable[cube_index] & 512 else None
                vertlist[10] = interpolate_vertex(threshold, p2, p6, v2, v6) if edgeTable[cube_index] & 1024 else None
                vertlist[11] = interpolate_vertex(threshold, p3, p7, v3, v7) if edgeTable[cube_index] & 2048 else None

                for i in range(0, 16, 3):
                    if triTable[cube_index][i] == -1:
                        break
                    v3 = vertlist[triTable[cube_index][i]]
                    v2 = vertlist[triTable[cube_index][i + 1]]
                    v1 = vertlist[triTable[cube_index][i + 2]]
                    vertices.extend([v1, v2, v3])
                    faces.append([len(vertices) - 3, len(vertices) - 2, len(vertices) - 1])

    return np.array(vertices), np.array(faces)


if __name__ == "__main__":
    # np.random.seed(0)
    grid = np.random.rand(2, 2, 2)
    grid = np.random.randint(2, size=(3, 3, 3))

    threshold = 0.5
    vertices, faces = marching_cubes(grid, threshold)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Plot the points
    x, y, z = np.where(grid < threshold)
    ax.scatter(x, y, z, c="r", s=50, marker="o")

    # Plot the triangles
    for face in faces:
        triangle = vertices[face]
        tri = Poly3DCollection([triangle], alpha=0.5, edgecolor="k")
        tri.set_facecolor((0, 0, 1, 0.5))
        ax.add_collection3d(tri)

    # Plot the edges of the cubes
    # fmt: off
    cube_edges_template = [
        [(0, 0, 0), (1, 0, 0)], [(1, 0, 0), (1, 1, 0)], [(1, 1, 0), (0, 1, 0)], [(0, 1, 0), (0, 0, 0)],  # Bottom face
        [(0, 0, 1), (1, 0, 1)], [(1, 0, 1), (1, 1, 1)], [(1, 1, 1), (0, 1, 1)], [(0, 1, 1), (0, 0, 1)],  # Top face
        [(0, 0, 0), (0, 0, 1)], [(1, 0, 0), (1, 0, 1)], [(1, 1, 0), (1, 1, 1)], [(0, 1, 0), (0, 1, 1)],   # Vertical edges
    ]
    # fmt: on
    for i in range(grid.shape[0] - 1):
        for j in range(grid.shape[1] - 1):
            for k in range(grid.shape[2] - 1):
                cube_edges = [[(i + v3[0], j + v3[1], k + v3[2]), (i + v2[0], j + v2[1], k + v2[2])] for v3, v2 in cube_edges_template]
                for edge in cube_edges:
                    edge_x, edge_y, edge_z = zip(*edge)
                    ax.plot(edge_x, edge_y, edge_z, color="black", linewidth=1)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.show()

    # Create the .obj file
    # create_obj(vertices, faces, "output_with_normals.obj")

"""
if __name__ == "__main__":
    #np.random.seed(0)
    grid = np.random.rand(2, 2, 2)
    grid = np.random.randint(2, size=(2, 2, 2))

    grid1 = 1 - np.array([[[0, 0], [0, 0]], [[0, 0], [0, 0]]])
    grid2 = 1 - np.array([[[0, 0], [0, 0]], [[1, 0], [0, 0]]])
    grid3 = 1 - np.array([[[0, 0], [0, 0]], [[1, 0], [1, 0]]])
    grid4 = 1 - np.array([[[0, 0], [0, 0]], [[1, 0], [0, 1]]])
    grid5 = 1 - np.array([[[1, 0], [1, 0]], [[0, 0], [1, 0]]])
    grid6 = 1 - np.array([[[1, 0], [1, 0]], [[1, 0], [1, 0]]])
    grid7 = 1 - np.array([[[1, 0], [1, 0]], [[0, 1], [1, 0]]])
    grid8 = 1 - np.array([[[0, 1], [1, 0]], [[1, 0], [0, 1]]])
    grid9 = 1 - np.array([[[1, 1], [1, 0]], [[1, 0], [0, 0]]])
    grid10 = 1 - np.array([[[1, 1], [1, 0]], [[0, 0], [1, 0]]])
    grid11 = 1 - np.array([[[0, 0], [0, 1]], [[1, 0], [0, 0]]])
    grid12 = 1 - np.array([[[0, 0], [0, 1]], [[1, 0], [1, 0]]])
    grid13 = 1 - np.array([[[0, 0], [0, 1]], [[0, 1], [1, 0]]])
    grid14 = 1 - np.array([[[0, 0], [1, 1]], [[1, 1], [0, 0]]])
    grid15 = 1 - np.array([[[1, 0], [1, 1]], [[1, 0], [0, 0]]])

    grids = [grid1, grid2, grid3, grid4, grid5, grid6, grid7, grid8, grid9, grid10, grid11, grid12, grid13, grid14, grid15]
    #grids = [grid10]
    grids = [1 - np.array([[[0, 0], [0, 0]], [[0, 1], [0, 0]]])]
    grids = [grid15]

    for index, grid in enumerate(grids):

        threshold = 0.5
        vertices, faces = marching_cubes(grid, threshold)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot the points
        x, y, z = np.where(grid < threshold)
        ax.scatter(x, y, z, c='w', s=999, marker='o', edgecolors='k', alpha=1)

        for (i, j, k) in zip(x, y, z):
            text = ''
            if i == 0 and j == 0 and k == 0: text = 'e0'
            if i == 1 and j == 0 and k == 0: text = 'e3'
            if i == 1 and j == 1 and k == 0: text = 'e2'
            if i == 0 and j == 1 and k == 0: text = 'e1'
            if i == 0 and j == 0 and k == 1: text = 'e4'
            if i == 1 and j == 0 and k == 1: text = 'e7'
            if i == 1 and j == 1 and k == 1: text = 'e6'
            if i == 0 and j == 1 and k == 1: text = 'e5'

            ax.text(i, j, k, text, color='black', ha='center', va='center', fontsize=20, zorder=100)
            #ax.plot([i], [j], [k], marker='o', markersize=31.5, color='w', markeredgecolor='k', alpha=1, zorder=2)
        
        ax.text(0.5, 1+0.1, 0, 'a1', color='black', ha='center', va='center', fontsize=20, zorder=100)
        ax.text(1+0.1, 0.5, 0-0.05, 'a2', color='black', ha='center', va='center', fontsize=20, zorder=100)
        ax.text(0.5, 1 + 0.1, 1 - 0.05, 'a5', color='black', ha='center', va='center', fontsize=20, zorder=100)
        ax.text(0-0.05, 0.5, 1+0.05, 'a4', color='black', ha='center', va='center', fontsize=20, zorder=100)
        ax.text(1+0.07, 0-0.07, 0.5, 'a11', color='black', ha='center', va='center', fontsize=20, zorder=100)
        ax.text(0 + 0.15, 0, 0.5+0.1, 'a8', color='black', ha='center', va='center', fontsize=20, zorder=100)

        # Plot the triangles
        for face in faces:
            triangle = vertices[face]
            tri = Poly3DCollection([triangle], alpha=0.5, edgecolor='k', zorder=200)
            tri.set_facecolor((0, 0, 1, 0.5))
            ax.add_collection3d(tri)
        
        # Plot the edges of the cubes
        cube_edges_template = [
            [(0, 0, 0), (1, 0, 0)], [(1, 0, 0), (1, 1, 0)], [(1, 1, 0), (0, 1, 0)], [(0, 1, 0), (0, 0, 0)],  # Bottom face
            [(0, 0, 1), (1, 0, 1)], [(1, 0, 1), (1, 1, 1)], [(1, 1, 1), (0, 1, 1)], [(0, 1, 1), (0, 0, 1)],  # Top face
            [(0, 0, 0), (0, 0, 1)], [(1, 0, 0), (1, 0, 1)], [(1, 1, 0), (1, 1, 1)], [(0, 1, 0), (0, 1, 1)]   # Vertical edges
        ]
        for i in range(grid.shape[0] - 1):
            for j in range(grid.shape[1] - 1):
                for k in range(grid.shape[2] - 1):
                    cube_edges = [
                        [(i + v3[0], j + v3[1], k + v3[2]), (i + v2[0], j + v2[1], k + v2[2])]
                        for v3, v2 in cube_edges_template
                    ]
                    for edge in cube_edges:
                        edge_x, edge_y, edge_z = zip(*edge)
                        ax.plot(edge_x, edge_y, edge_z, color='black', linewidth=1, zorder=1)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        # Set the camera position
        ax.view_init(elev=20, azim=35)
        plt.axis('off')
        plt.xlim(-0., 1.)
        plt.ylim(-0., 1.)
        ax.set_zlim(-0., 1.)
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

        plt.savefig(f"C:/Users/\Miquel/Desktop/MC/Example2.png")
        plt.close()

        #plt.show()
"""
"""
if __name__ == "__main__":
    #np.random.seed(0)
    grid = np.random.rand(2, 2, 2)
    grid = np.random.randint(2, size=(2, 2, 2))
    grid = np.array([[[0, 0], [0, 0]], [[0, 0], [0, 0]]])

    threshold = 0.5
    vertices, faces = marching_cubes(grid, threshold)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the points
    x, y, z = np.where(grid < threshold)
    ax.scatter(x, y, z, c='w', s=999, marker='o', edgecolors='k', alpha=1, zorder=10)

    for (i, j, k) in zip(x, y, z):
        text = ''
        if i == 0 and j == 0 and k == 0: text = 'e3'
        if i == 1 and j == 0 and k == 0: text = 'e2'
        if i == 1 and j == 1 and k == 0: text = 'e1'
        if i == 0 and j == 1 and k == 0: text = 'e0'
        if i == 0 and j == 0 and k == 1: text = 'e7'
        if i == 1 and j == 0 and k == 1: text = 'e6'
        if i == 1 and j == 1 and k == 1: text = 'e5'
        if i == 0 and j == 1 and k == 1: text = 'e4'

        ax.text(i, j, k, text, color='black', ha='center', va='center', fontsize=20, zorder=100)

    ax.text(0.5, 1+0.05, 0+0.05, 'a0', color='black', ha='center', va='center', fontsize=20, zorder=100)
    ax.text(1+0.05, 0.5, 0-0.05, 'a1', color='black', ha='center', va='center', fontsize=20, zorder=100)
    ax.text(0.5, 0-0.06, 0-0.05, 'a2', color='black', ha='center', va='center', fontsize=20, zorder=100)
    ax.text(0-0.05, 0.5, 0+0.05, 'a3', color='black', ha='center', va='center', fontsize=20, zorder=100)
    ax.text(0.5, 1+0.05, 1+0.05, 'a4', color='black', ha='center', va='center', fontsize=20, zorder=100)
    ax.text(1, 0.5-0.05, 1-0.1, 'a5', color='black', ha='center', va='center', fontsize=20, zorder=100)
    ax.text(0.5-0.05, 0, 1-0.1, 'a6', color='black', ha='center', va='center', fontsize=20, zorder=100)
    ax.text(0-0.05, 0.5, 1+0.05, 'a7', color='black', ha='center', va='center', fontsize=20, zorder=100)
    ax.text(0+0.1, 1, 0.5+0.1, 'a8', color='black', ha='center', va='center', fontsize=20, zorder=100)
    ax.text(1+0.05, 1+0.05, 0.5, 'a9', color='black', ha='center', va='center', fontsize=20, zorder=100)
    ax.text(1-0.07, 0-0.07, 0.5-0.05, 'a10', color='black', ha='center', va='center', fontsize=20, zorder=100)
    ax.text(0-0.07, 0-0.07, 0.5, 'a11', color='black', ha='center', va='center', fontsize=20, zorder=100)

    # Plot the triangles
    for face in faces:
        triangle = vertices[face]
        tri = Poly3DCollection([triangle], alpha=0.5, edgecolor='k')
        tri.set_facecolor((0, 0, 1, 0.5))
        ax.add_collection3d(tri)

    # Plot the edges of the cubes
    cube_edges_template = [
        [(0, 0, 0), (1, 0, 0)], [(1, 0, 0), (1, 1, 0)], [(1, 1, 0), (0, 1, 0)], [(0, 1, 0), (0, 0, 0)],  # Bottom face
        [(0, 0, 1), (1, 0, 1)], [(1, 0, 1), (1, 1, 1)], [(1, 1, 1), (0, 1, 1)], [(0, 1, 1), (0, 0, 1)],  # Top face
        [(0, 0, 0), (0, 0, 1)], [(1, 0, 0), (1, 0, 1)], [(1, 1, 0), (1, 1, 1)], [(0, 1, 0), (0, 1, 1)]   # Vertical edges
    ]
    for i in range(grid.shape[0] - 1):
        for j in range(grid.shape[1] - 1):
            for k in range(grid.shape[2] - 1):
                cube_edges = [
                    [(i + v3[0], j + v3[1], k + v3[2]), (i + v2[0], j + v2[1], k + v2[2])]
                    for v3, v2 in cube_edges_template
                ]
                for edge in cube_edges:
                    edge_x, edge_y, edge_z = zip(*edge)
                    ax.plot(edge_x, edge_y, edge_z, color='black', linewidth=1)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Set the camera position
    ax.view_init(elev=20, azim=-55)
    plt.axis('off')
    plt.xlim(-0., 1.)
    plt.ylim(-0., 1.)
    ax.set_zlim(-0., 1.)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    plt.savefig(f"C:/Users/Miquel/Desktop/MC/As.png")
    plt.close()

    #plt.show()
"""
