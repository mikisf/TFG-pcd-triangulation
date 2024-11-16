import numpy as np
from edgeTable import edgeTable
from triTable import triTable
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def interpolate_vertex(threshold, p1, p2, valp1, valp2):
    if abs(threshold - valp1) < 0.00001:
        return p1
    if abs(threshold - valp2) < 0.00001:
        return p2
    if abs(valp1 - valp2) < 0.00001:
        return p1
    mu = (threshold - valp1) / (valp2 - valp1)
    return p1 + mu * (p2 - p1)

def marching_cubes(grid, threshold):
    vertices = []
    faces = []
    nx, ny, nz = grid.shape

    for x in range(nx - 1):
        for y in range(ny - 1):
            for z in range(nz - 1):

                """
                z
                ^
                
                4---------[7]--------7
               /|                   /|
            [4] |                [6] |
            /   |                /   |
           5---------[5]--------6    |
           |    |               |    |
           |   [8]              |  [11]
           |    |               |    |
           |    0---------[3]---|----3  > y
          [9]  /              [10]  /
           | [0]                | [2]
           | /                  | /
           1---------[1]--------2

         x   
                """

                p0 = np.array([x, y, z])
                p1 = np.array([x + 1, y, z])
                p2 = np.array([x + 1, y + 1, z])
                p3 = np.array([x, y + 1, z])
                p4 = np.array([x, y, z + 1])
                p5 = np.array([x + 1, y, z + 1])
                p6 = np.array([x + 1, y + 1, z + 1])
                p7 = np.array([x, y + 1, z + 1])

                v0 = grid[p0[0]][p0[1]][p0[2]]
                v1 = grid[p1[0]][p1[1]][p1[2]]
                v2 = grid[p2[0]][p2[1]][p2[2]]
                v3 = grid[p3[0]][p3[1]][p3[2]]
                v4 = grid[p4[0]][p4[1]][p4[2]]
                v5 = grid[p5[0]][p5[1]][p5[2]]
                v6 = grid[p6[0]][p6[1]][p6[2]]
                v7 = grid[p7[0]][p7[1]][p7[2]]

                cube_index = 0
                if v0 < threshold: cube_index |= 1
                if v1 < threshold: cube_index |= 2
                if v2 < threshold: cube_index |= 4
                if v3 < threshold: cube_index |= 8
                if v4 < threshold: cube_index |= 16
                if v5 < threshold: cube_index |= 32
                if v6 < threshold: cube_index |= 64
                if v7 < threshold: cube_index |= 128

                print(cube_index)
                print(edgeTable[cube_index])
                print("{:03b}".format(edgeTable[cube_index]))

                if edgeTable[cube_index] == 0:
                    continue

                vertlist = [None] * 12
                if edgeTable[cube_index] & 1: vertlist[0] = interpolate_vertex(threshold, p0, p1, v0, v1)
                if edgeTable[cube_index] & 2: vertlist[1] = interpolate_vertex(threshold, p1, p2, v1, v2)
                if edgeTable[cube_index] & 4: vertlist[2] = interpolate_vertex(threshold, p2, p3, v2, v3)
                if edgeTable[cube_index] & 8: vertlist[3] = interpolate_vertex(threshold, p3, p0, v3, v0)
                if edgeTable[cube_index] & 16: vertlist[4] = interpolate_vertex(threshold, p4, p5, v4, v5)
                if edgeTable[cube_index] & 32: vertlist[5] = interpolate_vertex(threshold, p5, p6, v5, v6)
                if edgeTable[cube_index] & 64: vertlist[6] = interpolate_vertex(threshold, p6, p7, v6, v7)
                if edgeTable[cube_index] & 128: vertlist[7] = interpolate_vertex(threshold, p7, p4, v7, v4)
                if edgeTable[cube_index] & 256: vertlist[8] = interpolate_vertex(threshold, p0, p4, v0, v4)
                if edgeTable[cube_index] & 512: vertlist[9] = interpolate_vertex(threshold, p1, p5, v1, v5)
                if edgeTable[cube_index] & 1024: vertlist[10] = interpolate_vertex(threshold, p2, p6, v2, v6)
                if edgeTable[cube_index] & 2048: vertlist[11] = interpolate_vertex(threshold, p3, p7, v3, v7)

                print(vertlist)
                print(triTable[cube_index])

                for i in range(0, 16, 3):
                    if triTable[cube_index][i] == -1:
                        break
                    v1 = vertlist[triTable[cube_index][i]]
                    v2 = vertlist[triTable[cube_index][i + 1]]
                    v3 = vertlist[triTable[cube_index][i + 2]]
                    vertices.extend([v1, v2, v3])
                    faces.append([len(vertices) - 3, len(vertices) - 2, len(vertices) - 1])

    return np.array(vertices), np.array(faces)

#np.random.seed(0)
grid = np.random.rand(2, 2, 2)
grid = np.random.randint(2, size=(3, 3, 3))
threshold = 0.5
vertices, faces = marching_cubes(grid, threshold)

print("Vertices:")
print(vertices)
print("Faces:")
print(faces)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the points
x, y, z = np.where(grid < threshold)
ax.scatter(x, y, z, c='r', s=50, marker='o')

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
                [(i + v1[0], j + v1[1], k + v1[2]), (i + v2[0], j + v2[1], k + v2[2])]
                for v1, v2 in cube_edges_template
            ]
            for edge in cube_edges:
                edge_x, edge_y, edge_z = zip(*edge)
                ax.plot(edge_x, edge_y, edge_z, color='black', linewidth=1)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()