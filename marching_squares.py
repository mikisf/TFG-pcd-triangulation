import numpy as np
import matplotlib.pyplot as plt

MARCHING_SQUARES_LUT = {
    0: [],
    1: [(0, 3)],
    2: [(0, 1)],
    3: [(1, 3)],
    4: [(1, 2)],
    5: [(0, 3), (1, 2)],
    6: [(0, 2)],
    7: [(2, 3)],
    8: [(2, 3)],
    9: [(0, 2)],
    10: [(0, 1), (2, 3)],
    11: [(1, 2)],
    12: [(1, 3)],
    13: [(0, 1)],
    14: [(0, 3)],
    15: []
}

def interpolate(p1, p2, v1, v2, threshold):
    """Linear interpolation between two points p1 and p2 based on values v1 and v2."""
    t = (threshold - v1) / (v2 - v1)
    return p1 + t * (p2 - p1)

def marching_squares(grid, threshold):
    """Apply Marching Squares algorithm to a 2D grid."""
    nx, ny = grid.shape
    contours = []

    for x in range(nx - 1):
        for y in range(ny - 1):
            # Get values of the current cell's corners
            """
            y
            ^
            
            3 -- [2] -- 2
            |           |
           [3]         [1]
            |           |
            0 -- [0] -- 1    > x
            
            """

            p0 = np.array([x, y])
            p1 = np.array([x + 1, y])
            p2 = np.array([x + 1, y + 1])
            p3 = np.array([x, y + 1])

            v0 = grid[p0[0], p0[1]]
            v1 = grid[p1[0], p1[1]]
            v2 = grid[p2[0], p2[1]]
            v3 = grid[p3[0], p3[1]]
            
            square_index = 0
            if v0 < threshold: square_index |= 1
            if v1 < threshold: square_index |= 2
            if v2 < threshold: square_index |= 4
            if v3 < threshold: square_index |= 8

            edges = MARCHING_SQUARES_LUT[square_index]

            for (start_edge, end_edge) in edges:

                start_point = interpolate([p0, p1, p2, p3][start_edge], 
                                          [p0, p1, p2, p3][(start_edge + 1) % 4], 
                                          [v0, v1, v2, v3][start_edge], 
                                          [v0, v1, v2, v3][(start_edge + 1) % 4], 
                                          threshold)
                
                end_point = interpolate([p0, p1, p2, p3][end_edge], 
                                        [p0, p1, p2, p3][(end_edge + 1) % 4], 
                                        [v0, v1, v2, v3][end_edge], 
                                        [v0, v1, v2, v3][(end_edge + 1) % 4], 
                                        threshold)

                contours.append((start_point, end_point))

    return contours

# Test grid
np.random.seed(0)
# grid = np.random.randint(2, size=(10, 10))
grid = np.random.rand(10, 10)
contours = marching_squares(grid, 0.5)

# Plotting
plt.figure(figsize=(10, 10))

# Plot the grid
for i in range(grid.shape[0]):
    plt.plot([0, grid.shape[1] - 1], [i, i], color='black', linestyle='--', linewidth=1, zorder=0)
    plt.plot([i, i], [0, grid.shape[0] - 1], color='black', linestyle='--', linewidth=1, zorder=0)

# Plot the circles
x, y = np.indices(grid.shape)
plt.scatter(x.flatten(), y.flatten(), c=grid.flatten(), cmap='gray', s=500, edgecolor='black', linewidth=0.5, zorder=2)

# Add the values inside the circles
for i in range(grid.shape[0]):
    for j in range(grid.shape[1]):
        text_color = 'white' if grid[i, j] < 0.5 else 'black'
        plt.text(i, j, f'{grid[i, j]:.2f}', ha='center', va='center', color=text_color, fontsize=8, fontweight='bold', zorder=3)

# Plot the contours
for (start, end) in contours:
    plt.plot([start[0], end[0]], [start[1], end[1]], 'b-', linewidth=2)

plt.colorbar(label="Value")
plt.title("Marching Squares Contours")
plt.show()
