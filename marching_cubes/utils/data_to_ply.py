import os
import numpy as np
import re


def natural_sort_key(file_name):
    match = re.search(r"\d+", file_name)
    return int(match.group()) if match else file_name


def read_slices(slice_folder, resultion):
    """
    Reads 16-bit binary files and creates a 3D voxel grid.
    """

    slice_files = sorted(os.listdir(slice_folder), key=natural_sort_key)

    nx, ny = resultion
    nz = len(slice_files)
    voxel_grid = np.zeros((nx, ny, nz), dtype=np.float32)

    for z, file in enumerate(slice_files):
        file_path = os.path.join(slice_folder, file)
        with open(file_path, "rb") as f:
            slice_data = np.frombuffer(f.read(), dtype=">i2").reshape((nx, ny))
            voxel_grid[:, :, z] = slice_data

    voxel_grid = (voxel_grid - np.min(voxel_grid)) / (np.max(voxel_grid) - np.min(voxel_grid))

    return voxel_grid


def voxel_grid_to_ply(voxel_grid, threshold, ply_path):
    vertices = np.argwhere(voxel_grid > threshold)
    # Prepare PLY header
    ply_header = f"""ply
format ascii 1.0
element vertex {len(vertices)}
property float x
property float y
property float z
end_header
"""

    # Prepare vertex list
    vertex_list = []
    for vertex in vertices:
        x, y, z = vertex
        vertex_list.append(f"{x} {y} {2 * z}")

    # Write to PLY file
    with open(ply_path, "w") as ply_file:
        ply_file.write(ply_header)
        ply_file.write("\n".join(vertex_list))

    print(f"PLY file saved to {ply_path}")
