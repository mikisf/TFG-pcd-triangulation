import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))  # Add the parent directory to sys.path
from marching_cubes import marching_cubes
from utils.data_to_ply import read_slices
from utils.create_obj import create_obj

import numpy as np

if __name__ == "__main__":
    slice_folder = "marching_cubes/data/CThead"  # Replace with the path to the folder containing 113 binary slice files
    resolution = (256, 256)

    aspect_ratio = (1, 1, 2)  # Aspect ratio of the voxel grid
    threshold = 0.5  # Adjust threshold based on intensity for the surface

    voxel_grid = read_slices(slice_folder, resolution)

    vertices, faces = marching_cubes(voxel_grid, threshold)

    aspect_matrix = np.diag(aspect_ratio)
    vertices = vertices @ aspect_matrix

    create_obj(vertices, faces, "marching_cubes/data/CThead050.obj")
