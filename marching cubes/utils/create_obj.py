import numpy as np


def create_obj(vertices, faces, file_path):
    """
    Creates a .obj file from vertices, faces, and calculated normals.

    Args:
        vertices (ndarray): Array of vertices (Nx3).
        faces (ndarray): Array of faces (Mx3).
        file_path (str): Path to save the .obj file. Defaults to 'output_with_normals.obj'.
    """

    with open(file_path, "w") as obj_file:
        # Write vertices
        for vertex in vertices:
            obj_file.write(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")

        # Write faces, referencing normals
        for face in faces:
            # Faces reference both vertex and normal indices
            face_line = " ".join([f"{v + 1}//{v + 1}" for v in face])  # OBJ format uses 1-based indexing
            obj_file.write(f"f {face_line}\n")

    print(f"OBJ file with normals created at {file_path}")
