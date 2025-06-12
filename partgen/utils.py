import time

import numpy as np
import open3d as o3d
import trimesh
import xatlas
from trellis.utils import postprocessing_utils


def timer_decorator(func):
    """ count the time cost of a specific function

    """

    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.2f} seconds to run.")
        return result

    return wrapper

def decimate_and_parameterize_process(mesh, result_queue):
    result = decimate_and_parameterize(mesh, simplify_ratio=0.95, verbose=True)
    result_queue.put(("mesh", result))


def decimate_process(mesh, result_queue):
    result = decimate(mesh, simplify_ratio=0.95, verbose=True)
    result_queue.put(("mesh", result))


def decimate(mesh: trimesh.Trimesh, simplify_ratio: float=0.95, verbose=False):
    tik = time.time()
    vertices, faces = postprocessing_utils.postprocess_mesh(
        mesh.vertices,
        mesh.faces,
        postprocess_mode="simplify",
        simplify_ratio=simplify_ratio,
        fill_holes=False,
        verbose=verbose,
    )
    print(
        "decimate finished in {}s, {}vertices and {}faces".format(
            time.time() - tik, vertices.shape, faces.shape
        )
    )
    return trimesh.Trimesh(vertices=vertices, faces=faces)


def decimate_and_parameterize(mesh, simplify_ratio=0.95, verbose=False):
    tik = time.time()
    vertices, faces = postprocessing_utils.postprocess_mesh(
        mesh.vertices,
        mesh.faces,
        postprocess_mode="simplify",
        simplify_ratio=simplify_ratio,
        fill_holes=False,
        verbose=verbose,
    )
    tik_uv = time.time()
    vmapping, indices, uvs = xatlas.parametrize(vertices, faces)
    print("uv unwarping finished in {}s".format(time.time() - tik_uv))

    vertices = vertices[vmapping]
    faces = indices
    print(
        "decimate_and_process finished in {}s, {}vertices and {}faces".format(
            time.time() - tik, vertices.shape, faces.shape
        )
    )
    return vertices, faces, uvs

def normalise_mesh(mesh: trimesh.Trimesh):
    # normalise to [-0.5, 0.5]
    # notice that in the part-retexturing, we don't need to do this normalization since it's already normalised as a whole
    vertices = np.asarray(mesh.vertices)
    aabb = np.stack([vertices.min(0), vertices.max(0)])
    center = (aabb[0] + aabb[1]) / 2
    scale = (aabb[1] - aabb[0]).max()
    mesh.vertices = (vertices - center) / scale
    return mesh, center, scale

def denormalise_mesh(mesh: trimesh.Trimesh, center: np.ndarray, scale: float):
    # denormalise to original scale
    vertices = np.asarray(mesh.vertices)
    mesh.vertices = vertices * scale + center
    return mesh

def rotate_mesh_to_zup(mesh: trimesh.Trimesh):
    # rotate mesh to z-up
    # this is used for compatibility with trellis
    mesh.vertices = (
        mesh.vertices @ np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
    )
    return mesh

def voxelize(mesh: trimesh.Trimesh, resolution: int = 64, do_normalization: bool = False, is_input_yup: bool = False):
    mesh = mesh.copy()
    if do_normalization:
        mesh, _, _ = normalise_mesh(mesh) 
    
    if is_input_yup:
        mesh = rotate_mesh_to_zup(mesh)
    
    try:
        mesh = mesh.as_open3d
    except Exception as e:
        mesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(mesh.vertices),
            o3d.utility.Vector3iVector(mesh.faces),
        )
    # clamp vertices to the range [-0.5, 0.5]
    vertices = np.clip(np.asarray(mesh.vertices), -0.5 + 1e-6, 0.5 - 1e-6)
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(
        mesh,
        voxel_size=1 / resolution,
        min_bound=(-0.5, -0.5, -0.5),
        max_bound=(0.5, 0.5, 0.5),
    )
    vertices = np.array([voxel.grid_index for voxel in voxel_grid.get_voxels()])
    binary_voxel = np.zeros((resolution, resolution, resolution), dtype=bool)
    binary_voxel[vertices[:, 0], vertices[:, 1], vertices[:, 2]] = True
    return binary_voxel
