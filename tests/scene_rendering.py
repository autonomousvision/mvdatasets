import os
import sys
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

# load mvdatasets from parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mvdatasets.mvdataset import MVDataset
from mvdatasets.utils.profiler import Profiler
from mvdatasets.utils.rendering import render_scene
from mvdatasets.utils.geometry import linear_transformation_3d

# Set profiler
profiler = Profiler()  # nb: might slow down the code

# Set datasets path
datasets_path = "/home/stefano/Data"

# test dmsr
dataset_name = "dmsr"
scene_name = "dinning"
config = {}

# dataset loading
mv_data = MVDataset(
    dataset_name,
    scene_name,
    datasets_path,
    splits=["train", "test"],
)

# scene path (folder containing only mesh files)
scene_dir = "/home/stefano/Data/dmsr/dinning/meshes"

# get all files in scene dir
files = os.listdir(scene_dir)
nr_objects = len(files)
print(files)

# setup scene
o3d_scene = o3d.t.geometry.RaycastingScene()

# load meshes and add to scene
rotation = mv_data.global_transform[:3, :3]
translation = mv_data.global_transform[:3, 3]
for mesh_file in files:
    o3d_mesh = o3d.io.read_triangle_mesh(os.path.join(scene_dir, mesh_file))
    o3d_mesh_vertices = np.asarray(o3d_mesh.vertices)
    o3d_mesh_vertices = linear_transformation_3d(o3d_mesh_vertices, mv_data.global_transform)
    o3d_mesh.vertices = o3d.utility.Vector3dVector(o3d_mesh_vertices)
    o3d_mesh.compute_vertex_normals()
    o3d_scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(o3d_mesh))
    
# render mesh
splits = ["test", "train"]
for split in splits:
    save_path = os.path.join(datasets_path, dataset_name, scene_name, split)
    for camera in mv_data[split]:
        # print(camera.camera_idx)
        imgs = render_scene(camera, o3d_scene)
        geom_ids = imgs["geom_ids"]
        depth = imgs["depth"]
        # print(geom_ids.shape)
        # print(np.unique(geom_ids))
        plt.imshow(geom_ids, vmax=nr_objects)
        plt.show()
        break
        # save_nr = format(camera.camera_idx, "04d")
        # os.makedirs(os.path.join(save_path, "depth"), exist_ok=True)
        # os.makedirs(os.path.join(save_path, "instance_mask"), exist_ok=True)
        # np.save(os.path.join(save_path, "depth", f"d_{save_nr}"), depth)
        # np.save(os.path.join(save_path, "instance_mask", f"instance_mask_{save_nr}"), geom_ids)