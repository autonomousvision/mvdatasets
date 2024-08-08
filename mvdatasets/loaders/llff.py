from rich import print
import os
import copy
import numpy as np
import sys
import re
import pycolmap
import open3d as o3d
from tqdm import tqdm

from mvdatasets.scenes.camera import Camera
from mvdatasets.utils.geometry import rot_x_3d, deg2rad, get_min_max_cameras_distances
from mvdatasets.utils.pycolmap import read_points3D, read_cameras


def load_llff(
    scene_path,
    splits,
    config,
    verbose=False
):
    """llff data format loader

    Args:
        scene_path (str): path to the dataset scene folder
        splits (list): splits to load (e.g. ["train", "test"])
        config (dict): dict of config parameters

    Returns:
        cameras_splits (dict): dict of splits with lists of Camera objects
        global_transform (np.ndarray): (4, 4)
    """

    # CONFIG -----------------------------------------------------------------
    
    if "scene_type" not in config:
        config["scene_type"] = "bounded"
        if verbose:
            print(f"[bold yellow]WARNING[/bold yellow]: scene_type not in config, setting to {config['scene_type']}")
    else:
        valid_scene_types = ["bounded", "unbounded", "forward_facing"]
        if config["scene_type"] not in valid_scene_types:
            print(f"[bold red]ERROR[/bold red]: scene_type {config['scene_type']} must be a value in {valid_scene_types}")
            exit(1)
    
    if "translate_scene_x" not in config:
        config["translate_scene_x"] = 0.0
        if verbose:
            print(f"[bold yellow]WARNING[/bold yellow]: translate_scene_x not in config, setting to {config['translate_scene_x']}")
    
    if "translate_scene_y" not in config:
        config["translate_scene_y"] = 0.0
        if verbose:
            print(f"[bold yellow]WARNING[/bold yellow]: translate_scene_y not in config, setting to {config['translate_scene_y']}")
    
    if "translate_scene_z" not in config:
        config["translate_scene_z"] = 0.0
        if verbose:
            print(f"[bold yellow]WARNING[/bold yellow]: translate_scene_z not in config, setting to {config['translate_scene_z']}")
            
    if "rotate_scene_x_axis_deg" not in config:
        config["rotate_scene_x_axis_deg"] = 0.0
        if verbose:
            print(f"[bold yellow]WARNING[/bold yellow]: rotate_scene_x_axis_deg not in config, setting to {config['rotate_scene_x_axis_deg']}")
    
    if "test_camera_freq" not in config:
        config["test_camera_freq"] = 8
        if verbose:
            print(f"[bold yellow]WARNING[/bold yellow]: test_camera_freq not in config, setting to {config['test_camera_freq']}")
    
    if "train_test_overlap" not in config:
        config["train_test_overlap"] = False
        if verbose:
            print(f"[bold yellow]WARNING[/bold yellow]: train_test_overlap not in config, setting to {config['train_test_overlap']}")

    if "subsample_factor" not in config:
        config["subsample_factor"] = 1
        if verbose:
            print(f"[bold yellow]WARNING[/bold yellow]: subsample_factor not in config, setting to {config['subsample_factor']}")
    else:
        valid_subsample_factors = [1, 2, 4, 8]
        if config["subsample_factor"] not in valid_subsample_factors:
            raise ValueError(f"subsample_factor {config['subsample_factor']} must be a value in {valid_subsample_factors}")

    if "init_sphere_scale" not in config:
        config["init_sphere_scale"] = 0.1
        if verbose:
            print(f"[bold yellow]WARNING[/bold yellow]: init_sphere_scale not in config, setting to {config['init_sphere_scale']}")
    
    # if bounded, all scene content can be represented in the foreground bounding box
    if config["scene_type"] == "bounded":
        config["target_cameras_max_distance"] = 1.0
    # if unbounded, scene content extends beyond the foreground bounding sphere
    elif config["scene_type"] == "unbounded":
        config["target_cameras_max_distance"] = 0.5
    # if forward_facing, scene content is in front of the camera
    elif config["scene_type"] == "forward_facing":
        print("[bold red]ERROR[/bold red]: forward_facing scene type not implemented yet")
        exit(1)
    
    if verbose:
        print("load_llff config:")
        for k, v in config.items():
            print(f"\t{k}: {v}")
        
    # -------------------------------------------------------------------------
    
    # read colmap data
    
    reconstruction_path = os.path.join(scene_path, "sparse/0")
    reconstruction = pycolmap.Reconstruction(reconstruction_path)
    # print(reconstruction.summary())

    point_cloud = read_points3D(reconstruction)
    # point_cloud[:, 2] += config["translate_scene_z"]
    
    # get point cloud mean
    # point_cloud_mean = np.mean(point_cloud, axis=0)
    # only shift along z axis
    # point_cloud_mean[0] = 0
    # point_cloud_mean[1] = 0
    # point_cloud -= point_cloud_mean
    # # save point cloud as ply with o3d
    # o3d_point_cloud = o3d.geometry.PointCloud()
    # o3d_point_cloud.points = o3d.utility.Vector3dVector(point_cloud)
    # o3d.io.write_point_cloud(os.path.join("debug/point_clouds/mipnerf360", "garden.ply"), o3d_point_cloud)
    # exit()
    
    images_path = os.path.join(scene_path, "images")
    
    if config["subsample_factor"] > 1:
        subsample_factor = int(config["subsample_factor"])
        images_path += f"_{subsample_factor}"
    else:
        subsample_factor = 1
    
    cameras_meta = read_cameras(reconstruction, images_path)
    
    # # open poses_bounds.npy
    # poses_bounds_path = os.path.join(scene_path, "poses_bounds.npy")
    # # check if file exists
    # if os.path.exists(poses_bounds_path):
    #     poses_arr = np.load(poses_bounds_path)
    #     bounds = poses_arr[:, -2:]
    # else:
    #     bounds = np.array([0.01, 1.])
    
    # poses = []
    # for camera in cameras_meta:
    #     poses.append(camera["pose"])
    # poses = np.array(poses)
    
    # # TODO: forward_facing specific
    # if config["scene_type"] == "forward_facing":
    #     pass
    # else:
    #     # unbouded
    #     poses = unpad_poses(poses)
    #     # Rotate/scale poses to align ground with xy plane and fit to unit cube.
    #     poses, transform = transform_poses_pca(poses)
    #     poses = pad_poses(poses)

    #     print("transform", transform)
    #     print("poses", poses.shape)
        
    #     global_transform = transform
    
    # read images
    # images_list = sorted(os.listdir(images_path), key=lambda x: int(re.search(r'\d+', x).group()))
    poses_all = []
    pbar = tqdm(cameras_meta, desc="images", ncols=100)
    for i, camera in enumerate(pbar):
        
        traslation = camera["translation"]
        # traslation[2] += config["translate_scene_z"]
        
        w2c = np.eye(4)
        w2c[:3, :3] = camera["rotation"]
        w2c[:3, 3] = traslation
        c2w = np.linalg.inv(w2c)
        pose = c2w
        poses_all.append(pose)
        
    # center scene in the average camera position
    # scene_center = np.mean([np.mean(poses_all, axis=0)[:3, 3]], axis=0)
    # print("scene_center", scene_center)
    
    # poses_all_ = []
    # for pose in poses_all:
    #     pose_ = copy.deepcopy(pose)
    #     pose_[:3, 3] -= scene_center
    #     poses_all_.append(pose_)
    
    # find scene radius
    min_camera_distance, max_camera_distance = get_min_max_cameras_distances(poses_all)
    # print("min_camera_distance:", min_camera_distance)
    # print("max_camera_distance:", max_camera_distance)
    
    # define scene scale
    scene_scale = max_camera_distance
    # round to 2 decimals
    scene_scale = round(scene_scale, 2)
    
    # scene scale such that furthest away camera is at target distance
    scene_scale_mult = config["target_cameras_max_distance"] / (max_camera_distance + 1e-2)
    
    # # scene center as the point cloud center
    # point_cloud_ = point_cloud * scene_scale_mult
    # point_cloud_ -= np.mean(point_cloud_, axis=0)
    # points_norms = np.linalg.norm(point_cloud_, axis=1)
    # point_cloud_ = point_cloud_[points_norms < 0.5]
    # scene_center = np.mean(point_cloud_, axis=0)
    # scene_center[2] += config["translate_scene_z"]
    
    # global transform
    global_transform = np.eye(4)
    # rotate and scale
    rotate_scene_x_axis_deg = config["rotate_scene_x_axis_deg"]
    global_transform[:3, :3] = scene_scale_mult * rot_x_3d(deg2rad(rotate_scene_x_axis_deg))
    # translate
    translation_matrix = np.eye(4)
    translation_matrix[:3, 3] = [config["translate_scene_x"], config["translate_scene_y"], config["translate_scene_z"]]
    
    global_transform = translation_matrix @ global_transform
    
    # local transform
    local_transform = np.eye(4)
    
    # build cameras
    cameras_all = []
    pbar = tqdm(cameras_meta, desc="images", ncols=100)
    for i, camera in enumerate(pbar):
        
        pose = poses_all[i]
        
        # params
        params = camera["params"]
        intrinsics = np.eye(3)
        intrinsics[0, 0] = params["fx"] / subsample_factor
        intrinsics[1, 1] = params["fy"] / subsample_factor
        intrinsics[0, 2] = params["cx"] / subsample_factor
        intrinsics[1, 2] = params["cy"] / subsample_factor
        
        idx = camera["id"]
        cam_imgs = camera["img"][None, ...]
        
        camera = Camera(
            intrinsics=intrinsics,
            pose=pose,
            params=params,
            global_transform=global_transform,
            local_transform=local_transform,
            rgbs=cam_imgs,
            camera_idx=idx,
        )
        cameras_all.append(camera)
    
    # split cameras into train and test
    train_test_overlap = config["train_test_overlap"]
    test_camera_freq = config["test_camera_freq"]
    cameras_splits = {}
    for split in splits:
        cameras_splits[split] = []
        if split == "train":
            if train_test_overlap:
                # if train_test_overlap, use all cameras for training
                cameras_splits[split] = cameras_all
            # else use only a subset of cameras
            else:
                for i, camera in enumerate(cameras_all):
                    if i % test_camera_freq != 0:
                        cameras_splits[split].append(camera)
        if split == "test":
            # select a test camera every test_camera_freq cameras
            for i, camera in enumerate(cameras_all):
                if i % test_camera_freq == 0:
                    cameras_splits[split].append(camera)
    
    return {
        "cameras_splits": cameras_splits,
        "global_transform": global_transform,
        "point_clouds": [point_cloud],
        "config": config,
        "min_camera_distance": min_camera_distance,
        "max_camera_distance": max_camera_distance,
        "scene_scale": scene_scale,
        "scene_scale_mult": scene_scale_mult,
    }