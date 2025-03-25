from rich import print
from pathlib import Path
import os
import json
import numpy as np
from pycolmap import SceneManager
from tqdm import tqdm
from PIL import Image
from copy import deepcopy
from pyquaternion import Quaternion
from mvdatasets import Camera
from mvdatasets.geometry.primitives.point_cloud import PointCloud
from mvdatasets.utils.loader_utils import rescale
from mvdatasets.geometry.common import rot_euler_3d_deg
from mvdatasets.geometry.projections import radial_to_z_depth
from mvdatasets.utils.printing import print_warning
import matplotlib.pyplot as plt


def load(
    dataset_path: Path,
    scene_name: str,
    config: dict,
    verbose: bool = False,
):
    """Kubric data format loader.

    Args:
        dataset_path (Path): Path to the dataset folder.
        scene_name (str): Name of the scene / sequence to load.
        splits (list): Splits to load (e.g., ["train", "val"]).
        config (DatasetConfig): Dataset configuration parameters.
        verbose (bool, optional): Whether to print debug information. Defaults to False.

    Returns:
        dict: Dictionary of splits with lists of Camera objects.
        np.ndarray: Global transform (4, 4)
        str: Scene type
        List[PointCloud]: List of PointClouds
        float: Minimum camera distance
        float: Maximum camera distance
        float: Foreground scale multiplier
        float: Scene radius
        int: Number of frames per camera
        int: Number of sequence frames
        float: Frames per second
    """

    scene_path = dataset_path / scene_name
    splits = config["splits"]

    # Valid values for specific keys
    valid_values = {}

    # Validate specific keys
    for key, valid in valid_values.items():
        if key in config and config[key] not in valid:
            raise ValueError(f"{key} {config[key]} must be a value in {valid}")

    # Debugging output
    if verbose:
        print("config:")
        for k, v in config.items():
            print(f"\t{k}: {v}")

    # -------------------------------------------------------------------------

    # get all rgbs paths
    rgbs_dir = scene_path / "rgba"
    rgbs_paths = sorted(list(rgbs_dir.glob("*.png")))

    # get all depth paths
    depths_dir = scene_path / "depth"
    depths_paths = sorted(list(depths_dir.glob("*.tiff")))

    # get all segmentation paths
    segs_dir = scene_path / "segmentation"
    segs_paths = sorted(list(segs_dir.glob("*.png")))

    # open metadata.json
    metadata_path = scene_path / "metadata.json"
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    # # load absolute objects poses from dataset
    # instances = metadata["instances"]
    # instances_poses = {}
    # for idx, instance_metadata in enumerate(instances):
    #     instances_poses[idx] = []
    #     all_positions = []
    #     all_quaternions = []
    #     for position in instance_metadata["positions"]:
    #         all_positions.append(np.array(position))
    #     for quaternion in instance_metadata["quaternions"]:
    #         all_quaternions.append(Quaternion(quaternion))  # [w, x, y, z]
    #     # convert to rigid transformations
    #     for i in range(len(all_positions)):
    #         pose = np.eye(4)
    #         pose[:3, 3] = all_positions[i]
    #         pose[:3, :3] = all_quaternions[i].rotation_matrix
    #         instances_poses[idx].append(pose)

    # # calculate relative rotations between poses
    # instances_rot_relative = {}
    # for idx, poses in instances_poses.items():
    #     instances_rot_relative[idx] = [np.eye(3)]
    #     for i in range(1, len(poses)):
    #         rot_prec = poses[i - 1][:3, :3]
    #         rot_cur = poses[i][:3, :3]
    #         rot_relative = rot_prec.transpose() @ rot_cur
    #         instances_rot_relative[idx].append(rot_relative)

    # instances_poses_reframed = {}
    # for idx, poses in instances_poses.items():
    #     first_pose = np.eye(4)
    #     first_pose[:3, 3] = poses[0][:3, 3]  # copy position
    #     instances_poses_reframed[idx] = [first_pose]
    #     # reframe rotation of subsequent poses
    #     for i in range(1, len(poses)):
    #         rot_relative = instances_rot_relative[idx][i]
    #         rot_prec_reframed = instances_poses_reframed[idx][-1][:3, :3]
    #         rot_cur_reposed = rot_prec_reframed @ rot_relative
    #         pose_cur_reposed = np.eye(4)
    #         pose_cur_reposed[:3, :3] = rot_cur_reposed
    #         pose_cur_reposed[:3, 3] = poses[i][:3, 3]
    #         instances_poses_reframed[idx].append(pose_cur_reposed)

    # instances_poses = instances_poses_reframed

    # # TODO: remove
    # print("valid idxs", list(instances_poses.keys()))
    # dyn_idx = 0
    # all_centers = []
    # all_rots = []
    # for i in range(len(instances_poses[dyn_idx])):
    #     pose = instances_poses[dyn_idx][i]
    #     all_centers.append(pose[:3, 3])
    #     all_rots.append(pose[:3, :3])
    # sequence_lenght = 30
    # linewidth = 1.0

    # fig = plt.figure(figsize=(15, 15))
    # ax = fig.add_subplot(111, projection="3d")
    # # sample colors
    # colors = plt.cm.jet(np.linspace(0, 1, sequence_lenght))[:, :3]
    # for i in range(sequence_lenght - 1):

    #     # center at timestamp i
    #     center_i = all_centers[i]  # (3,)
    #     print("center_i", center_i)
    #     # center at timestamp i+1
    #     center_i1 = all_centers[i + 1]  # (3,)
    #     # get color
    #     color = colors[i]
    #     # alpha
    #     alpha = 1.0
    #     # plot line segment
    #     ax.plot3D(
    #         [center_i[0], center_i1[0]],
    #         [center_i[1], center_i1[1]],
    #         [center_i[2], center_i1[2]],
    #         color=color,
    #         alpha=alpha,
    #         linewidth=linewidth,
    #     )

    # scale = 0.1
    # up = "z"
    # for i in range(sequence_lenght):

    #     label = f"{i}"
    #     pose = instances_poses[dyn_idx][i]

    #     # get axis directions (normalized)
    #     x_dir = pose[:3, 0]
    #     x_dir /= np.linalg.norm(x_dir)
    #     y_dir = pose[:3, 1]
    #     y_dir /= np.linalg.norm(y_dir)
    #     z_dir = pose[:3, 2]
    #     z_dir /= np.linalg.norm(z_dir)
    #     # frame center
    #     pos = pose[:3, 3]

    #     # draw camera frame
    #     ax.quiver(
    #         pos[0],  # x
    #         pos[1] if up == "z" else pos[2],  # y
    #         pos[2] if up == "z" else pos[1],  # z
    #         x_dir[0],
    #         x_dir[1] if up == "z" else x_dir[2],
    #         x_dir[2] if up == "z" else x_dir[1],
    #         length=scale,
    #         color="r",
    #     )
    #     ax.quiver(
    #         pos[0],  # x
    #         pos[1] if up == "z" else pos[2],  # y
    #         pos[2] if up == "z" else pos[1],  # z
    #         y_dir[0],
    #         y_dir[1] if up == "z" else y_dir[2],
    #         y_dir[2] if up == "z" else y_dir[1],
    #         length=scale,
    #         color="g",
    #     )
    #     ax.quiver(
    #         pos[0],  # x
    #         pos[1] if up == "z" else pos[2],  # y
    #         pos[2] if up == "z" else pos[1],  # z
    #         z_dir[0],
    #         z_dir[1] if up == "z" else z_dir[2],
    #         z_dir[2] if up == "z" else z_dir[1],
    #         length=scale,
    #         color="b",
    #     )
    #     ax.text(
    #         pos[0],  # x
    #         pos[1] if up == "z" else pos[2],  # y
    #         pos[2] if up == "z" else pos[1],  # z
    #         label,
    #     )

    # plt.show()

    # get sequence length
    sequence_length = metadata["metadata"]["num_frames"]
    frame_rate = metadata["metadata"]["frame_rate"]
    resolution = metadata["metadata"]["resolution"]

    print("sequence_length", sequence_length)
    print("frame_rate", frame_rate)
    print("resolution", resolution)

    # get camera intrinsics
    focal_lenght = metadata["camera"]["focal_length"]
    sensor_width = metadata["camera"]["sensor_width"]
    # fx = focal lenght (mm) x image width (px) / sensor width (mm)
    fx = focal_lenght * resolution[0] / sensor_width
    fy = focal_lenght * resolution[1] / sensor_width
    cx = resolution[0] / 2
    cy = resolution[1] / 2
    intrinsics = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

    all_poses = []
    for i in range(sequence_length):
        # Get the camera pose info
        transl = metadata["camera"]["positions"][i]
        quat = metadata["camera"]["quaternions"][i]  # [w, x, y, z]
        quat = Quaternion(quat)
        rot = quat.rotation_matrix
        pose = np.eye(4)
        pose[:3, :3] = rot
        pose[:3, 3] = transl
        all_poses.append(pose)

    # rescale (optional)
    scene_radius_mult, min_camera_distance, max_camera_distance = rescale(
        all_poses, to_distance=config["max_cameras_distance"]
    )

    # scene radius
    scene_radius = max_camera_distance

    # global transform
    global_transform = np.eye(4)
    # rotate and scale
    rot = rot_euler_3d_deg(
        config["rotate_deg"][0], config["rotate_deg"][1], config["rotate_deg"][2]
    )
    global_transform[:3, :3] = scene_radius_mult * rot

    # local transform (flip y and z axis)
    local_transform = np.eye(4)
    local_transform[:3, :3] = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])

    # load data
    all_rgbs = None
    all_depths = None
    all_masks = None
    all_segs = None
    if config["pose_only"]:
        pass
    else:
        # load all rgbs
        all_rgbs = []
        pbar_rgbs = tqdm(rgbs_paths, desc="rgbs", ncols=100)
        for rgb_path in pbar_rgbs:
            rgb = np.array(Image.open(rgb_path))
            # remove alpha channel
            rgb = rgb[..., :3]
            all_rgbs.append(rgb)
        if config["load_depths"]:
            # load all depths (.tiff)
            all_depths = []
            pbar_depths = tqdm(depths_paths, desc="depths", ncols=100)
            for depth_path in pbar_depths:
                radial_depth = np.array(Image.open(depth_path))
                print(
                    "radial_depth",
                    radial_depth.shape,
                    radial_depth.dtype,
                    radial_depth.min(),
                    radial_depth.max(),
                )
                z_depth = radial_to_z_depth(radial_depth, fx, fy, cx, cy)
                print(
                    "z_depth",
                    z_depth.shape,
                    z_depth.dtype,
                    z_depth.min(),
                    z_depth.max(),
                )
                all_depths.append(z_depth)
        if config["load_masks"] or config["load_semantic_masks"]:
            # load all segmentations
            if config["load_masks"]:
                all_masks = []
            if config["load_semantic_masks"]:
                all_segs = []
            pbar_segs = tqdm(segs_paths, desc="segs", ncols=100)
            for seg_path in pbar_segs:
                seg = np.array(Image.open(seg_path))
                if config["load_semantic_masks"]:
                    all_segs.append(seg)
                if config["load_masks"]:
                    # create mask from segmentation
                    mask = np.zeros_like(seg)
                    mask[seg != 0] = 1
                    all_masks.append(mask)

    # cameras objects
    all_cameras = []
    for i, pose in enumerate(all_poses):

        cam_rgb = None
        if all_rgbs is not None:
            cam_rgb = all_rgbs[i][None, ...]  # (1, H, W, 3)

        cam_depth = None
        if all_depths is not None:
            cam_depth = all_depths[i][None, ..., None]  # (1, H, W, 1)

        cam_mask = None
        if all_masks is not None:
            cam_mask = all_masks[i][None, ..., None]  # (1, H, W, 1)

        cam_seg = None
        if all_segs is not None:
            cam_seg = all_segs[i][None, ..., None]  # (1, H, W, 1)

        cam_timestamp = np.array([i / frame_rate], dtype=np.float32)

        camera = Camera(
            intrinsics=intrinsics,
            pose=pose,
            rgbs=cam_rgb,
            depths=cam_depth,
            masks=cam_mask,
            semantic_masks=cam_seg,
            timestamps=cam_timestamp,
            global_transform=global_transform,
            local_transform=local_transform,
            camera_label=str(i),
            width=resolution[0],
            height=resolution[1],
            subsample_factor=int(config["subsample_factor"]),
        )
        all_cameras.append(camera)

    # split cameras into train and test
    train_test_overlap = config["train_test_overlap"]
    test_camera_freq = config["test_camera_freq"]
    cameras_splits = {}
    for split in splits:
        cameras_splits[split] = []
        if split == "train":
            if train_test_overlap:
                # if train_test_overlap, use all cameras for training
                cameras_splits[split] = all_cameras
            # else use only a subset of cameras
            else:
                for i, camera in enumerate(all_cameras):
                    if i % test_camera_freq != 0:
                        cameras_splits[split].append(camera)
        if split == "test":
            # select a test camera every test_camera_freq cameras
            for i, camera in enumerate(all_cameras):
                if i % test_camera_freq == 0:
                    cameras_splits[split].append(camera)

    return {
        "scene_type": config["scene_type"],
        "point_clouds": [],
        "cameras_splits": cameras_splits,
        "global_transform": global_transform,
        "min_camera_distance": min_camera_distance,
        "max_camera_distance": max_camera_distance,
        "scene_radius": scene_radius,
        "fps": frame_rate,
        "nr_per_camera_frames": 1,
        "nr_sequence_frames": len(cameras_splits["train"]),
    }
