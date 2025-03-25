from pyquaternion import Quaternion
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from mvdatasets.geometry.quaternions import quats_to_rots


if __name__ == "__main__":

    scene_path = Path("/home/stefano/Data/kubric/dynamic")

    # open metadata.json
    metadata_path = scene_path / "metadata.json"
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    # all_quaternions.append(Quaternion(quaternion))  # [w, x, y, z]
    # pose[:3, :3] = all_quaternions[i].rotation_matrix

    # load absolute objects poses from dataset
    instances = metadata["instances"]
    instances_poses = {}
    for idx, instance_metadata in enumerate(instances):

        all_positions = []
        all_quaternions = []
        for position in instance_metadata["positions"]:
            all_positions.append(np.array(position))
        for quaternion in instance_metadata["quaternions"]:
            # quat is in [w, x, y, z] format
            # convert to [x, y, z, w] format
            quaternion_np = np.array(
                [quaternion[1], quaternion[2], quaternion[3], quaternion[0]]
            )
            all_quaternions.append(quaternion_np)

        # convert to rigid transformations
        instances_poses[idx] = []

        # all other poses
        for i in range(0, len(all_positions)):
            pose = np.eye(4)
            pose[:3, 3] = all_positions[i]
            cur_rot = quats_to_rots(all_quaternions[i])
            pose[:3, :3] = cur_rot
            instances_poses[idx].append(pose)

    def invertTransform(pose):
        new_pose = np.eye(4)
        new_pose[:3, :3] = pose[:3, :3].T
        new_pose[:3, 3] = -pose[:3, :3].T @ pose[:3, 3]
        return new_pose

    # calculate relative rotations between poses
    instances_pose_relative = {}
    for idx in instances_poses.keys():
        first_pose = np.eye(4)
        # identity rotation
        first_pose[:3, 3] = instances_poses[idx][0][:3, 3]  # copy translation
        instances_pose_relative[idx] = [first_pose]
        # calculate relative rotations and translations
        for i in range(1, len(instances_poses[idx])):
            rot_relative = (
                instances_poses[idx][i - 1][:3, :3].T @ instances_poses[idx][i][:3, :3]
            )
            # transl_relative = instances_poses[idx][i - 1][:3, :3].T @ (instances_poses[idx][i][:3, 3] - instances_poses[idx][i - 1][:3, 3])
            transl_relative = (
                instances_poses[idx][i][:3, 3] - instances_poses[idx][i - 1][:3, 3]
            )
            pose_relative = np.eye(4)
            pose_relative[:3, :3] = rot_relative
            pose_relative[:3, 3] = transl_relative
            # pose_relative = invertTransform(instances_poses[idx][i - 1]) @ instances_poses[idx][i]
            instances_pose_relative[idx].append(pose_relative)

    instances_poses_reframed = {}
    for idx in instances_pose_relative.keys():
        instances_poses_reframed[idx] = [instances_pose_relative[idx][0]]
        # reframe rotation of subsequent poses
        for i in range(1, len(instances_pose_relative[idx])):
            pose_relative_cur = instances_pose_relative[idx][i]
            pose_prec_reframed = instances_poses_reframed[idx][-1]
            pose_cur_reposed = pose_prec_reframed @ pose_relative_cur
            instances_poses_reframed[idx].append(pose_cur_reposed)

    # TODO: remove
    print("valid idxs", list(instances_poses.keys()))
    dyn_idx = 0

    print("instances_poses", instances_poses[dyn_idx][1])
    print("instances_pose_relative", instances_pose_relative[dyn_idx][1])
    print("instances_poses_reframed", instances_poses_reframed[dyn_idx][1])

    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(121, projection="3d")
    sequence_lenght = 30
    linewidth = 1.0
    # sample colors
    colors = plt.cm.jet(np.linspace(0, 1, sequence_lenght))[:, :3]

    all_centers = []
    for i in range(len(instances_poses[dyn_idx])):
        pose = instances_poses[dyn_idx][i]
        all_centers.append(pose[:3, 3])

    for i in range(sequence_lenght - 1):

        # center at timestamp i
        center_i = all_centers[i]  # (3,)
        # print("center_i", center_i)
        # center at timestamp i+1
        center_i1 = all_centers[i + 1]  # (3,)
        # get color
        color = colors[i]
        # alpha
        alpha = 1.0
        # plot line segment
        ax.plot3D(
            [center_i[0], center_i1[0]],
            [center_i[1], center_i1[1]],
            [center_i[2], center_i1[2]],
            color=color,
            alpha=alpha,
            linewidth=linewidth,
        )

    scale = 0.1
    up = "z"
    for i in range(sequence_lenght):

        label = f"{i}"
        pose = instances_poses[dyn_idx][i]

        # get axis directions (normalized)
        x_dir = pose[:3, 0]
        x_dir /= np.linalg.norm(x_dir)
        y_dir = pose[:3, 1]
        y_dir /= np.linalg.norm(y_dir)
        z_dir = pose[:3, 2]
        z_dir /= np.linalg.norm(z_dir)
        # frame center
        pos = pose[:3, 3]

        # draw camera frame
        ax.quiver(
            pos[0],  # x
            pos[1] if up == "z" else pos[2],  # y
            pos[2] if up == "z" else pos[1],  # z
            x_dir[0],
            x_dir[1] if up == "z" else x_dir[2],
            x_dir[2] if up == "z" else x_dir[1],
            length=scale,
            color="r",
        )
        ax.quiver(
            pos[0],  # x
            pos[1] if up == "z" else pos[2],  # y
            pos[2] if up == "z" else pos[1],  # z
            y_dir[0],
            y_dir[1] if up == "z" else y_dir[2],
            y_dir[2] if up == "z" else y_dir[1],
            length=scale,
            color="g",
        )
        ax.quiver(
            pos[0],  # x
            pos[1] if up == "z" else pos[2],  # y
            pos[2] if up == "z" else pos[1],  # z
            z_dir[0],
            z_dir[1] if up == "z" else z_dir[2],
            z_dir[2] if up == "z" else z_dir[1],
            length=scale,
            color="b",
        )
        ax.text(
            pos[0],  # x
            pos[1] if up == "z" else pos[2],  # y
            pos[2] if up == "z" else pos[1],  # z
            label,
        )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    # axis equal
    ax.set_box_aspect([1, 1, 1])
    # ax.set_aspect("equal")
    #
    ax.set_title("Absolute poses")

    ax = fig.add_subplot(122, projection="3d")

    all_centers = []
    for i in range(len(instances_poses_reframed[dyn_idx])):
        pose = instances_poses_reframed[dyn_idx][i]
        # print("pose", pose.shape)
        all_centers.append(pose[:3, 3])

    for i in range(sequence_lenght - 1):

        # center at timestamp i
        center_i = all_centers[i]  # (3,)
        # print("center_i", center_i)
        # center at timestamp i+1
        center_i1 = all_centers[i + 1]  # (3,)
        # get color
        color = colors[i]
        # alpha
        alpha = 1.0
        # plot line segment
        ax.plot3D(
            [center_i[0], center_i1[0]],
            [center_i[1], center_i1[1]],
            [center_i[2], center_i1[2]],
            color=color,
            alpha=alpha,
            linewidth=linewidth,
        )

    scale = 0.1
    up = "z"
    for i in range(sequence_lenght):

        label = f"{i}"
        pose = instances_poses_reframed[dyn_idx][i]

        # get axis directions (normalized)
        x_dir = pose[:3, 0]
        x_dir /= np.linalg.norm(x_dir)
        y_dir = pose[:3, 1]
        y_dir /= np.linalg.norm(y_dir)
        z_dir = pose[:3, 2]
        z_dir /= np.linalg.norm(z_dir)
        # frame center
        pos = pose[:3, 3]

        # draw camera frame
        ax.quiver(
            pos[0],  # x
            pos[1] if up == "z" else pos[2],  # y
            pos[2] if up == "z" else pos[1],  # z
            x_dir[0],
            x_dir[1] if up == "z" else x_dir[2],
            x_dir[2] if up == "z" else x_dir[1],
            length=scale,
            color="r",
        )
        ax.quiver(
            pos[0],  # x
            pos[1] if up == "z" else pos[2],  # y
            pos[2] if up == "z" else pos[1],  # z
            y_dir[0],
            y_dir[1] if up == "z" else y_dir[2],
            y_dir[2] if up == "z" else y_dir[1],
            length=scale,
            color="g",
        )
        ax.quiver(
            pos[0],  # x
            pos[1] if up == "z" else pos[2],  # y
            pos[2] if up == "z" else pos[1],  # z
            z_dir[0],
            z_dir[1] if up == "z" else z_dir[2],
            z_dir[2] if up == "z" else z_dir[1],
            length=scale,
            color="b",
        )
        ax.text(
            pos[0],  # x
            pos[1] if up == "z" else pos[2],  # y
            pos[2] if up == "z" else pos[1],  # z
            label,
        )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    # axis equal
    # ax.set_aspect("equal")
    ax.set_box_aspect([1, 1, 1])
    ax.set_title("Reframed poses")

    plt.show()
