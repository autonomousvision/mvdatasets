import torch
import sys
import numpy as np
from pathlib import Path
import random
from typing import List, Union, Tuple
from mvdatasets.utils.printing import print_error
from dataclasses import dataclass


@dataclass
class Args:
    datasets_path: Path = Path("/home/stefano/Data")
    dataset_name: str = "dtu"
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def _set_seeds(self):
        # Set a random seed for reproducibility
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        # Check if CUDA (GPU support) is available
        if "cuda" in self.device:
            if not torch.cuda.is_available():
                print_error("CUDA is not available, change device to 'cpu'")
            else:
                # Set a random seed for GPU
                torch.cuda.manual_seed(self.seed)

    def __post_init__(self):
        # check path and dataset
        if not self.datasets_path.exists():
            print_error(f"Dataset path {self.datasets_path} does not exist.")
        # set seeds
        self._set_seeds()


def get_dataset_test_preset(dataset_name: str = "dtu") -> Tuple[str, List[str], dict]:

    # test dtu
    if dataset_name == "dtu":
        scene_name = "dtu_scan83"
        splits = ["train", "test"]
        pc_paths = [f"tests/assets/meshes/{dataset_name}/{scene_name}.ply"]

    # test blended-mvs
    elif dataset_name == "blended-mvs":
        scene_name = "bmvs_bear"
        splits = ["train", "test"]
        pc_paths = []

    # test blender
    elif dataset_name == "blender":
        scene_name = "lego"
        splits = ["train", "test"]
        pc_paths = ["tests/assets/point_clouds/blender/lego.ply"]

    # test shelly
    elif dataset_name == "shelly":
        scene_name = "khady"
        splits = ["train", "test"]
        pc_paths = [f"tests/assets/meshes/{dataset_name}/{scene_name}.ply"]

    # test blendernerf
    elif dataset_name == "blendernerf":
        scene_name = "plushy"
        splits = ["train", "test"]
        pc_paths = [f"tests/assets/meshes/{dataset_name}/{scene_name}.ply"]

    # test dmsr
    elif dataset_name == "dmsr":
        scene_name = "dinning"
        splits = ["train", "test"]
        pc_paths = [f"tests/assets/meshes/{dataset_name}/{scene_name}.ply"]

    # test refnerf
    elif dataset_name == "refnerf":
        scene_name = "car"
        splits = ["train", "test"]
        pc_paths = []

    # test ingp
    elif dataset_name == "ingp":
        scene_name = "fox"
        splits = ["train", "test"]
        pc_paths = []

    # test llff
    elif dataset_name == "llff":
        scene_name = "fern"
        splits = ["train", "test"]
        pc_paths = ["tests/assets/point_clouds/llff/fern.ply"]

    # test mipnerf360
    elif dataset_name == "mipnerf360":
        scene_name = "garden"
        splits = ["train", "test"]
        pc_paths = []

    # test d-nerf
    elif dataset_name == "d-nerf":
        scene_name = "bouncingballs"
        splits = ["train", "test"]
        pc_paths = []

    # test visor
    elif dataset_name == "visor":
        scene_name = "P01_01"
        splits = ["train", "test"]  # ["train", "val"]
        pc_paths = []

    # test panoptic-sports
    elif dataset_name == "panoptic-sports":
        scene_name = "basketball"
        splits = ["train", "test"]
        pc_paths = []

    else:
        print_error(f"{dataset_name} does not have a test preset.")

    return {"scene_name": scene_name, "splits": splits, "pc_paths": pc_paths}
