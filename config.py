import torch
import numpy as np
from pathlib import Path
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
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

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
        if not is_dataset_supported(self.dataset_name):
            print_error(f"{self.dataset_name} is not a supported dataset.")
        # set seeds
        self._set_seeds()


def is_dataset_supported(dataset_name: str) -> bool:
    datasets_supported = [
        "dtu",
        "blended-mvs",
        "blender",
        # "ingp",
        "blendernerf",
        "dmsr",
        "refnerf",
        "llff",
        "mipnerf360",
        "shelly",
        "d-nerf",
        "visor",
        "iphone",
        "panoptic-sports",
        "nerfies",
    ]
    dataset_name = dataset_name.lower()
    if dataset_name in datasets_supported:
        return True
    else:
        return False


def get_dataset_test_preset(dataset_name: str = "dtu") -> Tuple[str, List[str], dict]:

    if not is_dataset_supported(dataset_name):
        print_error(f"{dataset_name} is not a supported dataset.")

    # test dtu
    if dataset_name == "dtu":
        scene_name = "dtu_scan83"
        splits = ["train", "test"]
        pc_paths = [f"tests/assets/meshes/{dataset_name}/{scene_name}.ply"]
        # dataset specific config
        config = {
            "subsample_factor": 1,
        }

    # test blended-mvs
    elif dataset_name == "blended-mvs":
        scene_name = "bmvs_bear"
        splits = ["train", "test"]
        pc_paths = []
        # dataset specific config
        config = {}

    # test blender
    elif dataset_name == "blender":
        scene_name = "lego"
        splits = ["train", "test"]
        pc_paths = ["tests/assets/point_clouds/blender/lego.ply"]
        # dataset specific config
        config = {
            "test_skip": 20,
        }

    # test shelly
    elif dataset_name == "shelly":
        scene_name = "khady"
        splits = ["train", "test"]
        pc_paths = [f"tests/assets/meshes/{dataset_name}/{scene_name}.ply"]
        # dataset specific config
        config = {"test_skip": 4, "init_sphere_radius_mult": 0.2}

    # test blendernerf
    elif dataset_name == "blendernerf":
        scene_name = "plushy"
        splits = ["train", "test"]
        pc_paths = [f"tests/assets/meshes/{dataset_name}/{scene_name}.ply"]
        # dataset specific config
        config = {
            "test_skip": 10,
        }

    # test dmsr
    elif dataset_name == "dmsr":
        scene_name = "dinning"
        splits = ["train", "test"]
        pc_paths = [f"tests/assets/meshes/{dataset_name}/{scene_name}.ply"]
        # dataset specific config
        config = {
            "test_skip": 5,
        }

    # test refnerf
    elif dataset_name == "refnerf":
        scene_name = "car"
        splits = ["train", "test"]
        pc_paths = []
        # dataset specific config
        config = {
            "test_skip": 10,
        }

    # test ingp
    elif dataset_name == "ingp":
        scene_name = "fox"
        splits = ["train", "test"]
        pc_paths = []
        # dataset specific config
        config = {}

    # test llff
    elif dataset_name == "llff":
        scene_name = "fern"
        splits = ["train", "test"]
        pc_paths = ["tests/assets/point_clouds/llff/fern.ply"]
        # dataset specific config
        config = {
            "scene_type": "forward-facing",
        }

    # test mipnerf360
    elif dataset_name == "mipnerf360":
        scene_name = "garden"
        splits = ["train", "test"]
        pc_paths = []

        # dataset specific config
        config = {
            "scene_type": "unbounded",
            "subsample_factor": 8,
        }

        # scene specific config
        if scene_name == "bicycle":
            config["rotate_scene_x_axis_deg"] = -104
            config["translate_scene_z"] = 0.1

        if scene_name == "garden":
            config["rotate_scene_x_axis_deg"] = -120
            config["translate_scene_z"] = 0.2

        if scene_name == "bonsai":
            config["rotate_scene_x_axis_deg"] = -130
            config["translate_scene_z"] = 0.25

        if scene_name == "counter":
            config["rotate_scene_x_axis_deg"] = -125
            config["translate_scene_y"] = -0.1
            config["translate_scene_z"] = 0.25

        if scene_name == "kitchen":
            config["rotate_scene_x_axis_deg"] = -130
            config["translate_scene_z"] = 0.2

        if scene_name == "room":
            config["rotate_scene_x_axis_deg"] = -115

        if scene_name == "stump":
            config["rotate_scene_x_axis_deg"] = -137
            config["translate_scene_z"] = 0.25
            
    # test d-nerf
    elif dataset_name == "d-nerf":
        scene_name = "bouncingballs"
        splits = ["train", "test"]
        pc_paths = []
        # dataset specific config
        config = {}
        
    # test visor
    elif dataset_name == "visor":
        scene_name = "P01_01"
        splits = ["train", "test"]  # ["train", "val"]
        pc_paths = []
        # dataset specific config
        config = {}
        
    else:
        print_error(f"{dataset_name} does not have a test preset.")

    return {
        "scene_name": scene_name,
        "splits": splits,
        "pc_paths": pc_paths,
        "config": config,
    }
