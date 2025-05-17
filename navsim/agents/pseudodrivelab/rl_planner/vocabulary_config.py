from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.common.maps.abstract_map import SemanticMapLayer


@dataclass
class VocabularyConfig:
    """Global Vocabulary config."""

    # Trajectory parameters
    trajectory_path: str = "./navsim/agents/pseudodrivelab/trajectory_vocabulary/clustering_results/cluster_centers_20.npy"


    # LiDAR parameters
    lidar_min_x: float = -32
    lidar_max_x: float = 32
    lidar_min_y: float = -32
    lidar_max_y: float = 32
    max_height_lidar: float = 100.0
    pixels_per_meter: float = 4.0
    hist_max_per_pixel: int = 5
    lidar_split_height: float = 0.2
    use_ground_plane: bool = False
    lidar_seq_len: int = 1
    lidar_resolution_width: int = 256
    lidar_resolution_height: int = 256

    # BEV semantic map parameters
    bev_semantic_classes = {
        1: ("polygon", [SemanticMapLayer.LANE, SemanticMapLayer.INTERSECTION]),  # road
        2: ("polygon", [SemanticMapLayer.WALKWAYS]),  # walkways
        3: ("linestring", [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR]),  # centerline
        4: ("box", [
            TrackedObjectType.CZONE_SIGN,
            TrackedObjectType.BARRIER,
            TrackedObjectType.TRAFFIC_CONE,
            TrackedObjectType.GENERIC_OBJECT,
        ]),  # static_objects
        5: ("box", [TrackedObjectType.VEHICLE]),  # vehicles
        6: ("box", [TrackedObjectType.PEDESTRIAN]),  # pedestrians
    }
    bev_pixel_width: int = lidar_resolution_width
    bev_pixel_height: int = lidar_resolution_height
    bev_pixel_size: float = 0.25
    num_bev_classes: int = 7
    bev_features_channels: int = 64
    bev_down_sample_factor: int = 4
    bev_upsample_factor: int = 2

    # Bounding box parameters
    use_bounding_boxes: bool = True
    num_bounding_boxes: int = 30

    # Feature parameters
    latent: bool = True  # Set to True to disable LiDAR
    latent_rad_thresh: float = 4 * np.pi / 9

    # Camera parameters
    camera_width: int = 1024
    camera_height: int = 256

    # Model parameters
    hidden_size: int = 256
    num_layers: int = 3
    num_heads: int = 8
    dropout: float = 0.1

    # Loss weights
    trajectory_weight: float = 10.0
    bev_semantic_weight: float = 10.0
    bounding_box_weight: float = 1.0

    @property
    def bev_semantic_frame(self) -> Tuple[int, int]:
        return (self.bev_pixel_height, self.bev_pixel_width)

    @property
    def bev_radius(self) -> float:
        values = [
            self.lidar_min_x,
            self.lidar_max_x,
            self.lidar_min_y,
            self.lidar_max_y,
        ]
        return max([abs(value) for value in values]) 