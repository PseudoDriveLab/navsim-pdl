from dataclasses import dataclass
from typing import Tuple
from dataclasses import dataclass, field
from typing import Tuple, List, Dict, Any
import numpy as np
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.common.maps.abstract_map import SemanticMapLayer


@dataclass
class TransfuserConfig:
    """Global TransFuser config."""

    # image_architecture: str = "resnet50"
    image_architecture: str = "resnet18"
    lidar_architecture: str = "resnet34"
    pc_range: List[float] = field(default_factory=lambda: [-32.0, -0.0, -3.0,  32.0, 32.0, 3.0])
    num_points_in_pillar = 4
    latent: bool = False
    latent_rad_thresh: float = 4 * np.pi / 9

    max_height_lidar: float = 100.0
    pixels_per_meter: float = 4
    hist_max_per_pixel: int = 5

    lidar_min_x: float = -32
    lidar_max_x: float = 32
    lidar_min_y: float = -32
    lidar_max_y: float = 32

    lidar_split_height: float = 0.2
    use_ground_plane: bool = False

    # new
    lidar_seq_len: int = 1

    # camera_width: int = 1024
    # camera_height: int = 256
    lidar_resolution_width = 256
    lidar_resolution_height = 256
    bev_grid_h: int = int((lidar_max_y - lidar_min_y) * pixels_per_meter)  # 128
    bev_grid_w: int = int((lidar_max_x - lidar_min_x) * pixels_per_meter) // 2  # 128
    img_vert_anchors: int = 256 // 32
    img_horz_anchors: int = 1024 // 32
    lidar_vert_anchors: int = 256 // 32
    lidar_horz_anchors: int = 256 // 32

    block_exp = 4
    n_layer = 2  # Number of transformer layers used in the vision backbone
    n_head = 4
    n_scale = 4
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    # Mean of the normal distribution initialization for linear layers in the GPT
    gpt_linear_layer_init_mean = 0.0
    # Std of the normal distribution initialization for linear layers in the GPT
    gpt_linear_layer_init_std = 0.02
    # Initial weight of the layer norms in the gpt.
    gpt_layer_norm_init_weight = 1.0

    perspective_downsample_factor = 1
    transformer_decoder_join = True
    detect_boxes = True
    use_bev_semantic = True
    use_semantic = False
    use_depth = False
    add_features = True

    # Transformer

    _dim_: int = 256
    tf_d_model: int = 256
    tf_d_ffn: int = 512
    tf_num_layers: int = 1
    tf_num_head: int = 8
    tf_dropout: float = 0.1

    deformable_attention: Dict[str, Any] = field(default_factory=lambda: {
        'embed_dims': 256,
        # 'embed_dims': 192,
        #needs to be same with _num_mono_levels_
        'num_levels': 3,
        'num_points': 8,
        'im2col_step': 64,
        'dropout': 0.1,
        'batch_first': False
    })

    _num_mono_levels_ = 3

     # deformable attention 내부 config
    num_cams: int = 3
    num_bev_queue: int = 2
    return_intermediate: bool = False

    # detection
    num_bounding_boxes: int = 30
   
    # loss weights
    trajectory_weight: float = 10.0
    agent_class_weight: float = 10.0
    agent_box_weight: float = 1.0
    bev_semantic_weight: float = 50.0

    # BEV mapping
    bev_semantic_classes = {
        1: ("polygon", [SemanticMapLayer.LANE, SemanticMapLayer.INTERSECTION]),  # road
        2: ("polygon", [SemanticMapLayer.WALKWAYS]),  # walkways
        3: (
            "linestring",
            [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR],
        ),  # centerline
        4: (
            "box",
            [
                TrackedObjectType.CZONE_SIGN,
                TrackedObjectType.BARRIER,
                TrackedObjectType.TRAFFIC_CONE,
                TrackedObjectType.GENERIC_OBJECT,
            ],
        ),  # static_objects
        5: ("box", [TrackedObjectType.VEHICLE]),  # vehicles
        6: ("box", [TrackedObjectType.PEDESTRIAN]),  # pedestrians
    }

    bev_pixel_width  = int((32 - (-32)) * pixels_per_meter)
    bev_pixel_height = bev_pixel_width // 2
    bev_pixel_size: float = 1.0 / pixels_per_meter  # 0.5

    num_bev_classes = 7
    bev_features_channels: int = 64
    bev_down_sample_factor: int = 4
    bev_upsample_factor: int = 2
    

    @property
    def bev_semantic_frame(self) -> Tuple[int, int]:
        return (self.bev_grid_h, self.bev_grid_w)

    @property
    def bev_radius(self) -> float:
        values = [
            self.lidar_min_x,
            self.lidar_max_x,
            self.lidar_min_y,
            self.lidar_max_y,
        ]
        return max([abs(value) for value in values])
    
    def get(self, name: str, default=None):
        return getattr(self, name, default)