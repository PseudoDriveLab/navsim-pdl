from typing import Any, Dict, Optional, Union

import cv2
import numpy as np
import numpy.typing as npt
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchvision.utils as vutils
from nuplan.common.actor_state.oriented_box import OrientedBox
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.maps.abstract_map import SemanticMapLayer
from PIL import ImageColor

from navsim.agents.transfuser.transfuser_config import TransfuserConfig
from navsim.agents.transfuser.transfuser_features import BoundingBox2DIndex
from navsim.visualization.config import AGENT_CONFIG, MAP_LAYER_CONFIG


class TransfuserCallback(pl.Callback):
    """Visualization Callback for TransFuser during training."""

    def __init__(
        self,
        config: TransfuserConfig,
        num_plots: int = 3,
        num_rows: int = 2,
        num_columns: int = 2,
    ) -> None:
        """
        Initializes the visualization callback.
        :param config: global config dataclass of TransFuser
        :param num_plots: number of images tiles, defaults to 3
        :param num_rows: number of rows in image tile, defaults to 2
        :param num_columns: number of columns in image tile, defaults to 2
        """

        self._config = config

        self._num_plots = num_plots
        self._num_rows = num_rows
        self._num_columns = num_columns

    def on_validation_epoch_start(self, trainer: pl.Trainer, lightning_module: pl.LightningModule) -> None:
        """Inherited, see superclass."""

    def on_validation_epoch_end(self, trainer: pl.Trainer, lightning_module: pl.LightningModule) -> None:
        """Inherited, see superclass."""
        device = lightning_module.device
        for idx_plot in range(self._num_plots):
            features, targets = next(iter(trainer.val_dataloaders))
            features, targets = dict_to_device(features, device), dict_to_device(targets, device)
            with torch.no_grad():
                predictions = lightning_module.agent.forward(features)

            features, targets, predictions = (
                dict_to_device(features, "cpu"),
                dict_to_device(targets, "cpu"),
                dict_to_device(predictions, "cpu"),
            )
            grid = self._visualize_model(features, targets, predictions)
            trainer.logger.experiment.add_image(f"val_plot_{idx_plot}", grid, global_step=trainer.current_epoch)

    def on_test_epoch_start(self, trainer: pl.Trainer, lightning_module: pl.LightningModule) -> None:
        """Inherited, see superclass."""

    def on_test_epoch_end(self, trainer: pl.Trainer, lightning_module: pl.LightningModule) -> None:
        """Inherited, see superclass."""

    def on_train_epoch_start(self, trainer: pl.Trainer, lightning_module: pl.LightningModule) -> None:
        """Inherited, see superclass."""

    def on_train_epoch_end(
        self, trainer: pl.Trainer, lightning_module: pl.LightningModule, unused: Optional[Any] = None
    ) -> None:
        """Inherited, see superclass."""

        device = lightning_module.device
        for idx_plot in range(self._num_plots):
            features, targets = next(iter(trainer.train_dataloader))
            features, targets = dict_to_device(features, device), dict_to_device(targets, device)
            with torch.no_grad():
                predictions = lightning_module.agent.forward(features)

            features, targets, predictions = (
                dict_to_device(features, "cpu"),
                dict_to_device(targets, "cpu"),
                dict_to_device(predictions, "cpu"),
            )
            grid = self._visualize_model(features, targets, predictions)
            trainer.logger.experiment.add_image(f"train_plot_{idx_plot}", grid, global_step=trainer.current_epoch)

    

    def _visualize_model(self, features, targets, predictions):
        cam = features["camera_feature"]                          # float32 [0..1]
        B, N, C, H, W = cam.shape
        cam = cam.permute(0,1,3,4,2).reshape(B, N*H, W, C)        # (B, 4H, W, 3)
        cam = cam.permute(0,3,1,2).contiguous()                   # (B,3,4H,W)
        cam = F.interpolate(cam, size=(256,1024), mode="bilinear", align_corners=False)
        cam_uint8 = (cam.clamp(0,1)*255).byte().permute(0,2,3,1).cpu().numpy()  # (B,256,1024,3)

        # BEV GT / Pred (원래 128×256)
        bev_gt   = targets["bev_semantic_map"].cpu().numpy()      # (B,128,256)
        bev_pred = predictions["bev_semantic_map"].argmax(1).cpu().numpy()  # (B,128,256)

        # LiDAR + Box
        lidar_map      = features["lidar_feature"].squeeze(1).cpu().numpy()  # (B,256,256)
        agent_labels   = targets["agent_labels"].cpu().numpy()
        agent_states   = targets["agent_states"].cpu().numpy()
        trajectory_gt  = targets["trajectory"].cpu().numpy()
        agent_p_labels = predictions["agent_labels"].sigmoid().cpu().numpy()
        agent_p_states = predictions["agent_states"].cpu().numpy()
        trajectory_p   = predictions["trajectory"].cpu().numpy()

        tiles = []
        for i in range(self._num_rows * self._num_columns):
            # 전체 타일 크기: 높이 256+256=512, 너비 1024+512=1536
            tile = np.zeros((512, 1536, 3), dtype=np.uint8)

            # ── ① 왼쪽 상단: 카메라 원본 ───────────────────────────
            tile[0:256, 0:1024] = cam_uint8[i]

            # ── ② 왼쪽 하단: BEV GT (128→256↑, 256→512↑) ───────────
            gt_rgb = semantic_map_to_rgb(bev_gt[i], self._config)    # (128,256,3)
            gt_rgb = cv2.resize(gt_rgb, (512,256), interpolation=cv2.INTER_NEAREST)
            tile[256:512, 0:512] = gt_rgb

            # ── ③ 중앙 하단: BEV Pred (128→256↑, 256→512↑) ──────────
            pred_rgb = semantic_map_to_rgb(bev_pred[i], self._config)
            pred_rgb = cv2.resize(pred_rgb, (512,256), interpolation=cv2.INTER_NEAREST)
            tile[256:512, 512:1024] = pred_rgb

            # ── ④ 오른쪽 전체: LiDAR + Box ────────────────────────
            lidar_rgb = lidar_map_to_rgb(
                lidar_map[i],
                agent_states[i][agent_labels[i]],
                agent_p_states[i][agent_p_labels[i] > 0.5],
                trajectory_gt[i],
                trajectory_p[i],
                self._config
            )  # 이 함수가 (256,256,3) 리턴한다고 가정
            # LiDAR 맵도 256→512로 업샘플
            lidar_rgb = cv2.resize(lidar_rgb, (512,512), interpolation=cv2.INTER_NEAREST)
            tile[:, 1024:1536] = lidar_rgb

            tiles.append(torch.from_numpy(tile).permute(2,0,1))

        # (C, H, W)xB → 그리드
        return vutils.make_grid(tiles, nrow=self._num_rows, normalize=False)


def dict_to_device(dict: Dict[str, torch.Tensor], device: Union[torch.device, str]) -> Dict[str, torch.Tensor]:
    """
    Helper function to move tensors from dictionary to device.
    :param dict: dictionary of names and tensors
    :param device: torch device to move tensors to
    :return: dictionary with tensors on specified device
    """
    for key in dict.keys():
        dict[key] = dict[key].to(device)
    return dict


def semantic_map_to_rgb(semantic_map: npt.NDArray[np.int64], config: TransfuserConfig) -> npt.NDArray[np.uint8]:
    """
    Convert semantic map to RGB image.
    :param semantic_map: numpy array of segmentation map (multi-channel)
    :param config: global config dataclass of TransFuser
    :return: RGB image as numpy array
    """

    height, width = semantic_map.shape[:2]
    rgb_map = np.ones((height, width, 3), dtype=np.uint8) * 255

    for label in range(1, config.num_bev_classes):

        if config.bev_semantic_classes[label][0] == "linestring":
            hex_color = MAP_LAYER_CONFIG[SemanticMapLayer.BASELINE_PATHS]["line_color"]
        else:
            layer = config.bev_semantic_classes[label][-1][0]  # take color of first element
            hex_color = (
                AGENT_CONFIG[layer]["fill_color"]
                if layer in AGENT_CONFIG.keys()
                else MAP_LAYER_CONFIG[layer]["fill_color"]
            )

        rgb_map[semantic_map == label] = ImageColor.getcolor(hex_color, "RGB")
    return rgb_map[::-1, ::-1]


def lidar_map_to_rgb(
    lidar_map: npt.NDArray[np.int64],
    agent_states: npt.NDArray[np.float32],
    pred_agent_states: npt.NDArray[np.float32],
    trajectory: npt.NDArray[np.float32],
    pred_trajectory: npt.NDArray[np.float32],
    config: TransfuserConfig,
) -> npt.NDArray[np.uint8]:
    """
    Converts lidar histogram map with predictions and targets to RGB.
    :param lidar_map: lidar histogram raster
    :param agent_states: target agent bounding box states
    :param pred_agent_states: predicted agent bounding box states
    :param trajectory: target trajectory of human operator
    :param pred_trajectory: predicted trajectory of agent
    :param config: global config dataclass of TransFuser
    :return: RGB image for training visualization
    """
    gt_color, pred_color = (0, 255, 0), (255, 0, 0)
    point_size = 4

    height, width = lidar_map.shape[:2]

    def coords_to_pixel(coords):
        """Convert local coordinates to pixel indices."""
        pixel_center = np.array([[height / 2.0, width / 2.0]])
        coords_idcs = (coords / config.bev_pixel_size) + pixel_center
        return coords_idcs.astype(np.int32)

    rgb_map = (lidar_map * 255).astype(np.uint8)
    rgb_map = 255 - rgb_map[..., None].repeat(3, axis=-1)

    for color, agent_state_array in zip([gt_color, pred_color], [agent_states, pred_agent_states]):
        for agent_state in agent_state_array:
            agent_box = OrientedBox(
                StateSE2(*agent_state[BoundingBox2DIndex.STATE_SE2]),
                agent_state[BoundingBox2DIndex.LENGTH],
                agent_state[BoundingBox2DIndex.WIDTH],
                1.0,
            )
            exterior = np.array(agent_box.geometry.exterior.coords).reshape((-1, 1, 2))
            exterior = coords_to_pixel(exterior)
            exterior = np.flip(exterior, axis=-1)
            cv2.polylines(rgb_map, [exterior], isClosed=True, color=color, thickness=2)

    for color, traj in zip([gt_color, pred_color], [trajectory, pred_trajectory]):
        trajectory_indices = coords_to_pixel(traj[:, :2])
        for x, y in trajectory_indices:
            cv2.circle(rgb_map, (y, x), point_size, color, -1)  # -1 fills the circle

    return rgb_map[::-1, ::-1]
