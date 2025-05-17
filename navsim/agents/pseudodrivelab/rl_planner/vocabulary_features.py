from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import multiprocessing as mp
from functools import partial

import os
import cv2
import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as F
import pickle
from hydra.utils import instantiate

from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.common.maps.abstract_map import AbstractMap, MapObject, SemanticMapLayer
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.common.actor_state.oriented_box import OrientedBox
from shapely import affinity
from shapely.geometry import LineString, Polygon
from torchvision import transforms

from navsim.common.dataclasses import Trajectory, AgentInput, Annotations, Scene
from navsim.common.enums import BoundingBoxIndex, LidarIndex
from navsim.planning.scenario_builder.navsim_scenario_utils import tracked_object_types
from navsim.planning.training.abstract_feature_target_builder import AbstractFeatureBuilder, AbstractTargetBuilder
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer, PDMScorerConfig
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from navsim.traffic_agents_policies.abstract_traffic_agents_policy import AbstractTrafficAgentsPolicy
from navsim.traffic_agents_policies.navsim_IDM_traffic_agents import NavsimIDMTrafficAgents
from navsim.evaluate.pdm_score import pdm_score
from navsim.common.dataloader import MetricCacheLoader
from navsim.planning.scenario_builder.navsim_scenario_utils import ego_status_to_ego_state
from navsim.planning.simulation.planner.pdm_planner.observation.pdm_observation import PDMObservation

from navsim.agents.pseudodrivelab.rl_planner.vocabulary_config import VocabularyConfig
from omegaconf import OmegaConf

# config.yaml 또는 dict 형태의 구성 로딩 (예시)
cfg = OmegaConf.load("navsim/planning/script/config/common/agent/traffic_agents_policy/navsim_IDM_traffic_agents.yaml")

class VocabularyFeatureBuilder(AbstractFeatureBuilder):
    """Feature builder for Vocabulary agent."""

    def __init__(self, config: VocabularyConfig):
        """
        Initializes Vocabulary feature builder.
        :param config: Vocabulary configuration
        """
        super().__init__()
        self._config = config
        self._pdm_scores_cache = {}
        self._cache_dir = os.path.join(os.path.dirname(config.trajectory_path), "pdm_scores")
        os.makedirs(self._cache_dir, exist_ok=True)

    def get_unique_name(self) -> str:
        """Inherited, see superclass."""
        return self.__class__.__name__


    def _get_cached_pdm_score(self, scene_token: str, trajectory_idx: int) -> Optional[Dict[str, float]]:
        """Get cached PDM score if exists."""
        cache_file = os.path.join(self._cache_dir, f"pdm_score_{trajectory_idx}.pkl")
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                cache = pickle.load(f)
                return cache.get(scene_token)
        return None

    def _cache_pdm_score(self, scene_token: str, trajectory_idx: int, scores: Dict[str, float]):
        """Cache PDM score."""
        cache_file = os.path.join(self._cache_dir, f"pdm_score_{trajectory_idx}.pkl")
        cache = {}
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                cache = pickle.load(f)
        cache[scene_token] = scores
        with open(cache_file, 'wb') as f:
            pickle.dump(cache, f)

    def compute_features(self, agent_input: AgentInput) -> Dict[str, torch.Tensor]:
        """
        Computes features from agent input.
        :param agent_input: Agent input dataclass
        :return: Dictionary of feature tensors
        """
        features = {}

        # Camera features
        features["camera_feature"] = self._get_camera_feature(agent_input)

        # LiDAR features (optional)
        if not self._config.latent:
            features["lidar_feature"] = self._get_lidar_feature(agent_input)

        # Ego state features
        features["ego_state"] = torch.concatenate(
            [
                torch.tensor(agent_input.ego_statuses[-1].driving_command, dtype=torch.float32),
                torch.tensor(agent_input.ego_statuses[-1].ego_velocity, dtype=torch.float32),
                torch.tensor(agent_input.ego_statuses[-1].ego_acceleration, dtype=torch.float32),
            ],
        )

        return features

    def _get_camera_feature(self, agent_input: AgentInput) -> torch.Tensor:
        """
        Extract stitched camera from AgentInput
        :param agent_input: input dataclass
        :return: stitched front view image as torch tensor
        """

        cameras = agent_input.cameras[-1]

        # Crop to ensure 4:1 aspect ratio
        l0 = cameras.cam_l0.image[28:-28, 416:-416]
        f0 = cameras.cam_f0.image[28:-28]
        r0 = cameras.cam_r0.image[28:-28, 416:-416]

        # stitch l0, f0, r0 images
        stitched_image = np.concatenate([l0, f0, r0], axis=1)
        resized_image = cv2.resize(stitched_image, (1024, 256))
        tensor_image = transforms.ToTensor()(resized_image)

        return tensor_image

    def _get_lidar_feature(self, agent_input: AgentInput) -> torch.Tensor:
        """
        Compute LiDAR feature as 2D histogram, according to Transfuser
        :param agent_input: input dataclass
        :return: LiDAR histogram as torch tensors
        """

        # only consider (x,y,z) & swap axes for (N,3) numpy array
        lidar_pc = agent_input.lidars[-1].lidar_pc[LidarIndex.POSITION].T

        # NOTE: Code from
        # https://github.com/autonomousvision/carla_garage/blob/main/team_code/data.py#L873
        def splat_points(point_cloud):
            # 256 x 256 grid
            xbins = np.linspace(
                self._config.lidar_min_x,
                self._config.lidar_max_x,
                (self._config.lidar_max_x - self._config.lidar_min_x) * int(self._config.pixels_per_meter) + 1,
            )
            ybins = np.linspace(
                self._config.lidar_min_y,
                self._config.lidar_max_y,
                (self._config.lidar_max_y - self._config.lidar_min_y) * int(self._config.pixels_per_meter) + 1,
            )
            hist = np.histogramdd(point_cloud[:, :2], bins=(xbins, ybins))[0]
            hist[hist > self._config.hist_max_per_pixel] = self._config.hist_max_per_pixel
            overhead_splat = hist / self._config.hist_max_per_pixel
            return overhead_splat

        # Remove points above the vehicle
        lidar_pc = lidar_pc[lidar_pc[..., 2] < self._config.max_height_lidar]
        below = lidar_pc[lidar_pc[..., 2] <= self._config.lidar_split_height]
        above = lidar_pc[lidar_pc[..., 2] > self._config.lidar_split_height]
        above_features = splat_points(above)
        if self._config.use_ground_plane:
            below_features = splat_points(below)
            features = np.stack([below_features, above_features], axis=-1)
        else:
            features = np.stack([above_features], axis=-1)
        features = np.transpose(features, (2, 0, 1)).astype(np.float32)

        return torch.tensor(features)

    def _compute_2d_histogram(self, points: torch.Tensor) -> torch.Tensor:
        """
        Computes 2D histogram from point cloud.
        :param points: Point cloud tensor
        :return: 2D histogram tensor
        """
        # Convert points to pixel coordinates
        x = ((points[:, 0] - self._config.lidar_min_x) * self._config.pixels_per_meter).long()
        y = ((points[:, 1] - self._config.lidar_min_y) * self._config.pixels_per_meter).long()

        # Create histogram
        hist = torch.zeros(
            (1, self._config.lidar_resolution_height, self._config.lidar_resolution_width),
            dtype=torch.float32,
            device=points.device,
        )

        # Filter out points outside the range
        mask = (x >= 0) & (x < self._config.lidar_resolution_width) & \
               (y >= 0) & (y < self._config.lidar_resolution_height)
        x = x[mask]
        y = y[mask]

        # Update histogram
        hist[0, y, x] = 1.0

        # Normalize histogram
        hist = torch.clamp(hist, 0.0, self._config.hist_max_per_pixel)

        return hist


class VocabularyTargetBuilder(AbstractTargetBuilder):
    """Target builder for Vocabulary agent."""

    def __init__(self, trajectory_sampling: TrajectorySampling, config: VocabularyConfig):
        """
        Initializes Vocabulary target builder.
        :param trajectory_sampling: Trajectory sampling specification
        :param config: Vocabulary configuration
        """
        super().__init__()
        self._trajectory_sampling = trajectory_sampling
        self._config = config
        self._cache_dir = os.path.join(os.path.dirname(config.trajectory_path), "pdm_scores")
        os.makedirs(self._cache_dir, exist_ok=True)

        self.metric_cache_loader = MetricCacheLoader(Path("exp/metric_cache"))
        self._trajectory_centers = self._load_trajectory_centers(self._config.trajectory_path)
        
        # Initialize PDM scorer and simulator for multiprocessing
        self._scorer_config = PDMScorerConfig()
        self._scorer = PDMScorer(
            proposal_sampling=TrajectorySampling(
                time_horizon=4.0,
                interval_length=0.1
            ),
            config=self._scorer_config
        )
        self._simulator = PDMSimulator(
            proposal_sampling=TrajectorySampling(
                time_horizon=4.0,
                interval_length=0.1
            )
        )
        self._traffic_agents_policy = instantiate(
            cfg.reactive, self._simulator.proposal_sampling
        )

    def _load_trajectory_centers(self, trajectory_path: str) -> np.ndarray:
        """Load trajectory centers from file."""
        if not os.path.exists(trajectory_path):
            raise FileNotFoundError(f"Trajectory file not found: {trajectory_path}")
        return np.load(trajectory_path).reshape(-1, 8, 3)

    def get_unique_name(self) -> str:
        """Inherited, see superclass."""
        return self.__class__.__name__

    def _compute_pdm_score_worker(self, trajectory: np.ndarray, trajectory_idx: int, scene_token: str, initial_token: str) -> Tuple[int, Dict[str, float]]:
        """Worker function for computing PDM score."""
        # Convert numpy array to Trajectory object
        trajectory_sampling = TrajectorySampling(
            time_horizon=4.0,
            interval_length=0.5
        )
        trajectory_obj = Trajectory(
            trajectory_sampling=trajectory_sampling,
            poses=trajectory
        )

        # Score the trajectory using pdm_score function
        score_row, ego_simulated_states = pdm_score(
            metric_cache=self.metric_cache_loader.get_from_token(initial_token),
            model_trajectory=trajectory_obj,
            future_sampling=self._simulator.proposal_sampling,
            simulator=self._simulator,
            scorer=self._scorer,
            traffic_agents_policy=self._traffic_agents_policy
        )

        # Convert score row to dictionary
        scores = {
            'pdm_score': float(score_row['pdm_score'].iloc[0]),
            'no_at_fault_collisions': float(score_row['no_at_fault_collisions'].iloc[0]),
            'drivable_area_compliance': float(score_row['drivable_area_compliance'].iloc[0]),
            'driving_direction_compliance': float(score_row['driving_direction_compliance'].iloc[0]),
            'traffic_light_compliance': float(score_row['traffic_light_compliance'].iloc[0]),
            'ego_progress': float(score_row['ego_progress'].iloc[0]),
            'time_to_collision_within_bound': float(score_row['time_to_collision_within_bound'].iloc[0]),
            'lane_keeping': float(score_row['lane_keeping'].iloc[0]),
            'history_comfort': float(score_row['history_comfort'].iloc[0]),
            'multiplicative_metrics_prod': float(score_row['multiplicative_metrics_prod'].iloc[0]),
            'weighted_metrics': score_row['weighted_metrics'].iloc[0].tolist(),
            'weighted_metrics_array': score_row['weighted_metrics_array'].iloc[0].tolist()
        }

        return trajectory_idx, scores

    def _is_trajectory_reachable(self, trajectory: np.ndarray, ego_state: Dict[str, float]) -> bool:
        """
        Check if trajectory is reachable from current ego state.
        :param trajectory: Trajectory array
        :param ego_state: Dictionary containing ego state (position, velocity, heading)
        :return: Boolean indicating if trajectory is reachable
        """
        # Get trajectory start point
        traj_start = trajectory[0, :2]  # x, y
        traj_heading = np.arctan2(trajectory[1, 1] - trajectory[0, 1], 
                                trajectory[1, 0] - trajectory[0, 0])
        
        # Get ego state
        ego_velocity = ego_state['velocity']
        ego_pos = np.array([ego_state['x']+0.5*ego_velocity, ego_state['y']])
        ego_heading = ego_state['heading']
        
        # Calculate distance and heading difference
        distance = np.linalg.norm(traj_start - ego_pos)
        heading_diff = np.abs(np.arctan2(np.sin(traj_heading - ego_heading), 
                                       np.cos(traj_heading - ego_heading)))
        
        # Define reachability thresholds
        max_distance = 0.5  # meters
        max_heading_diff = np.pi / 4  # 45 degrees
        
        # Check if trajectory is reachable
        is_reachable = (
            distance <= max_distance and
            heading_diff <= max_heading_diff
        )
        
        return is_reachable

    def _is_trajectory_in_drivable_area(self, trajectory: np.ndarray, semantic_map: torch.Tensor) -> bool:
        """
        Check if trajectory is in drivable area using BEV semantic map.
        :param trajectory: Trajectory array
        :param semantic_map: BEV semantic map tensor
        :return: Boolean indicating if trajectory is in drivable area
        """
        # Convert trajectory points to pixel coordinates
        x = torch.tensor((trajectory[:, 1] - self._config.lidar_min_x) * self._config.pixels_per_meter).long()
        y = torch.tensor((-trajectory[:, 0] - self._config.lidar_min_y) * self._config.pixels_per_meter).long()
        
        # Filter out points outside the range
        mask = (x >= 0) & (x < self._config.bev_pixel_width) & \
               (y >= 0) & (y < self._config.bev_pixel_height)
        
        if not mask.all():
            return False
            
        # Check if all points are in drivable area (class_id 0)
        for x_coord, y_coord in zip(x[mask], y[mask]):
            if semantic_map[y_coord, x_coord] == 0:  # 0 is drivable area
                return False
        return True

    def compute_targets(self, scene: Scene) -> Dict[str, torch.Tensor]:
        """
        Computes targets from scene using multiprocessing.
        :param scene: Scene object
        :return: Dictionary of target tensors
        """
        targets = {}
        scene_token = scene.scene_metadata.scene_token
        initial_token = scene.scene_metadata.initial_token

        # Get ego state
        ego_state = {
            'x': 0.,
            'y': 0.,
            'heading': 0.,
            'velocity': np.linalg.norm([scene.frames[-1].ego_status.ego_velocity[0],
                                      scene.frames[-1].ego_status.ego_velocity[1]])
        }

        # Compute BEV semantic map
        frame_idx = scene.scene_metadata.num_history_frames - 1
        annotations = scene.frames[frame_idx].annotations
        ego_pose = StateSE2(*scene.frames[frame_idx].ego_status.ego_pose)

        semantic_map = self._compute_bev_semantic_map(annotations, scene.map_api, ego_pose)
        
        # Filter reachable and drivable trajectories
        reachable_indices = []
        reachable_trajectories = []
        cnt_reachable = 0
        cnt_drivable = 0
        for i, trajectory in enumerate(self._trajectory_centers):
            cnt_reachable += self._is_trajectory_reachable(trajectory, ego_state)
            cnt_drivable += self._is_trajectory_in_drivable_area(trajectory, semantic_map)
            if (self._is_trajectory_reachable(trajectory, ego_state) and 
                self._is_trajectory_in_drivable_area(trajectory, semantic_map)):
                reachable_indices.append(i)
                reachable_trajectories.append(trajectory)

        if not reachable_trajectories:
            # If no reachable trajectories, use all trajectories
            reachable_indices = list(range(len(self._trajectory_centers)))
            reachable_trajectories = self._trajectory_centers

        print(cnt_reachable, cnt_drivable, len(reachable_trajectories))
        # Create a pool of workers
        num_workers = min(mp.cpu_count(), len(reachable_trajectories))
        with mp.Pool(processes=num_workers) as pool:
            # Create partial function with fixed arguments
            compute_score = partial(
                self._compute_pdm_score_worker,
                scene_token=scene_token,
                initial_token=initial_token
            )
            
            # Compute scores in parallel
            results = []
            for i, trajectory in enumerate(reachable_trajectories):
                results.append(pool.apply_async(compute_score, (trajectory, reachable_indices[i])))
            
            # Collect results
            pdm_scores = []
            pdm_components = []
            for result in results:
                trajectory_idx, scores = result.get()
                pdm_scores.append(scores['pdm_score'])
                pdm_components.append(scores)

        # Create full trajectory vocabulary tensor with reachability mask
        trajectory_vocabulary = torch.tensor(self._trajectory_centers, dtype=torch.float32)
        reachability_mask = torch.zeros(len(self._trajectory_centers), dtype=torch.bool)
        reachability_mask[reachable_indices] = True

        targets['bev_semantic_map'] = semantic_map
        # targets['trajectory_vocabulary'] = trajectory_vocabulary
        targets['reachability_mask'] = reachability_mask
        targets['pdm_scores'] = torch.tensor(pdm_scores, dtype=torch.float32)
        targets['pdm_components'] = pdm_components

        # import matplotlib.pyplot as plt
        # import matplotlib.cm as cm
        # import matplotlib.colors as colors

        # fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        # extent = [-32, 32, -32, 32]  # in meters

        # axes[0].imshow(semantic_map, origin='upper', extent=extent)
        # axes[0].set_xlabel("Right [m]")
        # axes[0].set_ylabel("Forward [m]")
        # axes[0].set_title("BEV Semantic Map")

        # # (2) Trajectories with PDM score
        # norm = colors.Normalize(vmin=0.0, vmax=1.0)
        # cmap = cm.get_cmap('viridis')
        
        # for t, p in zip(trajectory_vocabulary[reachability_mask], targets['pdm_scores']):
        #     color = cmap(norm(p.item()))
        #     axes[1].plot(t[:, 1], t[:, 0], color=color)

        # axes[1].set_xlim([-32, 32])
        # axes[1].set_ylim([-32, 32])
        # axes[1].set_xlabel("Right [m]")
        # axes[1].set_ylabel("Forward [m]")
        # axes[1].set_title("Trajectory Vocabulary")

        # # Colorbar for PDM score
        # sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        # sm.set_array([])
        # fig.colorbar(sm, ax=axes[1], label='PDM Score')

        # plt.tight_layout()
        # plt.show()
        return targets

    def _compute_agent_targets(
        self, scene: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes agent targets from scene.
        :param scene: Dictionary of scene tensors
        :return: Tuple of agent states and labels tensors
        """
        # Get agent states and labels
        agent_states = scene["agent_states"]
        agent_labels = scene["agent_labels"]

        # Filter agents based on proximity
        ego_position = scene["ego_position"]
        agent_positions = agent_states[:, :2]
        distances = torch.norm(agent_positions - ego_position, dim=1)
        mask = distances < self._config.bev_radius

        # Filter agents based on number of bounding boxes
        if mask.sum() > self._config.num_bounding_boxes:
            # Sort by distance and keep closest agents
            _, indices = torch.sort(distances[mask])
            mask = torch.zeros_like(mask)
            mask[indices[:self._config.num_bounding_boxes]] = True

        # Apply mask
        agent_states = agent_states[mask]
        agent_labels = agent_labels[mask]

        # Pad to fixed size
        if len(agent_states) < self._config.num_bounding_boxes:
            padding = torch.zeros(
                (self._config.num_bounding_boxes - len(agent_states), agent_states.shape[1]),
                dtype=agent_states.dtype,
                device=agent_states.device,
            )
            agent_states = torch.cat([agent_states, padding], dim=0)
            agent_labels = torch.cat([agent_labels, torch.zeros_like(padding[:, 0])], dim=0)

        return agent_states, agent_labels

    def _compute_bev_semantic_map(
        self, annotations: Annotations, map_api: AbstractMap, ego_pose: StateSE2
    ) -> torch.Tensor:
        """
        Creates sematic map in BEV
        :param annotations: annotation dataclass
        :param map_api: map interface of nuPlan
        :param ego_pose: ego pose in global frame
        :return: 2D torch tensor of semantic labels
        """

        bev_semantic_map = np.zeros(self._config.bev_semantic_frame, dtype=np.int64)
        for label, (entity_type, layers) in self._config.bev_semantic_classes.items():
            if entity_type == "polygon":
                entity_mask = self._compute_map_polygon_mask(map_api, ego_pose, layers)
            elif entity_type == "linestring":
                entity_mask = self._compute_map_linestring_mask(map_api, ego_pose, layers)
            else:
                entity_mask = self._compute_box_mask(annotations, layers)
            bev_semantic_map[entity_mask] = label

        return torch.Tensor(bev_semantic_map)

    def _compute_map_polygon_mask(
        self, map_api: AbstractMap, ego_pose: StateSE2, layers: List[SemanticMapLayer]
    ) -> npt.NDArray[np.bool_]:
        """
        Compute binary mask given a map layer class
        :param map_api: map interface of nuPlan
        :param ego_pose: ego pose in global frame
        :param layers: map layers
        :return: binary mask as numpy array
        """

        map_object_dict = map_api.get_proximal_map_objects(
            point=ego_pose.point, radius=self._config.bev_radius, layers=layers
        )
        map_polygon_mask = np.zeros(self._config.bev_semantic_frame[::-1], dtype=np.uint8)
        for layer in layers:
            for map_object in map_object_dict[layer]:
                polygon: Polygon = self._geometry_local_coords(map_object.polygon, ego_pose)
                exterior = np.array(polygon.exterior.coords).reshape((-1, 1, 2))
                exterior = self._coords_to_pixel(exterior)
                cv2.fillPoly(map_polygon_mask, [exterior], color=255)
        # OpenCV has origin on top-left corner
        map_polygon_mask = np.rot90(map_polygon_mask)[::-1]
        return map_polygon_mask > 0

    def _compute_map_linestring_mask(
        self, map_api: AbstractMap, ego_pose: StateSE2, layers: List[SemanticMapLayer]
    ) -> npt.NDArray[np.bool_]:
        """
        Compute binary of linestring given a map layer class
        :param map_api: map interface of nuPlan
        :param ego_pose: ego pose in global frame
        :param layers: map layers
        :return: binary mask as numpy array
        """
        map_object_dict = map_api.get_proximal_map_objects(
            point=ego_pose.point, radius=self._config.bev_radius, layers=layers
        )
        map_linestring_mask = np.zeros(self._config.bev_semantic_frame[::-1], dtype=np.uint8)
        for layer in layers:
            for map_object in map_object_dict[layer]:
                linestring: LineString = self._geometry_local_coords(map_object.baseline_path.linestring, ego_pose)
                points = np.array(linestring.coords).reshape((-1, 1, 2))
                points = self._coords_to_pixel(points)
                cv2.polylines(
                    map_linestring_mask,
                    [points],
                    isClosed=False,
                    color=255,
                    thickness=2,
                )
        # OpenCV has origin on top-left corner
        map_linestring_mask = np.rot90(map_linestring_mask)[::-1]
        return map_linestring_mask > 0

    def _compute_box_mask(self, annotations: Annotations, layers: TrackedObjectType) -> npt.NDArray[np.bool_]:
        """
        Compute binary of bounding boxes in BEV space
        :param annotations: annotation dataclass
        :param layers: bounding box labels to include
        :return: binary mask as numpy array
        """
        box_polygon_mask = np.zeros(self._config.bev_semantic_frame[::-1], dtype=np.uint8)
        for name_value, box_value in zip(annotations.names, annotations.boxes):
            agent_type = tracked_object_types[name_value]
            if agent_type in layers:
                # box_value = (x, y, z, length, width, height, yaw) TODO: add intenum
                x, y, heading = box_value[0], box_value[1], box_value[-1]
                box_length, box_width, box_height = (
                    box_value[3],
                    box_value[4],
                    box_value[5],
                )
                agent_box = OrientedBox(StateSE2(x, y, heading), box_length, box_width, box_height)
                exterior = np.array(agent_box.geometry.exterior.coords).reshape((-1, 1, 2))
                exterior = self._coords_to_pixel(exterior)
                cv2.fillPoly(box_polygon_mask, [exterior], color=255)
        # OpenCV has origin on top-left corner
        box_polygon_mask = np.rot90(box_polygon_mask)[::-1]
        return box_polygon_mask > 0


    @staticmethod
    def _query_map_objects(
        self, map_api: AbstractMap, ego_pose: StateSE2, layers: List[SemanticMapLayer]
    ) -> List[MapObject]:
        """
        Queries map objects
        :param map_api: map interface of nuPlan
        :param ego_pose: ego pose in global frame
        :param layers: map layers
        :return: list of map objects
        """

        # query map api with interesting layers
        map_object_dict = map_api.get_proximal_map_objects(point=ego_pose.point, radius=self, layers=layers)
        map_objects: List[MapObject] = []
        for layer in layers:
            map_objects += map_object_dict[layer]
        return map_objects

    @staticmethod
    def _geometry_local_coords(geometry: Any, origin: StateSE2) -> Any:
        """
        Transform shapely geometry in local coordinates of origin.
        :param geometry: shapely geometry
        :param origin: pose dataclass
        :return: shapely geometry
        """

        a = np.cos(origin.heading)
        b = np.sin(origin.heading)
        d = -np.sin(origin.heading)
        e = np.cos(origin.heading)
        xoff = -origin.x
        yoff = -origin.y

        translated_geometry = affinity.affine_transform(geometry, [1, 0, 0, 1, xoff, yoff])
        rotated_geometry = affinity.affine_transform(translated_geometry, [a, b, d, e, 0, 0])

        return rotated_geometry

    def _coords_to_pixel(self, coords):
        """
        Transform local coordinates in pixel indices of BEV map
        :param coords: _description_
        :return: _description_
        """

        # NOTE: remove half in backward direction
        pixel_center = np.array([[self._config.bev_pixel_height / 2.0, self._config.bev_pixel_width / 2.0]])
        coords_idcs = (coords / self._config.bev_pixel_size) + pixel_center

        return coords_idcs.astype(np.int32)


    def _compute_pdm_score(self, trajectory: np.ndarray, scene: Scene) -> Dict[str, float]:
        """
        Compute PDM score for a trajectory using navsim's PDM scorer (sequential version).
        Returns a dictionary containing the final score and its components.
        """
        # Initialize PDM scorer with default config
        scorer_config = PDMScorerConfig()
        scorer = PDMScorer(
            proposal_sampling=TrajectorySampling(
                time_horizon=4.0,
                interval_length=0.1
            ),
            config=scorer_config
        )

        # Initialize simulator
        simulator = PDMSimulator(
            proposal_sampling=TrajectorySampling(
                time_horizon=4.0,
                interval_length=0.1
            )
        )

        # Initialize traffic agents policy with default config
        traffic_agents_policy: NavsimIDMTrafficAgents = instantiate(
            cfg.reactive, simulator.proposal_sampling
        )

        # Convert numpy array to Trajectory object
        trajectory_sampling = TrajectorySampling(
            time_horizon=4.0,
            interval_length=0.5
        )
        trajectory_obj = Trajectory(
            trajectory_sampling=trajectory_sampling,
            poses=trajectory
        )

        # Score the trajectory using pdm_score function
        score_row, ego_simulated_states = pdm_score(
            metric_cache=self.metric_cache_loader.get_from_token(scene.scene_metadata.initial_token),
            model_trajectory=trajectory_obj,
            future_sampling=simulator.proposal_sampling,
            simulator=simulator,
            scorer=scorer,
            traffic_agents_policy=traffic_agents_policy
        )

        # Convert score row to dictionary
        scores = {
            'pdm_score': float(score_row['pdm_score'].iloc[0]),
            'no_at_fault_collisions': float(score_row['no_at_fault_collisions'].iloc[0]),
            'drivable_area_compliance': float(score_row['drivable_area_compliance'].iloc[0]),
            'driving_direction_compliance': float(score_row['driving_direction_compliance'].iloc[0]),
            'traffic_light_compliance': float(score_row['traffic_light_compliance'].iloc[0]),
            'ego_progress': float(score_row['ego_progress'].iloc[0]),
            'time_to_collision_within_bound': float(score_row['time_to_collision_within_bound'].iloc[0]),
            'lane_keeping': float(score_row['lane_keeping'].iloc[0]),
            'history_comfort': float(score_row['history_comfort'].iloc[0]),
            'multiplicative_metrics_prod': float(score_row['multiplicative_metrics_prod'].iloc[0]),
            'weighted_metrics': score_row['weighted_metrics'].iloc[0].tolist(),
            'weighted_metrics_array': score_row['weighted_metrics_array'].iloc[0].tolist()
        }

        return scores

    def compute_targets_sequential(self, scene: Scene) -> Dict[str, torch.Tensor]:
        """
        Computes targets from scene sequentially (without multiprocessing).
        :param scene: Scene object
        :return: Dictionary of target tensors
        """
        targets = {}
        scene_token = scene.scene_metadata.scene_token
        
        # Compute PDM scores for all trajectories
        pdm_scores = []
        pdm_components = []
        for i, trajectory in enumerate(self._trajectory_centers):
            # Compute and cache new scores
            scores = self._compute_pdm_score_sequential(trajectory, scene)
            pdm_scores.append(scores['pdm_score'])
            pdm_components.append(scores)
            
        targets['trajectory_vocabulary'] = torch.tensor(self._trajectory_centers, dtype=torch.float32)
        targets['pdm_scores'] = torch.tensor(pdm_scores, dtype=torch.float32)
        targets['pdm_components'] = pdm_components

        return targets 