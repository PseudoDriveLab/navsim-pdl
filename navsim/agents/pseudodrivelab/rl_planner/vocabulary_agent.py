from typing import Any, Dict, List, Optional, Union

import pytorch_lightning as pl
import torch
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import SensorConfig
from navsim.planning.training.abstract_feature_target_builder import AbstractFeatureBuilder, AbstractTargetBuilder
from navsim.agents.pseudodrivelab.rl_planner.vocabulary_config import VocabularyConfig
from navsim.agents.pseudodrivelab.rl_planner.vocabulary_features import VocabularyFeatureBuilder, VocabularyTargetBuilder
import numpy as np
import os


class VocabularyAgent(AbstractAgent):
    """Agent interface for Vocabulary baseline."""

    def __init__(
        self,
        config: VocabularyConfig,
        lr: float,
        checkpoint_path: Optional[str] = None,
        trajectory_sampling: TrajectorySampling = TrajectorySampling(time_horizon=4, interval_length=0.5),
    ):
        """
        Initializes Vocabulary agent.
        :param config: global config of Vocabulary agent
        :param lr: learning rate during training
        :param checkpoint_path: optional path string to checkpoint, defaults to None
        :param trajectory_sampling: trajectory sampling specification
        """
        super().__init__(trajectory_sampling)

        self._config = config
        self._lr = lr
        self._checkpoint_path = checkpoint_path
        
        # Load trajectory centers
        self._trajectory_centers = self._load_trajectory_centers(self._config.trajectory_path)

    def _load_trajectory_centers(self, trajectory_path: str) -> np.ndarray:
        """Load trajectory centers from file."""
        if not os.path.exists(trajectory_path):
            raise FileNotFoundError(f"Trajectory file not found: {trajectory_path}")
        return np.load(trajectory_path).reshape(-1, 8, 3)

    def name(self) -> str:
        """Inherited, see superclass."""
        return self.__class__.__name__

    def initialize(self) -> None:
        """Inherited, see superclass."""
        if self._checkpoint_path:
            if torch.cuda.is_available():
                state_dict: Dict[str, Any] = torch.load(self._checkpoint_path)["state_dict"]
            else:
                state_dict: Dict[str, Any] = torch.load(self._checkpoint_path, map_location=torch.device("cpu"))[
                    "state_dict"
                ]
            self.load_state_dict({k.replace("agent.", ""): v for k, v in state_dict.items()})

    def get_sensor_config(self) -> SensorConfig:
        """Inherited, see superclass."""
        # NOTE: Vocabulary only uses current frame (with index 3 by default)
        history_steps = [3]
        return SensorConfig(
            cam_f0=history_steps,
            cam_l0=history_steps,
            cam_l1=False,
            cam_l2=False,
            cam_r0=history_steps,
            cam_r1=False,
            cam_r2=False,
            cam_b0=False,
            lidar_pc=history_steps if not self._config.latent else False,
        )

    def get_target_builders(self) -> List[AbstractTargetBuilder]:
        """Inherited, see superclass."""
        return [VocabularyTargetBuilder(trajectory_sampling=self._trajectory_sampling, config=self._config)]

    def get_feature_builders(self) -> List[AbstractFeatureBuilder]:
        """Inherited, see superclass."""
        return [VocabularyFeatureBuilder(config=self._config)]

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Inherited, see superclass."""
        raise NotImplementedError

    def compute_loss(
        self,
        features: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        predictions: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Inherited, see superclass."""
        raise NotImplementedError

    def get_optimizers(
        self,
    ) -> Union[Optimizer, Dict[str, Union[Optimizer, LRScheduler]]]:
        """Inherited, see superclass."""
        raise NotImplementedError

    def get_training_callbacks(self) -> List[pl.Callback]:
        """Inherited, see superclass."""
        return [] 