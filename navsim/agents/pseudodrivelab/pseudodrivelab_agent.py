from typing import Any, Dict, List, Optional, Union

import pytorch_lightning as pl
import torch
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from navsim.agents.abstract_agent import AbstractAgent
from navsim.agents.pseudodrivelab.pseudodrivelab_callback import PseudodrivelabCallback
from navsim.agents.pseudodrivelab.pseudodrivelab_config import PseudodrivelabConfig
from navsim.agents.pseudodrivelab.pseudodrivelab_features import PseudodrivelabFeatureBuilder, PseudodrivelabTargetBuilder
from navsim.agents.pseudodrivelab.pseudodrivelab_loss import pseudodrivelab_loss
from navsim.agents.pseudodrivelab.pseudodrivelab_model import PseudodrivelabModel
from navsim.common.dataclasses import SensorConfig
from navsim.planning.training.abstract_feature_target_builder import AbstractFeatureBuilder, AbstractTargetBuilder


class PseudodrivelabAgent(AbstractAgent):
    """Agent interface for Pseudolab."""

    def __init__(
        self,
        config: PseudodrivelabConfig,
        lr: float,
        checkpoint_path: Optional[str] = None,
        trajectory_sampling: TrajectorySampling = TrajectorySampling(time_horizon=4, interval_length=0.5),
    ):
        """
        Initializes Pseudolab agent.
        :param config: global config of Pseudolab agent
        :param lr: learning rate during training
        :param checkpoint_path: optional path string to checkpoint, defaults to None
        :param trajectory_sampling: trajectory sampling specification
        """
        super().__init__(trajectory_sampling)

        self._config = config
        self._lr = lr

        self._checkpoint_path = checkpoint_path
        self._pseudolab_model = PseudodrivelabModel(self._trajectory_sampling, config)

    def name(self) -> str:
        """Inherited, see superclass."""
        return self.__class__.__name__

    def initialize(self) -> None:
        """Inherited, see superclass."""
        if torch.cuda.is_available():
            state_dict: Dict[str, Any] = torch.load(self._checkpoint_path)["state_dict"]
        else:
            state_dict: Dict[str, Any] = torch.load(self._checkpoint_path, map_location=torch.device("cpu"))[
                "state_dict"
            ]
        self.load_state_dict({k.replace("agent.", ""): v for k, v in state_dict.items()})

    def get_sensor_config(self) -> SensorConfig:
        """Inherited, see superclass."""
        # EDITTED: Pseudolab by Hyeongkyun Kim
        # - Extended input version : Input camera 3ea to 7ea
        # TODO: Pseudolab by Hyeongkyun Kim
        # - Advanced version : Consider temporal info (t-1)
        history_steps = [3]
        return SensorConfig(
            cam_f0=history_steps,
            cam_l0=history_steps,
            cam_l1=history_steps,
            cam_l2=history_steps,
            cam_r0=history_steps,
            cam_r1=history_steps,
            cam_r2=history_steps,
            cam_b0=False,
            lidar_pc=history_steps if not self._config.latent else False,
        )

    def get_target_builders(self) -> List[AbstractTargetBuilder]:
        """Inherited, see superclass."""
        return [PseudodrivelabTargetBuilder(trajectory_sampling=self._trajectory_sampling, config=self._config)]

    def get_feature_builders(self) -> List[AbstractFeatureBuilder]:
        """Inherited, see superclass."""
        return [PseudodrivelabFeatureBuilder(config=self._config)]

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Inherited, see superclass."""
        return self._pseudolab_model(features)

    def compute_loss(
        self,
        features: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        predictions: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Inherited, see superclass."""
        return pseudodrivelab_loss(targets, predictions, self._config)

    def get_optimizers(
        self,
    ) -> Union[Optimizer, Dict[str, Union[Optimizer, LRScheduler]]]:
        """Inherited, see superclass."""
        return torch.optim.Adam(self._pseudolab_model.parameters(), lr=self._lr)

    def get_training_callbacks(self) -> List[pl.Callback]:
        """Inherited, see superclass."""
        return [PseudodrivelabCallback(self._config)]
