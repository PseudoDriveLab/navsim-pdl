from typing import Dict, Tuple

import pytorch_lightning as pl
from torch import Tensor
import torch

from navsim.agents.abstract_agent import AbstractAgent


class AgentLightningModule(pl.LightningModule):
    """Pytorch lightning wrapper for learnable agent."""

    def __init__(self, agent: AbstractAgent):
        """
        Initialise the lightning module wrapper.
        :param agent: agent interface in NAVSIM
        """
        super().__init__()
        self.agent = agent
    
        # for full_name, module in self.agent._transfuser_model.named_modules():
        #     if isinstance(module, torch.nn.Linear):
        #         # 2) name을 클로저에 캡쳐
        #         def make_hook(name):
        #             def hook(m, inp, out):
        #                 out_tensors = out if isinstance(out, (tuple, list)) else (out,)
        #                 for t in out_tensors:
        #                     if isinstance(t, torch.Tensor) and not torch.isfinite(t).all():
        #                         # 3) 상세 출력
        #                         print(f"\n=== [NaN DETECTED] in Linear module `{name}` ===")
        #                         # 입력 정보
        #                         for i, x in enumerate(inp):
        #                             if isinstance(x, torch.Tensor):
        #                                 print(f"  input[{i}]: shape={tuple(x.shape)}, "
        #                                       f"min={x.min().item():.3e}, max={x.max().item():.3e}, "
        #                                       f"mean={x.mean().item():.3e}")
        #                         # 파라미터 정보
        #                         w = m.weight
        #                         b = m.bias if m.bias is not None else None
        #                         print(f"  weight: min={w.min().item():.3e}, max={w.max().item():.3e}, "
        #                               f"mean={w.mean().item():.3e}")
        #                         if b is not None:
        #                             print(f"  bias:   min={b.min().item():.3e}, max={b.max().item():.3e}, "
        #                                   f"mean={b.mean().item():.3e}")
        #                         # 출력 정보
        #                         print(f"  output: shape={tuple(t.shape)}, "
        #                               f"min={t.min().item():.3e}, max={t.max().item():.3e}, "
        #                               f"mean={t.mean().item():.3e}")
        #                         # 스택 트레이스
        #                         import traceback; traceback.print_stack(limit=5)
        #                         # 멈추기
        #                         raise RuntimeError(f"NaN detected in forward of Linear `{name}`")
        #             return hook

        #         # 3) hook 등록
        #         module.register_forward_hook(make_hook(full_name))

    def _step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], logging_prefix: str) -> Tensor:
        """
        Propagates the model forward and backwards and computes/logs losses and metrics.
        :param batch: tuple of dictionaries for feature and target tensors (batched)
        :param logging_prefix: prefix where to log step
        :return: scalar loss
        """
        features, targets = batch
        prediction = self.agent.forward(features)
        loss = self.agent.compute_loss(features, targets, prediction)
        self.log(f"{logging_prefix}/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def training_step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], batch_idx: int) -> Tensor:
        """
        Step called on training samples
        :param batch: tuple of dictionaries for feature and target tensors (batched)
        :param batch_idx: index of batch (ignored)
        :return: scalar loss
        """
        return self._step(batch, "train")

    def validation_step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], batch_idx: int):
        """
        Step called on validation samples
        :param batch: tuple of dictionaries for feature and target tensors (batched)
        :param batch_idx: index of batch (ignored)
        :return: scalar loss
        """
        return self._step(batch, "val")

    def configure_optimizers(self):
        """Inherited, see superclass."""
        return self.agent.get_optimizers()
