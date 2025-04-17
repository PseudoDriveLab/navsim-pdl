import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Function

class MultiScaleDeformAttnFunction_fp32_torch(nn.Module):
    """
    Pure PyTorch implementation of multi-scale deformable attention as an nn.Module,
    enabling automatic gradient propagation.
    Named to match the original Function-based API.
    """
    def __init__(self, im2col_step: int = 64):
        super().__init__()
        self.im2col_step = im2col_step

    def forward(self,
                value: torch.Tensor,
                spatial_shapes: torch.Tensor,
                level_start_index: torch.Tensor,
                sampling_locations: torch.Tensor,
                attention_weights: torch.Tensor) -> torch.Tensor:
        # value: (B, L, C)
        B, L, C = value.shape
        _, num_query, num_heads, num_levels, num_points, _ = sampling_locations.shape
        head_dim = C // num_heads

        # reshape value for multi-head
        value = value.view(B, L, num_heads, head_dim)
        output = value.new_zeros(B, num_query, num_heads, head_dim)

        for lvl in range(num_levels):
            H, W = spatial_shapes[lvl]
            start = int(level_start_index[lvl])
            end = start + H * W

            # extract per-level features and reshape for grid_sample
            value_l = value[:, start:end, :, :].contiguous().view(B, H, W, num_heads, head_dim)
            # (B, num_heads, head_dim, H, W)
            value_l = value_l.permute(0, 3, 4, 1, 2).contiguous()
            N = B * num_heads
            value_l = value_l.view(N, head_dim, H, W)

            # prepare sampling grid: (B, num_query, num_heads, num_points, 2)
            sampling_loc = sampling_locations[:, :, :, lvl, :, :]
            # (B, num_heads, num_query, num_points, 2)
            sampling_loc = sampling_loc.permute(0, 2, 1, 3, 4)
            # (N, num_query, num_points, 2)
            grid = sampling_loc.reshape(N, num_query, num_points, 2)

            # grid_sample expects (N, C, H, W) and grid (N, H_out, W_out, 2)
            sampled = F.grid_sample(
                value_l,
                grid,
                mode='bilinear',
                padding_mode='zeros',
                align_corners=False
            )  # (N, head_dim, num_query, num_points)

            # reshape back: (B, num_query, num_heads, num_points, head_dim)
            sampled = sampled.view(B, num_heads, head_dim, num_query, num_points)
            sampled = sampled.permute(0, 3, 1, 4, 2)

            # apply attention weights
            attn = attention_weights[:, :, :, lvl, :].unsqueeze(-1)
            output += (sampled * attn).sum(dim=3)

        # merge heads: (B, num_query, C)
        return output.view(B, num_query, C)