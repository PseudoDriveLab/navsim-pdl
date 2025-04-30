import torch
import torch.nn as nn
import torch.nn.functional as F
from .multi_scale_deform_attn_function_fp32_torch import MultiScaleDeformAttnFunction_fp32_torch

class TemporalSelfAttention(nn.Module):
    def __init__(
        self,
        embed_dims: int = 256,
        num_heads: int = 8,
        num_levels: int = 1,
        num_points: int = 4,
        num_bev_queue: int = 2,
        im2col_step: int = 64,
        dropout: float = 0.1
    ):
        super().__init__()
        assert embed_dims % num_heads == 0

        self.embed_dims    = embed_dims
        self.num_heads     = num_heads
        self.num_levels    = num_levels
        self.num_points    = num_points
        self.num_bev_queue = num_bev_queue

        # offset / weight 레이어
        self.sampling_offsets = nn.Linear(
            embed_dims * num_bev_queue,
            num_bev_queue * num_heads * num_levels * num_points * 2
        )
        self.attention_weights = nn.Linear(
            embed_dims * num_bev_queue,
            num_bev_queue * num_heads * num_levels * num_points
        )
        self.value_proj  = nn.Linear(embed_dims, embed_dims)
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.dropout     = nn.Dropout(dropout)
        self.ms_deform_attn = MultiScaleDeformAttnFunction_fp32_torch(im2col_step)

        # 초기화
        nn.init.constant_(self.sampling_offsets.weight, 0.0)
        nn.init.constant_(self.attention_weights.weight, 0.0)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.xavier_uniform_(self.output_proj.weight)

    def forward(
        self,
        query: torch.Tensor,               # (B, Q, C)
        reference_points: torch.Tensor,    # (B, Q, Lv, 2 or 4)
        spatial_shapes: torch.Tensor,      # (Lv, 2)
        level_start_index: torch.Tensor,   # (Lv,)
        bev_h: int,
        bev_w: int
    ) -> torch.Tensor:
        B, Q, C = query.shape

        # 1) history+current 결합 ▶ (B·queue, Q, C)
        value = query.unsqueeze(1).repeat(1, self.num_bev_queue, 1, 1)\
                     .view(B * self.num_bev_queue, Q, C)
        value_proj = self.value_proj(value)  # (B·queue, Q, C)

        # 2) residual
        identity = query

        # 3) offset/weight 추출을 위한 cat ▶ (B, Q, C·queue)
        cat = torch.cat([
            value_proj.view(B, self.num_bev_queue, Q, C)[:, 0],  # history
            query                                               # current
        ], dim=-1)
        
        sampling_offsets = self.sampling_offsets(cat).view(
            B, Q, self.num_heads, self.num_bev_queue,
            self.num_levels, self.num_points, 2
        )
        attention_weights = self.attention_weights(cat).view(
            B, Q, self.num_heads, self.num_bev_queue,
            self.num_levels, self.num_points
        )
        attention_weights = F.softmax(attention_weights, dim=-1)

        # 4) sampling_locations 계산
        if reference_points.shape[-1] == 2:
            # restore original global BEV normalization
            # normalizer: (1,1,1,1,1,2) → [bev_w, bev_h]
            normalizer = torch.tensor([bev_w, bev_h], device=query.device) \
                .view(1,1,1,1,1,2)
            # ref: (B, Q, 1, 1, Lv, 1, 2)
            ref = reference_points.unsqueeze(2).unsqueeze(3).unsqueeze(5)
            
            sampling_locations = ref + sampling_offsets / normalizer

            # 3) 각 레벨별 분모로 나눠 줘서 sampling_locations 계산
            sampling_locations = ref + sampling_offsets / normalizer
            # print(f"[TSA] sampling_offsets  min={sampling_offsets.min():.4f}, max={sampling_offsets.max():.4f}")
            # print(f"[TSA] level_norm         {spatial_shapes.tolist()}")
            # print(f"[TSA] level_norm         {normalizer.shape}")
            # print(f"[TSA] sampling_locations min={sampling_locations.min():.4f}, max={sampling_locations.max():.4f}")
        else:
            offset_norm = spatial_shapes[:, None].unsqueeze(0).unsqueeze(0).unsqueeze(0)  # (1,1,1,Lv,1,2)
            sampling_offsets = sampling_offsets / offset_norm
            sampling_locations = reference_points[..., None, None, :2] + sampling_offsets

            # print(f"[TSA] sampling_offsets  min={sampling_offsets.min():.4f}, max={sampling_offsets.max():.4f}")
            # print(f"[TSA] offset_norm        {spatial_shapes.tolist()}")
            # print(f"[TSA] sampling_locations min={sampling_locations.min():.4f}, max={sampling_locations.max():.4f}")
        # 5) (B, Q, H, queue, Lv, P, 2) → (B·queue, Q, H, Lv, P, 2)
        Bq, Qq, H, Qp, Lv, P, _ = sampling_locations.shape
        sampling_locations = (
            sampling_locations
            .permute(0, 3, 1, 2, 4, 5, 6)    # (B, queue, Q, H, Lv, P, 2)
            .reshape(B * self.num_bev_queue, Q, H, Lv, P, 2)
        )
        attention_weights = (
            attention_weights
            .permute(0, 3, 1, 2, 4, 5)        # (B, queue, Q, H, Lv, P)
            .reshape(B * self.num_bev_queue, Q, H, Lv, P)
        )
        bev_shape = torch.tensor([[bev_h, bev_w]], device=query.device, dtype=torch.long)      # (1,2)
        bev_index = torch.tensor([0], device=query.device, dtype=torch.long)  
        # 6) deformable attention
        out = self.ms_deform_attn(
            value_proj,           # (B·queue, Q, C)
            bev_shape,       # (Lv, 2)
            bev_index,    # (Lv,)
            sampling_locations,   # (B·queue, Q, H, Lv, P, 2)
            attention_weights     # (B·queue, Q, H, Lv, P)
        )

        # 7) queue 평균 + projection + residual
        out = out.view(self.num_bev_queue, B, Q, C).mean(0)  # (B, Q, C)
        out = self.output_proj(out)
        return self.dropout(out) + identity
