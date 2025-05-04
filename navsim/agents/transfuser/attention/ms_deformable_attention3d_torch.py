import torch
import torch.nn as nn
import math
from .multi_scale_deform_attn_function_fp32_torch import MultiScaleDeformAttnFunction_fp32_torch

class MSDeformableAttention3D(nn.Module):
    def __init__(self, embed_dims=256, num_heads=8, num_levels=4, num_points=8,
                 im2col_step=64, dropout=0.1, batch_first=True):
        super(MSDeformableAttention3D, self).__init__()
        if embed_dims % num_heads != 0:
            raise ValueError("embed_dims must be divisible by num_heads")
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.im2col_step = im2col_step
        self.dropout = nn.Dropout(dropout)
        self.batch_first = batch_first
        self.ms_deform_attn = MultiScaleDeformAttnFunction_fp32_torch(im2col_step)
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        self.sampling_offsets = nn.Linear(
            embed_dims, num_heads * num_levels * num_points * 2
        )
        self.attention_weights = nn.Linear(
            embed_dims, num_heads * num_levels * num_points
        )
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.constant_(self.value_proj.bias, 0.)
        nn.init.constant_(self.sampling_offsets.weight, 0.)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (2 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], dim=-1)
        grid_init = grid_init / grid_init.abs().max(dim=-1, keepdim=True)[0]
        grid_init = grid_init.view(self.num_heads, 1, 1, 2).repeat(1, self.num_levels, self.num_points, 1)
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= (i + 1)
        self.sampling_offsets.bias.data = grid_init.view(-1)
        nn.init.constant_(self.attention_weights.weight, 0.)
        nn.init.constant_(self.attention_weights.bias, 0.)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.output_proj.bias, 0.)

    def forward(self, query, query_pos=None, key=None, value=None, identity=None,
                reference_points=None, spatial_shapes=None, level_start_index=None,
                key_padding_mask=None, **kwargs):
    #     print(">>> [MSDef3D DEBUG] reference_points:", 
    #   reference_points.shape, reference_points.dtype, reference_points.device)
    #     print(">>> [MSDef3D DEBUG] spatial_shapes:", 
    #   spatial_shapes.shape, spatial_shapes.dtype, spatial_shapes.device, spatial_shapes)
        if value is None:
            value = query
        if identity is None:
            identity = query

        # 1) positional encoding 반영
        if query_pos is not None:
            query = query + query_pos

        bs, num_query, _ = query.shape
        bs, num_value, _ = value.shape
        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value
        value = self.value_proj(value)  # (bs, N, C)

        # 2) sampling_offsets, attention_weights reshape
        sampling_offsets = self.sampling_offsets(query).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 2
        )
        
    #     print(">>> [MSDef3D DEBUG] sampling_offsets:", 
    #   sampling_offsets.shape, sampling_offsets.dtype, sampling_offsets.device, 
    #   "min/max=", sampling_offsets.min().item(), sampling_offsets.max().item())
        
        attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points)

        attention_weights = attention_weights.softmax(-1)

        attention_weights = attention_weights.view(bs, num_query,
                                                   self.num_heads,
                                                   self.num_levels,
                                                   self.num_points)
        
        if reference_points.shape[-1] == 2:
            # reference_points : (bs, nq, D, 2)  (D = num_Z_anchors)
            bs, nq, D, _ = reference_points.shape        # Z-anchor 수
            offset_norm = torch.stack([spatial_shapes[..., 1],
                                       spatial_shapes[..., 0]], -1)  # (L,2)
            bs, num_query, num_Z_anchors, xy = reference_points.shape
            # (bs, nq, 1, 1, 1, D, 2)
            ref = reference_points[:, :, None, None, None, :, :]

            # sampling_offsets : (bs,nq,H,L,P,2) → 정규화 후
            sampling_offsets = sampling_offsets / offset_norm[None, None, None, :, None, :]
            # (bs,nq,H,L,P,D,2)
            bs, num_query, num_heads, num_levels, num_all_points, xy = sampling_offsets.shape
            sampling_offsets = sampling_offsets.view(
                bs, num_query, num_heads, num_levels, num_all_points // num_Z_anchors, num_Z_anchors, xy)

            # (bs,nq,H,L,P,D,2)
            sampling_locations = ref + sampling_offsets
            bs, num_query, num_heads, num_levels, num_points, num_Z_anchors, xy = sampling_locations.shape
            assert num_all_points == num_points * num_Z_anchors
            # flatten P*D 축
            sampling_locations = sampling_locations.view(
                bs, num_query, num_heads, num_levels, num_all_points, xy)  # -> (bs,nq,H,L,P*D,2)
        else:
            raise ValueError("reference_points 마지막 차원은 2여야 합니다.")

        
        # print(">>> REF finite:", torch.isfinite(reference_points).all().item())
        # if not torch.isfinite(sampling_locations).all():
        #     print("  num_inf:", int(torch.isinf(reference_points).sum().item()),
        #         "num_nan:", int(torch.isnan(reference_points).sum().item()), "total:",
        #         int(reference_points.numel()))
        # print(">>> REF min/max:", float(reference_points.min()), float(reference_points.max()))

        # # 2) offsets/div 결과 통계
        # ratio = sampling_offsets / spatial_shapes_
        # print(">>> RATIO finite:", torch.isfinite(ratio).all().item())
        # print(">>> RATIO min/max:", float(ratio.min()), float(ratio.max()))

        # # 3) sampling_locations 자체 통계
        # print(">>> LOC finite:", torch.isfinite(sampling_locations).all().item())
        # print(">>> LOC has_inf:", torch.isinf(sampling_locations).any().item())
        # print(">>> LOC has_nan:", torch.isnan(sampling_locations).any().item())
        # if not torch.isfinite(sampling_locations).all():
        #     print("  num_inf:", int(torch.isinf(sampling_locations).sum().item()),
        #         "num_nan:", int(torch.isnan(sampling_locations).sum().item()))

        # print(">>> LOC overall min/max (ignoring inf):",
        #     float(torch.nan_to_num(sampling_locations, nan=float('nan'),
        #                             posinf=float('nan'), neginf=float('nan')).min()),
        #     float(torch.nan_to_num(sampling_locations, nan=float('nan'),
        #                             posinf=float('nan'), neginf=float('nan')).max()))
        # print(">>> [MSDef3D DEBUG] spatial_shapes_:", spatial_shapes_.shape, spatial_shapes_)
        # print(">>> [MSDef3D DEBUG] sampling_offsets/spatial_shapes_ min/max=",
        #     (sampling_offsets / spatial_shapes_).min().item(),
        #     (sampling_offsets / spatial_shapes_).max().item())
        # print(">>> [MSDef3D DEBUG] sampling_locations min/max=",
        #     sampling_locations.min().item(),
        #     sampling_locations.max().item())
        
        # for lvl in range(self.num_levels):
        #     grid_lvl = sampling_locations[..., lvl, :, :]  # (bs, num_query, num_heads, num_points, 2)
        #     print(f"[DEBUG MSDeform3D] lvl={lvl} grid_lvl min={grid_lvl.min():.4f} max={grid_lvl.max():.4f}")
        #     print(f"[DEBUG MSDeform3D] full sampling_locations shape={sampling_locations.shape}")
        #     print(f"[DEBUG MSDeform3D] attention_weights shape={attention_weights.shape}, min={attention_weights.min():.4f}, max={attention_weights.max():.4f}")

        # 4) deformable attention 호출
        output = self.ms_deform_attn(
            value,
            spatial_shapes,
            level_start_index,
            sampling_locations,
            attention_weights
        )

        # 5) projection + residual
        output = self.output_proj(output)
        return self.dropout(output) + identity
