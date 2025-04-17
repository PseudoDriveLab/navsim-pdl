import torch
import torch.nn as nn
import torch.nn.functional as F
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
        self.sampling_offsets = nn.Linear(embed_dims, num_heads*num_levels*num_points*2)
        self.attention_weights = nn.Linear(embed_dims, num_heads*num_levels*num_points)
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.constant_(self.value_proj.bias, 0.)
        nn.init.constant_(self.sampling_offsets.weight, 0.)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (2*math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], dim=-1)
        grid_init = grid_init / grid_init.abs().max(dim=-1, keepdim=True)[0]
        grid_init = grid_init.view(self.num_heads, 1, 1, 2).repeat(1, self.num_levels, self.num_points, 1)
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= (i+1)
        self.sampling_offsets.bias.data = grid_init.view(-1)
        nn.init.constant_(self.attention_weights.weight, 0.)
        nn.init.constant_(self.attention_weights.bias, 0.)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.output_proj.bias, 0.)
    
    def forward(self, query, key=None, value=None, identity=None, query_pos=None,
                key_padding_mask=None, reference_points=None, spatial_shapes=None,
                level_start_index=None, **kwargs):
        if value is None:
            value = query
        if identity is None:
            identity = query

        bs, num_query, _ = query.shape
        value = self.value_proj(value)  # (bs, N, embed_dims)

        # 1) sampling_offsets, attention_weights reshape
        sampling_offsets = self.sampling_offsets(query)  # (bs, num_query, heads*levels*points*2)
        sampling_offsets = sampling_offsets.view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 2
        )
        attention_weights = self.attention_weights(query)  # (bs, num_query, heads*levels*points)
        attention_weights = attention_weights.softmax(-1)
        attention_weights = attention_weights.view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points
        )

        # 2) 모듈 호출 (apply 아님)
        output = self.ms_deform_attn(
            value,                   # (bs, N, C)
            spatial_shapes,          # (levels, 2)
            level_start_index,       # (levels,)
            sampling_offsets,        # (bs, num_query, heads, levels, points, 2)
            attention_weights        # (bs, num_query, heads, levels, points)
        )

        output = self.output_proj(output)
        return self.dropout(output) + identity