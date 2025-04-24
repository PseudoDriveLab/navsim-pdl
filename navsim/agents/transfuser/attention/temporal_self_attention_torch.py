import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .multi_scale_deform_attn_function_fp32_torch import MultiScaleDeformAttnFunction_fp32_torch

class TemporalSelfAttention(nn.Module):
    def __init__(self, embed_dims=256, num_heads=8, num_levels=4, num_points=4,
                 num_bev_queue=2, im2col_step=64, dropout=0.1):
        super(TemporalSelfAttention, self).__init__()
        if embed_dims % num_heads != 0:
            raise ValueError("embed_dims must be divisible by num_heads")
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.num_bev_queue = num_bev_queue
        self.im2col_step = im2col_step
        self.dropout = nn.Dropout(dropout)
        self.ms_deform_attn = MultiScaleDeformAttnFunction_fp32_torch(im2col_step)
        self.sampling_offsets = nn.Linear(embed_dims*num_bev_queue,
                                          num_bev_queue*num_heads*num_levels*num_points*2)
        self.attention_weights = nn.Linear(embed_dims*num_bev_queue,
                                           num_bev_queue*num_heads*num_levels*num_points)
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self._init_weights()
    
    def _init_weights(self):
        nn.init.constant_(self.sampling_offsets.weight, 0.)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (2*math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], dim=-1)
        grid_init = grid_init / grid_init.abs().max(dim=-1, keepdim=True)[0]
        grid_init = grid_init.view(self.num_heads, 1, 1, 2).repeat(1, self.num_levels*self.num_bev_queue, self.num_points, 1)
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= (i+1)
        self.sampling_offsets.bias.data = grid_init.view(-1)
        nn.init.constant_(self.attention_weights.weight, 0.)
        nn.init.constant_(self.attention_weights.bias, 0.)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.constant_(self.value_proj.bias, 0.)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.output_proj.bias, 0.)
    
    def forward(self,
                query,
                key=None,
                value=None,
                identity=None,
                query_pos=None,
                key_padding_mask=None,
                reference_points=None,
                spatial_shapes=None,
                level_start_index=None,
                flag='decoder',
                **kwargs):
        if value is None:
            bs, len_bev, c = query.shape
            value = torch.stack([query, query], 1).reshape(bs*2, len_bev, c)
        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos

        bs, num_query, _ = query.shape
        _, num_value, _ = value.shape
        # print('query shape: ', query.shape)
        # print('value shape: ', value.shape)
        # print(spatial_shapes.shape)
        # print((spatial_shapes[:,0] * spatial_shapes[:,1]).sum())
        spatial_shapes=torch.tensor([[kwargs['bev_h'], kwargs['bev_w']]], device=query.device)
        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value
        
        combined = torch.cat([value[:bs], query], dim=-1)
        
        value = self.value_proj(value)
        sampling_offsets = self.sampling_offsets(combined)
        attention_weights = self.attention_weights(combined)
        attention_weights = F.softmax(attention_weights, dim=-1)

        # queue 차원 포함
        sampling_offsets = sampling_offsets.view(
            bs, num_query, self.num_heads, self.num_bev_queue,
            self.num_levels, self.num_points, 2)            # (bs,q,h,Q,Lv,P,2)

        attention_weights = attention_weights.view(
            bs, num_query, self.num_heads, self.num_bev_queue,
            self.num_levels, self.num_points)               # (bs,q,h,Q,Lv,P)

        # deform-attn 으로 넘길 때 (bs*Q, …) 형태로 펼치기
        sampling_offsets = sampling_offsets.permute(0, 3, 1, 2, 4, 5, 6)\
                                        .reshape(bs*self.num_bev_queue, num_query,
                                                    self.num_heads, self.num_levels,
                                                    self.num_points, 2)

        attention_weights = attention_weights.permute(0, 3, 1, 2, 4, 5)\
                                            .reshape(bs*self.num_bev_queue, num_query,
                                                    self.num_heads, self.num_levels,
                                                    self.num_points)

        output = self.ms_deform_attn(
            value,
            spatial_shapes,
            level_start_index,
            sampling_offsets,
            attention_weights
        )
        # --- queue → batch 축소 ---
        output = output.view(self.num_bev_queue, bs, num_query, self.embed_dims).mean(0)

        output = self.output_proj(output)
        return self.dropout(output) + identity