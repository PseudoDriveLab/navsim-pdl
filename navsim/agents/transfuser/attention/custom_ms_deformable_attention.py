import math
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F

# multi_scale_deform_attn_function_fp32_torch.py 파일에 정의된 함수를 임포트합니다.
from .multi_scale_deform_attn_function_fp32_torch import MultiScaleDeformAttnFunction_fp32_torch

class CustomMSDeformableAttention(nn.Module):
    """
    Deformable DETR에서 사용되는 Attention 모듈.
    
    Deformable DETR: End-to-End Object Detection을 위한 Deformable Transformers 논문
    (https://arxiv.org/pdf/2010.04159.pdf)를 기반으로 합니다.
    
    Args:
        embed_dims (int): 입력 임베딩 차원 (기본값: 256).
        num_heads (int): 병렬 attention head의 수 (기본값: 8).
        num_levels (int): attention에 사용되는 feature map level의 수 (기본값: 4).
        num_points (int): 각 head마다 각 query에 대해 샘플링할 포인트 수 (기본값: 4).
        im2col_step (int): image-to-column 연산 시 step (기본값: 64).
        dropout (float): 결과에 적용할 dropout 비율 (기본값: 0.1).
        batch_first (bool): 입력 텐서의 배치 차원이 첫 번째인지 여부.
        norm_cfg (dict): 추가 normalization 설정 (미사용).
        init_cfg (dict): 초기화 설정 (미사용).
    """
    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 num_levels=4,
                 num_points=4,
                 im2col_step=64,
                 dropout=0.1,
                 batch_first=False,
                 norm_cfg=None,
                 init_cfg=None):
        super(CustomMSDeformableAttention, self).__init__()
        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, but got {embed_dims} and {num_heads}')
        dim_per_head = embed_dims // num_heads
        self.norm_cfg = norm_cfg
        self.dropout = nn.Dropout(dropout)
        self.batch_first = batch_first
        self.fp16_enabled = False  # FP16 지원 관련 flag
        self.ms_deform_attn = MultiScaleDeformAttnFunction_fp32_torch(im2col_step)
        # dim_per_head가 2의 거듭제곱인지 확인 (CUDA 효율을 위해 권장)
        def _is_power_of_2(n):
            if (not isinstance(n, int)) or (n < 0):
                raise ValueError(f'invalid input for _is_power_of_2: {n} (type: {type(n)})')
            return (n & (n - 1) == 0) and n != 0
        
        if not _is_power_of_2(dim_per_head):
            warnings.warn("You'd better set embed_dims such that each attention head has a dimension that is a power of 2 "
                          "for efficiency in CUDA implementations.")
        
        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        
        self.sampling_offsets = nn.Linear(embed_dims, num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(embed_dims, num_heads * num_levels * num_points)
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.init_weights()

    def init_weights(self):
        # nn.init를 사용하여 가중치 및 편향 초기화 (mmcv 함수 대신 torch.nn.init 활용)
        nn.init.constant_(self.sampling_offsets.weight, 0.)
        # 각 head에 대해 각 sampling point의 초기 grid 값을 계산하여 bias에 설정
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], dim=-1)  # (num_heads, 2)
        # 각 head마다 값을 정규화한 후 (num_heads, 1, 1, 2)
        grid_init = (grid_init / grid_init.abs().max(dim=-1, keepdim=True)[0]).view(self.num_heads, 1, 1, 2)
        grid_init = grid_init.repeat(1, self.num_levels, self.num_points, 1)
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= (i + 1)
        # bias 초기값을 grid_init으로 설정합니다.
        nn.init.constant_(self.sampling_offsets.bias, 0.)
        self.sampling_offsets.bias.data.copy_(grid_init.view(-1))
        
        nn.init.constant_(self.attention_weights.weight, 0.)
        nn.init.constant_(self.attention_weights.bias, 0.)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.constant_(self.value_proj.bias, 0.)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.output_proj.bias, 0.)
        self._is_init = True

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
        """
        Forward 함수: MultiScaleDeformable Attention 연산을 수행합니다.
        
        Args:
            query (Tensor): (num_query, bs, embed_dims)
            key (Tensor): (num_key, bs, embed_dims)
            value (Tensor): (num_key, bs, embed_dims)
            identity (Tensor): residual 연결용, None이면 query 사용.
            query_pos (Tensor): query에 대한 positional encoding.
            key_padding_mask (Tensor): (bs, num_key) 형태의 패딩 마스크.
            reference_points (Tensor): 정규화된 reference points, shape: (bs, num_query, num_levels, 2)
                또는 (bs, num_query, num_levels, 4).
            spatial_shapes (Tensor): 각 레벨의 spatial shape, shape: (num_levels, 2).
            level_start_index (Tensor): 각 레벨의 시작 인덱스, shape: (num_levels,).
            flag (str): 디코더 등 사용 시 구분용 (기본값: 'decoder').
        Returns:
            Tensor: (num_query, bs, embed_dims)의 결과.
        """
        if value is None:
            value = query
        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos

        if not self.batch_first:
            # (num_query, bs, embed_dims) -> (bs, num_query, embed_dims)
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)
            
        bs, num_query, _ = query.shape
        bs, num_value, _ = value.shape
        # spatial_shapes가 가진 (H, W) 곱의 합이 num_value와 일치해야 합니다.
        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value

        value = self.value_proj(value)
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], 0.0)
        value = value.view(bs, num_value, self.num_heads, -1)
        
        sampling_offsets = self.sampling_offsets(query).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 2)
        attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points)
        attention_weights = attention_weights.softmax(-1)
        attention_weights = attention_weights.view(bs, num_query, self.num_heads, self.num_levels, self.num_points)

        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack([spatial_shapes[:, 1], spatial_shapes[:, 0]], dim=-1)
            sampling_locations = reference_points[:, :, None, :, None, :] + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, :2] + \
                                 sampling_offsets / self.num_points * reference_points[:, :, None, :, None, 2:] * 0.5
        else:
            raise ValueError(f'Last dim of reference_points must be 2 or 4, but got {reference_points.shape[-1]} instead.')

        if torch.cuda.is_available() and value.is_cuda:
            # fp16 환경에서도 안정성을 위해 fp32 구현을 사용합니다.
            output = self.ms_deform_attn(
                value,               # (bs, num_value, embed_dims)
                spatial_shapes,      # (num_levels, 2)
                level_start_index,   # (num_levels,)
                sampling_locations,  # (bs, num_query, heads, levels, points, 2)
                attention_weights    # (bs, num_query, heads, levels, points)
            )
        else:
            # CUDA를 사용하지 않는 경우 효율적 CPU 구현이 필요합니다.
            # 필요에 따라 torch.nn.functional.grid_sample 등을 활용하여 구현할 수 있습니다.
            raise NotImplementedError("CPU version of multi-scale deformable attention is not implemented.")

        output = self.output_proj(output)
        if not self.batch_first:
            output = output.permute(1, 0, 2)
        return self.dropout(output) + identity
