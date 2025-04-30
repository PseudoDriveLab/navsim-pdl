import torch
import torch.nn as nn
from .ms_deformable_attention3d_torch import MSDeformableAttention3D

class SpatialCrossAttention(nn.Module):
    def __init__(self, embed_dims=256, num_cams=4, dropout=0.1, deformable_attention_config=None):
        super().__init__()
        cfg = deformable_attention_config or {}
        self.deformable_attention = MSDeformableAttention3D(**cfg)
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.output_proj.bias, 0.0)
        self.dropout = nn.Dropout(dropout)
        self.num_cams = num_cams

    def forward(self, query, query_pos=None, key=None, value=None,
                reference_points_cam=None, spatial_shapes=None, level_start_index=None):
        B, L_bev, C = query.shape
        Nc = self.num_cams
        P = reference_points_cam.size(3)

        # 1) positional encoding 반영
        if query_pos is not None:
            query = query + query_pos

        # 2) query 복제 및 flatten
        query_flat = query.unsqueeze(1).expand(-1, Nc, -1, -1).reshape(B * Nc, L_bev, C)
        query_pos_flat = None
        if query_pos is not None:
            query_pos_flat = query_pos.unsqueeze(1).expand(-1, Nc, -1, -1).reshape(B * Nc, L_bev, C)

        # 3) reference points flatten
        ref_flat = reference_points_cam.permute(1, 0, 2, 3, 4).reshape(B * Nc, L_bev, P, 2)
        # print(f"[SCA] query_flat   shape={query_flat.shape}")
        # print(f"[SCA] key/value    shape={key.shape}/{value.shape}")
        # print(f"[SCA] ref_flat     shape={ref_flat.shape}")
        # 4) deformable attention 호출
        out = self.deformable_attention(
            query=query_flat,
            query_pos=query_pos_flat,
            key=key,
            value=value,
            reference_points=ref_flat,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index
        )  # (B*Nc, L_bev, C)
        
        # 5) camera 축 합치기
        out = out.view(B, Nc, L_bev, C).mean(dim=1)  # (B, L_bev, C)

        # 6) projection + residual
        out = self.output_proj(out) + query
        return self.dropout(out)
