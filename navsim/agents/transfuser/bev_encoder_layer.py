import torch
import torch.nn as nn
from .attention import TemporalSelfAttention, SpatialCrossAttention

class BEVFormerLayer(nn.Module):
    def __init__(
        self,
        attn_cfgs,
        feedforward_channels=512,
        ffn_dropout=0.1,
        operation_order=('self_attn','norm','cross_attn','norm','ffn','norm')
    ):
        super().__init__()
        self.operation_order = operation_order
        self.num_cams = attn_cfgs[1]['num_cams']
        
        # Temporal Self-Attn
        tcfg = attn_cfgs[0]
        self.temporal_attn = TemporalSelfAttention(
            embed_dims=tcfg['embed_dims'],
            num_heads=tcfg['num_heads'],
            num_levels=tcfg['num_levels'],
            num_points=tcfg['num_points'],
            num_bev_queue=tcfg['num_bev_queue'],
            im2col_step=tcfg['im2col_step'],
            dropout=tcfg['dropout']
        )
        self.norm1 = nn.LayerNorm(tcfg['embed_dims'])

        # Spatial Cross-Attn
        scfg = attn_cfgs[1]
        self.spatial_attn = SpatialCrossAttention(
            embed_dims=scfg['embed_dims'],
            num_cams=scfg['num_cams'],
            deformable_attention_config=scfg['deformable_attention'],
            dropout=scfg['dropout']
        )
        self.norm2 = nn.LayerNorm(scfg['embed_dims'])

        # FFN
        self.ffn   = nn.Sequential(
            nn.Linear(scfg['embed_dims'], feedforward_channels),
            nn.ReLU(),
            nn.Dropout(ffn_dropout),
            nn.Linear(feedforward_channels, scfg['embed_dims'])
        )
        self.norm3 = nn.LayerNorm(scfg['embed_dims'])

    def forward(self, bev_query, key, value, **kwargs):
        bev_h              = kwargs['bev_h']
        bev_w              = kwargs['bev_w']
        bev_query_pos = kwargs.get('bev_pos', None)
        hybrid_ref2d       = kwargs['reference_points']       # (2B, Q, Lv, 2)
        ref3d              = kwargs['reference_points_cam']   # (Nc, B, Q, P, 2)
        spatial_shapes     = kwargs['spatial_shapes']         # (Lv,2)
        level_start_index  = kwargs['level_start_index']      # (Lv,)
        prev_bev           = kwargs.get('prev_bev', None)     # 혹은 None

        # 1) BEV 크기
        B, Q, C = bev_query.shape

        # 2) Temporal Self-Attn 용 합치기
        if prev_bev is None:
            # 첫 프레임: history=current
            # (B, Q, C) → (B,2,Q,C) → (2B, Q, C)
            combined = bev_query.unsqueeze(1) \
                            .repeat(1, 2, 1, 1) \
                            .view(2*B, Q, C)
        else:
            # prev_bev 자체가 (2B, Q, C) 로 이미 만들어져 있음
            combined = prev_bev

        # 3) TemporalSelfAttention 호출
        #    (2B, Q, C) → same shape out2
        out2 = self.temporal_attn(
            query=combined,
            reference_points=hybrid_ref2d,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            bev_h=bev_h,
            bev_w=bev_w
        )  # → (2B, Q, C)

        # 4) 뒤 절반(현재 프레임)만 떼어내기
        out = out2[B:]       # (2B, Q, C)[B:2B] → (B, Q, C)
        out = self.norm1(out)

        # 5) Spatial Cross-Attn
        #    key/value: (Nc, HW, B, C) 형식
        B, Nc, HW, C = key.shape[2], self.num_cams, key.shape[1], key.shape[-1]
        # (Nc, HW, B, C) → (B, Nc, HW, C) → (B*Nc, HW, C)
        key_flat = key.permute(2, 0, 1, 3).reshape(B * Nc, HW, C)
        val_flat = value.permute(2, 0, 1, 3).reshape(B * Nc, HW, C)
        out = self.spatial_attn(
            query=out,
            query_pos=bev_query_pos,
            key=key_flat,
            value=val_flat,
            reference_points_cam=ref3d,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
        )
        out = self.norm2(out)

        # 6) FFN & residual
        res = out
        out = self.ffn(out)
        out = res + out
        return self.norm3(out)
