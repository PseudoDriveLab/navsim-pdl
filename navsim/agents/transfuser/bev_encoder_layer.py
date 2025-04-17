import torch
import torch.nn as nn
from .attention import TemporalSelfAttention, MSDeformableAttention3D

class SpatialCrossAttention(nn.Module):
    def __init__(self, embed_dims=256, num_cams=6, pc_range=None, dropout=0.1,
                 deformable_attention_config=None, batch_first=False):
        super(SpatialCrossAttention, self).__init__()
        self.embed_dims = embed_dims
        self.num_cams = num_cams
        self.pc_range = pc_range
        self.batch_first = batch_first
        self.dropout = nn.Dropout(dropout)
        # deformable_attention_config로 전달된 파라미터를 사용하여 MSDeformableAttention3D를 생성
        # deformable_attention_config가 없다면 기본 인자를 사용합니다.
        self.deformable_attention = MSDeformableAttention3D(**(deformable_attention_config if deformable_attention_config is not None else {}))
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self._init_weight()
        
    def _init_weight(self):
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.output_proj.bias, 0.)
    
    def forward(self, query, key, value, reference_points_cam, bev_mask, residual=None, query_pos=None, 
                spatial_shapes=None, level_start_index=None, **kwargs):
        """
        Args:
            query (Tensor): shape (B, num_query, embed_dims)
            key, value: 사용하지 않으면 query를 대체
            reference_points_cam (Tensor): 각 카메라별 reference points, 
                shape 예: (num_cams, B, num_total_reference, D, 2)
            bev_mask (List[Tensor]): 각 이미지별, 카메라별 활성 query mask. 
                shape: [num_cams x B, 1, num_query] 또는 각 이미지별로 다른 형태로 주어짐.
        """
        if key is None:
            key = query
        if value is None:
            value = key
        inp_residual = query if residual is None else residual
        if query_pos is not None:
            query = query + query_pos

        B, num_query, C = query.size()

        # Step 1. bev_mask를 사용하여 각 카메라별 활성 query 인덱스 추출
        indexes = []  # 각 카메라별 활성 query index 리스트
        for cam_idx in range(self.num_cams):
            # 예시: bev_mask[cam_idx]의 shape이 (B, num_query)라고 가정하고,
            # 각 이미지별 활성 쿼리의 index를 추출합니다.
            # 실제 데이터에 맞게 bev_mask 전처리가 필요합니다.
            # 아래는 첫 번째 배치의 mask를 기준으로 인덱스 추출 후, 반복 적용하는 예시입니다.
            mask_per_cam = bev_mask[cam_idx]  # shape: (B, num_query)
            active_idx = mask_per_cam.nonzero(as_tuple=False)  # (num_active, 2) -> [batch_index, query_index]
            # 각 배치별로 분리: 여기서는 모든 배치가 동일한 index를 가진다고 가정
            # 실제로는 배치마다 다를 수 있으므로 별도 처리가 필요합니다.
            indexes.append(active_idx[:, 1])
        
        # 최대 활성 query 개수 (모든 카메라에서 동일한 max length를 사용)
        max_len = max([idx.numel() for idx in indexes])
        
        # Step 2. 각 카메라별로 query와 reference points를 재배치
        # 재배치된 텐서 초기화: (B, num_cams, max_len, C)
        queries_rebatch = query.new_zeros(B, self.num_cams, max_len, C)
        # reference_points_cam의 shape는 (num_cams, B, num_total_reference, D, 2)
        D = reference_points_cam.size(3)
        reference_points_rebatch = reference_points_cam.new_zeros(B, self.num_cams, max_len, D, 2)
        
        # 각 배치에 대해 카메라 별 활성 query를 재배치합니다.
        for b in range(B):
            for cam_idx in range(self.num_cams):
                idx = indexes[cam_idx]
                n_active = idx.numel()
                if n_active > 0:
                    queries_rebatch[b, cam_idx, :n_active, :] = query[b, idx, :]
                    reference_points_rebatch[b, cam_idx, :n_active, :, :] = reference_points_cam[cam_idx, b, idx, :, :]
                else:
                    # 만약 해당 카메라에 활성 query가 없다면 패스
                    continue
        
        # Step 3. reshape해서 deformable attention 호출
        # (B * num_cams, max_len, C)
        queries_reshape = queries_rebatch.view(B * self.num_cams, max_len, C)
        # reshape reference_points_rebatch: (B*num_cams, max_len, D, 2)
        ref_points_reshape = reference_points_rebatch.view(B * self.num_cams, max_len, D, 2)
        
        # 만약 key/value는 전체 query에 대해 제공되었다면 reshape가 필요하거나, 카메라 별로 분리되어 있는지 확인합니다.
        # 여기서는 간단하게 key, value를 reshape하지 않고 deformable_attention에 전달한다 가정합니다.
        out = self.deformable_attention(query=queries_reshape, key=key, value=value,
                                        reference_points=ref_points_reshape, 
                                        spatial_shapes=spatial_shapes, level_start_index=level_start_index, **kwargs)
        
        # out: (B*num_cams, max_len, C) -> reshape back to (B, num_cams, max_len, C)
        out = out.view(B, self.num_cams, max_len, C)
        
        # Step 4. 원래 query 순서에 맞게 결과 집계 (여기서는 단순히 평균하는 예시)
        # 초기화된 slots: (B, num_query, C)
        slots = query.new_zeros(B, num_query, C)
        count = query.new_zeros(B, num_query, 1)
        for b in range(B):
            for cam_idx in range(self.num_cams):
                idx = indexes[cam_idx]
                n_active = idx.numel()
                if n_active > 0:
                    # 쿼리 결과를 해당 index에 더합니다.
                    slots[b, idx, :] += out[b, cam_idx, :n_active, :]
                    count[b, idx, :] += 1
        # 0으로 나누는 것을 방지
        count = torch.clamp(count, min=1.0)
        slots = slots / count
        
        out_proj = self.output_proj(slots)
        return self.dropout(out_proj) + inp_residual

class BEVFormerLayer(nn.Module):
    def __init__(self, attn_cfgs, feedforward_channels=512, ffn_dropout=0.1, operation_order=('self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm')):
        super(BEVFormerLayer, self).__init__()
        self.operation_order = operation_order
        
        temporal_cfg = attn_cfgs[0]
        self.temporal_attn = TemporalSelfAttention(
            embed_dims=temporal_cfg.get('embed_dims', 256),
            num_heads=temporal_cfg.get('num_heads', 8),
            num_levels=temporal_cfg.get('num_levels', 1),
            num_points=temporal_cfg.get('num_points', 4),
            num_bev_queue=temporal_cfg.get('num_bev_queue', 2),
            im2col_step=temporal_cfg.get('im2col_step', 64),
            dropout=temporal_cfg.get('dropout', 0.1)
        )
        
        spatial_cfg = attn_cfgs[1]
        deformable_cfg = spatial_cfg.get('deformable_attention', {})
        self.spatial_attn = SpatialCrossAttention(
            embed_dims=spatial_cfg.get('embed_dims', 256),
            num_cams=spatial_cfg.get('num_cams', 6),
            pc_range=spatial_cfg.get('pc_range', None),
            dropout=spatial_cfg.get('dropout', 0.1),
            deformable_attention_config=deformable_cfg,
            batch_first=False
        )
        
        self.norm1 = nn.LayerNorm(256)
        self.norm2 = nn.LayerNorm(256)
        self.norm3 = nn.LayerNorm(256)
        self.ffn = nn.Sequential(
            nn.Linear(256, feedforward_channels),
            nn.ReLU(),
            nn.Dropout(ffn_dropout),
            nn.Linear(feedforward_channels, 256)
        )
        self.norm4 = nn.LayerNorm(256)
    
    def forward(self, bev_query, key, value, **kwargs):
        out = bev_query
        out = self.temporal_attn(out, key, value, **kwargs)
        out = self.norm1(out)
        out = self.spatial_attn(out, key, value, **kwargs)
        out = self.norm2(out)
        out = self.ffn(out)
        out = self.norm3(out)
        return out