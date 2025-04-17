import math
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention import GroupMultiheadAttention, CustomMSDeformableAttention

##############################################
# 1. Positional Encoding (PositionEmbeddingSine)
##############################################
class PositionEmbeddingSine(nn.Module):
    """Sine 기반 포지셔널 인코딩.
    
    Args:
        num_pos_feats (int): 위치 임베딩 feature의 수의 절반 (기본값: 128).
        temperature (float): 온도 파라미터 (기본값: 10000).
        normalize (bool): True이면 정규화 수행 (기본값: False).
        scale (float, optional): 정규화 시 사용되는 스케일. normalize가 True일 때 전달해야 함.
    """
    def __init__(self, num_pos_feats=128, temperature=10000, normalize=False, scale=None):
        super(PositionEmbeddingSine, self).__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and not normalize:
            raise ValueError("If scale is passed, normalize should be True")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x):
        """
        Args:
            x (Tensor): 입력 feature의 텐서, shape (batch, channels, height, width).
        Returns:
            pos (Tensor): Positional encoding, shape (batch, 2*num_pos_feats, height, width).
        """
        bs, _, H, W = x.shape
        # 마스크가 없다면 전체가 유효하다고 가정
        mask = torch.zeros((bs, H, W), device=x.device, dtype=torch.bool)
        not_mask = ~mask

        # 누적 합을 통해 각 pixel의 위치 값을 계산
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos

##############################################
# 2. FFN (Feed-Forward Network)
##############################################
class FFN(nn.Module):
    """Transformer의 Feed-Forward Network.
    
    이 FFN은 mmcv에서 사용되던 디테일을 그대로 반영합니다.
    여러 FC 레이어, 활성화 함수, 드롭아웃, 그리고 residual 연결 (add_identity)을 지원합니다.
    
    Args:
        embed_dims (int): 입력 임베딩 차원.
        feedforward_channels (int): FFN 내부 은닉층 차원.
        num_fcs (int): 사용될 FC 레이어 수 (기본값: 2).
        ffn_drop (float): 드롭아웃 비율 (기본값: 0.0).
        act_cfg (dict): 활성화 함수 설정, 예: dict(type='ReLU', inplace=True).
        add_identity (bool): residual 연결 여부 (기본값: True).
    """
    def __init__(self,
                 embed_dims=256,
                 feedforward_channels=512,
                 num_fcs=2,
                 ffn_drop=0.1,
                 act_cfg=dict(type='ReLU', inplace=True),
                 add_identity=True):
        super(FFN, self).__init__()
        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.num_fcs = num_fcs
        self.ffn_drop = ffn_drop
        self.add_identity = add_identity

        # 활성화 함수 선택: ReLU 또는 GELU (필요에 따라 확장 가능)
        if act_cfg['type'] == 'ReLU':
            self.activation = nn.ReLU(inplace=act_cfg.get('inplace', True))
        elif act_cfg['type'] == 'GELU':
            self.activation = nn.GELU()
        else:
            raise ValueError(f"Unsupported activation type: {act_cfg['type']}")

        # 여러 FC 레이어 생성
        self.fcs = nn.ModuleList()
        for i in range(num_fcs):
            in_channels = embed_dims if i == 0 else feedforward_channels
            out_channels = feedforward_channels if i < num_fcs - 1 else embed_dims
            self.fcs.append(nn.Linear(in_channels, out_channels))
        self.dropout = nn.Dropout(ffn_drop)

        self.init_weights()

    def init_weights(self):
        """모든 FC 레이어를 Xavier uniform으로 초기화하고 편향은 0으로 초기화."""
        for fc in self.fcs:
            nn.init.xavier_uniform_(fc.weight)
            nn.init.constant_(fc.bias, 0.)

    def forward(self, x):
        residual = x
        out = x
        for i, fc in enumerate(self.fcs):
            out = fc(out)
            if i < self.num_fcs - 1:
                out = self.activation(out)
                out = self.dropout(out)
        out = self.dropout(out)
        if self.add_identity:
            out = out + residual
        return out

##############################################
# 3. DetrTransformerDecoderLayer
##############################################
class DetrTransformerDecoderLayer(nn.Module):
    """DETR Transformer Decoder Layer (순수 PyTorch 구현).
    
    이 레이어는 mmcv에서 사용되던 모든 디테일(자기-attention, cross-attention, FFN, 정규화)을 그대로 반영합니다.
    
    구성 순서:
        1. Self-Attention → Norm
        2. Cross-Attention → Norm
        3. FFN → Norm
    
    Args:
        self_attn_cfg (dict): self-attention 설정 
            예: {'embed_dims': 256, 'num_heads': 8, 'dropout': 0.0, 'batch_first': True}
        cross_attn_cfg (dict): cross-attention 설정 (구성은 self_attn_cfg와 유사)
        ffn_cfg (dict): FFN 설정 
            예: {
                    'embed_dims': 256,
                    'feedforward_channels': 1024,
                    'num_fcs': 2,
                    'ffn_drop': 0.0,
                    'act_cfg': {'type': 'ReLU', 'inplace': True},
                    'add_identity': True
                  }
        norm_cfg (dict): 정규화 설정 (여기서는 LayerNorm만 사용)
    """
    def __init__(self,
                 self_attn_cfg=dict(type='GroupMultiheadAttention', embed_dims=256, num_heads=8, dropout=0.1, group=1, batch_first=True),
                 cross_attn_cfg=dict(type='CustomMSDeformableAttention', embed_dims=256, num_levels=1, batch_first=True),
                 ffn_cfg=dict(embed_dims=256, feedforward_channels=1024, num_fcs=2, ffn_drop=0.1,
                              act_cfg=dict(type='ReLU', inplace=True), add_identity=True),
                 norm_cfg=dict(type='LN')):
        super(DetrTransformerDecoderLayer, self).__init__()
        
        # batch_first가 True 임을 강제 (DETR의 모든 모듈은 batch_first여야 함)
        if 'batch_first' not in self_attn_cfg:
            self_attn_cfg['batch_first'] = True
        else:
            assert self_attn_cfg['batch_first'] is True, "self_attn_cfg['batch_first'] must be True"
        if 'batch_first' not in cross_attn_cfg:
            cross_attn_cfg['batch_first'] = True
        else:
            assert cross_attn_cfg['batch_first'] is True, "cross_attn_cfg['batch_first'] must be True"

        self.self_attn = GroupMultiheadAttention(**self_attn_cfg)
        self.cross_attn = CustomMSDeformableAttention(**cross_attn_cfg)
        self.embed_dims = self_attn_cfg['embed_dims']
        self.ffn = FFN(embed_dims=ffn_cfg['embed_dims'],
                       feedforward_channels=ffn_cfg['feedforward_channels'],
                       num_fcs=ffn_cfg.get('num_fcs', 2),
                       ffn_drop=ffn_cfg.get('ffn_drop', 0.1),
                       act_cfg=ffn_cfg.get('act_cfg', dict(type='ReLU', inplace=True)),
                       add_identity=ffn_cfg.get('add_identity', True))
        # mmcv에서는 build_norm_layer를 사용하지만 여기서는 nn.LayerNorm을 그대로 사용
        self.norm1 = nn.LayerNorm(self.embed_dims)
        self.norm2 = nn.LayerNorm(self.embed_dims)
        self.norm3 = nn.LayerNorm(self.embed_dims)

    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor = None,
                value: torch.Tensor = None,
                query_pos: torch.Tensor = None,
                key_pos: torch.Tensor = None,
                self_attn_mask: torch.Tensor = None,
                cross_attn_mask: torch.Tensor = None,
                key_padding_mask: torch.Tensor = None,
                **kwargs) -> torch.Tensor:
        """
        Args:
            query (Tensor): (batch, num_queries, embed_dims) 입력 query.
            key (Tensor, optional): (batch, num_keys, embed_dims) key. None이면 query 사용.
            value (Tensor, optional): (batch, num_keys, embed_dims) value. None이면 key 사용.
            query_pos (Tensor, optional): query에 대한 positional encoding.
            key_pos (Tensor, optional): key에 대한 positional encoding.
            self_attn_mask (Tensor, optional): self attention mask.
            cross_attn_mask (Tensor, optional): cross attention mask.
            key_padding_mask (Tensor, optional): key padding mask.
        Returns:
            Tensor: (batch, num_queries, embed_dims) 출력.
        """
        # 1. Self-Attention Block
        residual = query
        q_self = query if query_pos is None else query + query_pos
        self_attn_out, _ = self.self_attn(q=q_self, k=q_self, v=q_self, attn_mask=self_attn_mask)
        query = residual + self_attn_out
        query = self.norm1(query)

        # 2. Cross-Attention Block
        if key is None:
            key = query
        if value is None:
            value = key
        residual = query
        q_cross = query if query_pos is None else query + query_pos
        k_cross = key if key_pos is None else key + key_pos
        cross_attn_out, _ = self.cross_attn(q=q_cross, k=k_cross, v=value,
                                             attn_mask=cross_attn_mask,
                                             key_padding_mask=key_padding_mask)
        query = residual + cross_attn_out
        query = self.norm2(query)

        # 3. FFN Block
        residual = query
        ffn_out = self.ffn(query)
        query = residual + ffn_out
        query = self.norm3(query)

        return query
