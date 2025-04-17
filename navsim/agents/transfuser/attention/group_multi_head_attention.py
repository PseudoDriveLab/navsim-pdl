import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings

class GroupMultiheadAttention(nn.Module):
    """Group Multihead Attention.
    
    nn.MultiheadAttention를 기반으로 하며, 그룹별로 입력 텐서를 split하여 병렬로 attention을 계산한 후 합칩니다.
    또한 positional encoding을 입력으로 받아 더할 수 있습니다.
    
    Args:
        embed_dims (int): 임베딩 차원.
        num_heads (int): Attention 헤드 수.
        attn_drop (float): Attention dropout 비율.
        proj_drop (float): Projection dropout 비율.
        group (int): 그룹 수.
        batch_first (bool): True이면 입력 텐서가 (batch, seq, embed_dim) 형태.
    """
    def __init__(self,
                 embed_dims,
                 num_heads,
                 attn_drop=0.0,
                 proj_drop=0.0,
                 group=1,
                 dropout_layer=dict(type='Dropout', drop_prob=0.0),
                 batch_first=False,
                 **kwargs):
        super(GroupMultiheadAttention, self).__init__()
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.group = group
        self.batch_first = batch_first
        
        self.attn = nn.MultiheadAttention(embed_dims, num_heads, dropout=attn_drop, **kwargs)
        self.proj_drop = nn.Dropout(proj_drop)
        # dropout_layer 대신 nn.Dropout을 사용 (구성에 따라 확장 가능)
        self.dropout_layer = nn.Dropout(dropout_layer.get('drop_prob', 0.0))
        
    def forward(self,
                query,
                key=None,
                value=None,
                identity=None,
                query_pos=None,
                key_pos=None,
                attn_mask=None,
                key_padding_mask=None,
                **kwargs):
        if key is None:
            key = query
        if value is None:
            value = key
        if identity is None:
            identity = query
        if key_pos is None and query_pos is not None:
            if query_pos.shape == key.shape:
                key_pos = query_pos
            else:
                warnings.warn(f'Missing key positional encoding in GroupMultiheadAttention.')
        if query_pos is not None:
            query = query + query_pos
        if key_pos is not None:
            key = key + key_pos

        # 입력 텐서 형태 변환: 기본적으로 nn.MultiheadAttention는 (seq, batch, embed_dim) 형태 사용
        if self.batch_first:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)

        num_queries = query.shape[0]
        bs = query.shape[1]
        if self.training:
            # 그룹별로 split 한 후 concat (group attention)
            query = torch.cat(query.split(num_queries // self.group, dim=0), dim=1)
            key = torch.cat(key.split(num_queries // self.group, dim=0), dim=1)
            value = torch.cat(value.split(num_queries // self.group, dim=0), dim=1)
        out = self.attn(query=query, key=key, value=value, attn_mask=attn_mask, key_padding_mask=key_padding_mask)[0]
        if self.training:
            out = torch.cat(out.split(bs, dim=1), dim=0)
        if self.batch_first:
            out = out.transpose(0, 1)
        return identity + self.dropout_layer(self.proj_drop(out))