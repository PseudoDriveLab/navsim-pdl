import torch
import math

def build_sine_bev_pos(bev_h: int, bev_w: int, dim: int, device):
    """
    BEVFormer 원본 형태의 2D sin‑cos positional encoding
    out: (1, dim, H, W)  –  gradient X (buffer)
    """
    assert dim % 4 == 0, "dim must be divisible by 4"
    dim_quarter = dim // 4                          # 64 when dim=256

    y_embed, x_embed = torch.meshgrid(
        torch.arange(bev_h, device=device).float() + 0.5,
        torch.arange(bev_w, device=device).float() + 0.5,
        indexing="ij"
    )                                               # (H, W)

    inv_freq = 1.0 / (10000 ** (
        torch.arange(dim_quarter, device=device).float() / dim_quarter
    ))                                              # (64,)

    pos_x = (x_embed.reshape(-1, 1) * inv_freq)     # (H*W, 64)
    pos_y = (y_embed.reshape(-1, 1) * inv_freq)

    emb = torch.cat((
        pos_x.sin(), pos_x.cos(),                  # 2×64
        pos_y.sin(), pos_y.cos()                   # 2×64
    ), dim=1)                                      # (H*W, 256)

    # (1, dim, H, W)
    return emb.t().view(1, dim, bev_h, bev_w)


def get_sinusoidal_embedding(num, dim):
    """
    num: 만들고자 하는 임베딩 개수 (예: Nc 또는 num_levels)
    dim: 임베딩 차원 (C)
    returns: (num, dim) 텐서
    """
    pe = torch.zeros(num, dim)
    position = torch.arange(0, num, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, dim, 2, dtype=torch.float32) * -(math.log(10000.0) / dim)
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe  # (num, dim)