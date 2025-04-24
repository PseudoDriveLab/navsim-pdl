import copy
import timm
import torch
import torch.nn as nn
import math
from .fpn import FPN
from .bev_encoder_layer import BEVFormerLayer
from .bevformer_encoder import BEVFormerEncoder
from navsim.agents.transfuser.transfuser_config import TransfuserConfig

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

class BEVFormerV2(nn.Module):
    """
    BEVFormerV2 with added Top‑Down FPN for semantic head compatibility.
    Returns (bev_feature_upscale, bev_feature, None).
    """
    def __init__(self, cfg: TransfuserConfig):
        super().__init__()
        # 1) ResNet50 → FPN
        self.dim    = cfg._dim_                   # 256
        self.pc_range = cfg.pc_range
        self.num_points_in_pillar = cfg.num_points_in_pillar
        self.num_mono_levels = cfg._num_mono_levels_
        self.prev_bev = None
        self.level_embeds = nn.Parameter(torch.randn(cfg._num_mono_levels_, self.dim))
        self.cams_embeds  = nn.Parameter(torch.randn(cfg.num_cams, self.dim))
        bev_h = cfg.bev_grid_h   # 예) 30
        bev_w = cfg.bev_grid_w
        self.bev_queries = nn.Parameter(torch.zeros(bev_h * bev_w, self.dim))
        nn.init.xavier_uniform_(self.bev_queries)
        self.register_buffer('bev_pos',
                             build_sine_bev_pos(bev_h, bev_w, self.dim, device='cpu'),
                             persistent=False)
        # backbone
        self.backbone = timm.create_model(
            cfg.image_architecture, pretrained=True,
            features_only=True, out_indices=(1,2,3)
        )
        in_chs = self.backbone.feature_info.channels()
        self.fpn = FPN(
            in_channels=in_chs,
            out_channels=self.dim,
            num_outs=self.num_mono_levels,
            start_level=0,
            add_extra_convs='on_output',
            relu_before_extra_convs=True
        )

        # 2) BEV Encoder (Temporal + Spatial)
        temp_cfg = dict(
            embed_dims=cfg.tf_d_model,
            num_heads=cfg.tf_num_head,
            num_levels=1,
            num_points=cfg.num_points_in_pillar,
            num_bev_queue=cfg.get('num_bev_queue',2),
            im2col_step=64,
            dropout=cfg.tf_dropout
        )
        spat_cfg = cfg.deformable_attention.copy()
        spat_cfg.pop('type', None)
        spat_cfg.update({
            'embed_dims': cfg.tf_d_model,
            'num_cams':   cfg.num_cams,
            'pc_range':   cfg.pc_range,
            'dropout':    cfg.tf_dropout,
            'batch_first': False,
        })
        base_layer = BEVFormerLayer(
            attn_cfgs=[temp_cfg, spat_cfg],
            feedforward_channels=cfg.tf_d_ffn,
            ffn_dropout=cfg.tf_dropout,
            operation_order=('self_attn','norm','cross_attn','norm','ffn','norm')
        )
        layers = nn.ModuleList([copy.deepcopy(base_layer)
                                 for _ in range(cfg.tf_num_layers)])
        self.encoder = BEVFormerEncoder(
            layers=layers,
            pc_range=cfg.pc_range,
            num_points_in_pillar=cfg.num_points_in_pillar,
            return_intermediate=getattr(cfg, 'return_intermediate', False),
        )

        # 3) Top‑Down FPN for BEV semantic (as in original TransfuserBackbone)
        c = cfg.bev_features_channels  # default 64
        self.td_conv1   = nn.Conv2d(self.dim, c, kernel_size=1)
        self.td_conv2   = nn.Conv2d(c, c, kernel_size=3, padding=1)
        self.td_conv3   = nn.Conv2d(c, c, kernel_size=3, padding=1)
        self.td_upsample  = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.td_upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self,
                camera_feature: torch.Tensor,       # (B,N_cam,3,H,W)
                lidar_feature:  torch.Tensor,       # (unused)
                R_list: torch.Tensor,               # (B,N_cam,3,3)
                t_list: torch.Tensor,               # (B,N_cam,3)
                K_list: torch.Tensor):              # (B,N_cam,3,3)

        if camera_feature.dtype == torch.uint8:
            camera_feature = camera_feature.float() / 255.0
        B, N_cam, _, H_img, W_img = camera_feature.shape

        # 1) backbone + FPN
        feats = self.backbone(camera_feature.flatten(0, 1))
        mlvl_feats = [
            f.view(B, N_cam, -1, f.shape[-2], f.shape[-1]) for f in self.fpn(feats)
        ]  # list[(B,N_cam,dim,H,W)]

        # 2) lidar2img
        lidar2img = torch.zeros((B, N_cam, 4, 4), device=camera_feature.device)
        for b in range(B):
            for i in range(N_cam):
                ext = torch.eye(4, device=camera_feature.device)
                ext[:3, :3] = R_list[b, i]
                ext[:3, 3]  = t_list[b, i]
                K4 = torch.eye(4, device=camera_feature.device)
                K4[:3, :3] = K_list[b, i]
                lidar2img[b, i] = K4 @ ext

        # 3) feat_flatten (+ embeds)
        feat_flatten = []
        spatial_shapes = []
        for lvl, feat in enumerate(mlvl_feats):
            bs, num_cam, c, h, w = feat.shape
            spatial_shape = (h, w)
            feat = feat.flatten(3).permute(1, 0, 3, 2)
            feat = feat + self.cams_embeds[:, None, None, :].to(feat.dtype)
            feat = feat + self.level_embeds[None, None, lvl:lvl + 1, :].to(feat.dtype)
            spatial_shapes.append(spatial_shape)
            feat_flatten.append(feat)
        
        feat_flatten = torch.cat(feat_flatten, 2)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=self.bev_pos.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))

        feat_flatten = feat_flatten.permute(0, 2, 1, 3)  # (num_cam, H*W, bs, embed_dims)

        # 4) BEV query / pos
        bev_q  = self.bev_queries.unsqueeze(1).repeat(1, B, 1)
        bev_pos = self.bev_pos.to(feat_flatten.device).repeat(B, 1, 1, 1)
        bev_pos = bev_pos.flatten(2).permute(2, 0, 1)
        # 5) Encoder
        enc_out = self.encoder(
            bev_q, feat_flatten, feat_flatten,
            bev_h=self.bev_pos.shape[2],
            bev_w=self.bev_pos.shape[3],
            bev_pos=bev_pos,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            prev_bev=self.prev_bev,
            lidar2img=lidar2img,
            img_shape=(H_img, W_img)
        )  # (num_q,B,dim)
        self.prev_bev = enc_out.detach()
        
        # 6) BEV map + Top‑Down FPN
        bev_map = enc_out.permute(1, 2, 0).reshape(
            B, self.dim, self.bev_pos.shape[2], self.bev_pos.shape[3])
        p5 = self.td_conv1(bev_map)                               # 256 → 64
        p4 = self.relu(self.td_conv2(self.td_upsample(p5)))        # 2× up
        p3 = self.relu(self.td_conv3(self.td_upsample2(p4)))       # 2× up

        bev_feature_upscale = p3                                   # (B, 64, H·4, W·4)
        return bev_feature_upscale, bev_map, None
