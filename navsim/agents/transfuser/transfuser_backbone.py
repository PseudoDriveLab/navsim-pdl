import copy
import timm
import torch
import torch.nn as nn
from .fpn import FPN
from .bev_encoder_layer import BEVFormerLayer
from .bevformer_encoder import BEVFormerEncoder
from navsim.agents.transfuser.transfuser_config import TransfuserConfig

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

    def forward(self, camera_feature: torch.Tensor, lidar_feature: torch.Tensor, R_list: torch.Tensor, t_list: torch.Tensor, K_list: torch.Tensor):
        # 1) backbone + FPN
        print(camera_feature.shape, R_list.shape, t_list.shape, K_list.shape)
        _, _, H_img, W_img = camera_feature.shape
        lidar2img = []
        for i in range(8):
            R = R_list[i]           # (3,3)
            t = t_list[i].view(3,1) # (3,1)
            K = K_list[i]           # (3,3)
            ext = torch.eye(4, device=R.device)
            ext[:3,:3] = R
            ext[:3,3:] = t
            K4 = torch.eye(4, device=K.device)
            K4[:3,:3] = K
            P = K4 @ ext          # (4×4)
            lidar2img.append(P)
        lidar2img = torch.stack(lidar2img, dim=0)  # (6,4,4)

        feats    = self.backbone(camera_feature)
        fpn_outs = self.fpn(feats)
        bev_feat = fpn_outs[-1]               # (B, 256, H, W)
        B,C,H,W  = bev_feat.shape

        # 2) BEV query → Encoder
        bev_q = bev_feat.flatten(2).permute(2,0,1)  # (num_q, B, 256)
        spatial_shapes    = torch.tensor([[H,W]], device=bev_feat.device)
        level_start_index = torch.tensor([0],    device=bev_feat.device)
        enc_out = self.encoder(
            bev_q, bev_q, bev_q,
            bev_h=H, bev_w=W,
            bev_pos=None,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            prev_bev=self.prev_bev,
            lidar2img=lidar2img,
            img_shape=(H_img, W_img)
        )
        # reshape back to map (B,256,H,W)
        bev_feat2 = enc_out.permute(1,2,0).reshape(B, C, H, W)
        self.prev_bev = enc_out.detach()
        # 3) Top‑Down FPN → BEV semantic map feature (64ch)
        p5 = self.td_conv1(bev_feat2)             # 256→64
        p4 = self.relu(self.td_conv2(self.td_upsample(p5)))
        p3 = self.relu(self.td_conv3(self.td_upsample2(p4)))
        bev_feature_upscale = p3                  # (B,64, 4×H, 4×W) if two upsamples

        # 4) Return exactly 3 items for TransfuserModel compatibility
        #    bev_feature_upscale (for semantic head),
        #    bev_feat2           (for decoder),
        #    None                (placeholder)
        return bev_feature_upscale, bev_feat2, None
