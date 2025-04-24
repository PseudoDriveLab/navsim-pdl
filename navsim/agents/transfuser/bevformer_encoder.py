import torch
import torch.nn as nn
import numpy as np

# 단순 버전의 digit_version 함수 (mmcv 없이 버전 비교용)
def digit_version(version_str):
    # 예: "1.8.1" -> (1, 8)
    return tuple(map(int, version_str.split('.')[:2]))

TORCH_VERSION = torch.__version__

class BEVFormerEncoder(nn.Module):
    """
    BEVFormerEncoder는 BEV query를 초기화하고,
    공간 크로스 어텐션(Spatial Cross-Attention)과 시간적 셀프 어텐션(Temporal Self-Attention)을 통해
    멀티카메라 피처를 융합하여 현재 프레임의 BEV 정보를 생성합니다.
    
    이 구현은 mmcv 의존성을 제거하고 PyTorch 기본 모듈만 사용하며,
    원본 encoder.py의 디테일한 부분을 그대로 재현합니다.
    
    Args:
        layers (nn.ModuleList): 순차적으로 적용할 Transformer layer들의 리스트.
        pc_range (list or tuple): 포인트 클라우드 범위 [x_min, y_min, z_min, x_max, y_max, z_max].
        num_points_in_pillar (int): 각 pillar(기둥) 내에 샘플링할 포인트 수.
        return_intermediate (bool): 중간 layer의 출력을 반환할지 여부.
        dataset_type (str): 데이터셋 종류 (기본값 'nuscenes').
    """
    def __init__(self, layers, pc_range, num_points_in_pillar=4, return_intermediate=False):
        super(BEVFormerEncoder, self).__init__()
        self.layers = layers
        self.return_intermediate = return_intermediate
        self.num_points_in_pillar = num_points_in_pillar
        self.pc_range = pc_range
        self.fp16_enabled = False

    @staticmethod
    def get_reference_points(H, W, Z=8, num_points_in_pillar=4, dim='3d', bs=1, device='cuda', dtype=torch.float):
        """
        BEV grid에 대한 reference point를 생성합니다.
        3D 모드에서는 x, y, z 좌표를 균등 분할하여 반환하며,
        2D 모드에서는 BEV 평면상의 좌표를 반환합니다.
        """
        if dim == '3d':
            zs = torch.linspace(0.5, Z - 0.5, num_points_in_pillar, dtype=dtype, device=device).view(-1, 1, 1).expand(num_points_in_pillar, H, W) / Z
            xs = torch.linspace(0.5, W - 0.5, W, dtype=dtype, device=device).view(1, 1, W).expand(num_points_in_pillar, H, W) / W
            ys = torch.linspace(0.5, H - 0.5, H, dtype=dtype, device=device).view(1, H, 1).expand(num_points_in_pillar, H, W) / H
            # (num_points_in_pillar, H, W, 3)
            ref_3d = torch.stack((xs, ys, zs), -1)
            # 재배열: (num_points_in_pillar, H*W, 3)
            ref_3d = ref_3d.permute(0, 3, 1, 2).flatten(2).permute(0, 2, 1)
            # (bs, num_points_in_pillar, H*W, 3)
            ref_3d = ref_3d.unsqueeze(0).repeat(bs, 1, 1, 1)
            return ref_3d
        elif dim == '2d':
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, H - 0.5, H, dtype=dtype, device=device),
                torch.linspace(0.5, W - 0.5, W, dtype=dtype, device=device)
            )
            # (H, W) -> (1, H*W)
            ref_y = ref_y.reshape(-1)[None] / H
            ref_x = ref_x.reshape(-1)[None] / W
            # (1, H*W, 2)
            ref_2d = torch.stack((ref_x, ref_y), -1)
            # (bs, H*W, 1, 2)
            ref_2d = ref_2d.repeat(bs, 1, 1).unsqueeze(2)
            return ref_2d

    def point_sampling(
        self,
        reference_points,      # (bs, num_p, num_q, 3)
        lidar2img: torch.Tensor,# (bs, N_cam,4,4)
        img_shape: tuple[int,int]
    ):
        """
        3D ref points를 주어진 cam projection으로 투영하고 mask 생성.
        Args:
            reference_points: (bs, num_p, num_q, 3) in [0,1]
            lidar2img:         (bs, N_cam, 4, 4)
            img_shape:         (H_img, W_img)
        Returns:
            reference_points_cam: (N_cam, bs, num_q, num_p, 2)
            bev_mask:             (N_cam, bs, num_q, num_p)
        """
        bs, num_p, num_q, _ = reference_points.shape
        H_img, W_img = img_shape

        # 1) normalized → real-world
        rp = reference_points.clone()
        rp[...,0] = rp[...,0]*(self.pc_range[3]-self.pc_range[0]) + self.pc_range[0]
        rp[...,1] = rp[...,1]*(self.pc_range[4]-self.pc_range[1]) + self.pc_range[1]
        rp[...,2] = rp[...,2]*(self.pc_range[5]-self.pc_range[2]) + self.pc_range[2]
        ones = torch.ones_like(rp[..., :1])
        pts4 = torch.cat([rp, ones], dim=-1)  # (bs,num_p,num_q,4)

        # 2) batch & cam 차원 확장 후 matmul
        P = lidar2img.unsqueeze(2).unsqueeze(3)    # (bs,N_cam,1,1,4,4)
        pts = pts4.unsqueeze(1).unsqueeze(-1)      # (bs,1,num_p,num_q,4,1)
        cam_pts = (P @ pts).squeeze(-1)            # (bs,N_cam,num_p,num_q,4)

        # 3) 이미지평면으로
        eps = 1e-5
        depth = cam_pts[..., 2:3].clamp(min=eps)
        uv = cam_pts[..., :2] / depth              # (bs,N_cam,num_p,num_q,2)
        mask = depth > eps

        # 4) 정규화
        uv[...,0] /= W_img
        uv[...,1] /= H_img
        valid = (mask &
                 (uv[...,0:1]>0)&(uv[...,0:1]<1)&
                 (uv[...,1:2]>0)&(uv[...,1:2]<1))

        # (N_cam, bs, num_q, num_p, 2)
        reference_points_cam = uv.permute(1,0,3,2,4)
        bev_mask            = valid.squeeze(-1).permute(1,0,3,2)  # (N_cam,bs,num_q,num_p)

        return reference_points_cam, bev_mask

    def forward(self, bev_query, key, value, *args,
                bev_h=None,
                bev_w=None,
                bev_pos=None,
                spatial_shapes=None,
                level_start_index=None,
                valid_ratios=None,
                prev_bev=None,
                shift=0.0,
                lidar2img: torch.Tensor = None,   # (bs, N_cam,4,4)
                img_shape: tuple[int,int]   = None,
                **kwargs):
        """
        Forward 함수는 BEV query와 멀티카메라 feature들을 입력받아 여러 Transformer layer를 통과시킵니다.
        공간 및 시간 어텐션에 필요한 positional encoding, reference points, 그리고 유효 마스크를 계산한 뒤,
        각 layer에 전달하여 최종 BEV representation을 얻습니다.
        """
        assert lidar2img is not None and img_shape is not None
        # print(bev_query.shape)
        output = bev_query
        intermediate = []

        bs = bev_query.size(1)
        device = bev_query.device
        dtype = bev_query.dtype

        # 3D, 2D reference point 생성
        ref_3d = self.get_reference_points(
            bev_h, bev_w, self.pc_range[5] - self.pc_range[2],
            num_points_in_pillar=self.num_points_in_pillar,
            dim='3d', bs=bs, device=device, dtype=dtype)
        ref_2d = self.get_reference_points(
            bev_h, bev_w, dim='2d', bs=bs, device=device, dtype=dtype)

        reference_points_cam, bev_mask = self.point_sampling(ref_3d, lidar2img, img_shape)

        # 원본에서 버그 재현을 위한 코드: shift_ref_2d는 복제 후 shift 적용
        shift_ref_2d = ref_2d.clone()
        if not torch.is_tensor(shift):
            shift = torch.tensor(shift, device=device, dtype=dtype)
        shift_ref_2d += shift

        # (num_query, bs, embed_dims) -> (bs, num_query, embed_dims)
        bev_query = bev_query.permute(1, 0, 2)
        # print(bev_query.shape)
        bev_pos = bev_pos.permute(1, 0, 2)
        bs, len_bev, _ = bev_query.shape
        num_bev_level = ref_2d.shape[2]

        if prev_bev is not None:
            prev_bev = torch.stack(
                [prev_bev, bev_query], 1).reshape(bs*2, len_bev, -1)
            hybird_ref_2d = torch.stack([shift_ref_2d, ref_2d], 1).reshape(bs * 2, len_bev, num_bev_level, 2)
        else:
            hybird_ref_2d = torch.stack([ref_2d, ref_2d], 1).reshape(bs * 2, len_bev, num_bev_level, 2)
        # print(bev_query.shape)
        for layer in self.layers:
            output = layer(
                bev_query,
                key,
                value,
                bev_pos=bev_pos,
                query_pos=bev_pos,
                reference_points=hybird_ref_2d,       # temporal self‑attn용 2D ref
                reference_points_cam=reference_points_cam,  # spatial cross‑attn용 3D ref
                bev_mask=bev_mask,                    # spatial cross‑attn용 mask
                bev_h=bev_h,
                bev_w=bev_w,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                prev_bev=prev_bev,
                **kwargs
            )
            bev_query = output
            if self.return_intermediate:
                intermediate.append(output)
        if self.return_intermediate:
            return torch.stack(intermediate)
        return output 