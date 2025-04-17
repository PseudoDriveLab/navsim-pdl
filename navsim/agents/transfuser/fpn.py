import torch
import torch.nn as nn
import torch.nn.functional as F

class FPN(nn.Module):
    def __init__(self,
                 in_channels,      # 예: [512, 1024, 2048]
                 out_channels,     # 예: _dim_ = 256
                 num_outs,         # 예: _num_mono_levels_ = 5
                 start_level=0,
                 add_extra_convs='on_output',
                 relu_before_extra_convs=True):
        super(FPN, self).__init__()
        assert isinstance(in_channels, (list, tuple))
        self.in_channels = in_channels[start_level:]
        self.out_channels = out_channels
        self.num_outs = num_outs
        self.start_level = start_level
        self.add_extra_convs = add_extra_convs
        self.relu_before_extra_convs = relu_before_extra_convs

        # lateral_conv: 1x1 conv to reduce channel dimensions
        self.lateral_convs = nn.ModuleList()
        # fpn_conv: 3x3 conv to smooth output feature maps
        self.fpn_convs = nn.ModuleList()
        for in_channels in self.in_channels:
            lateral_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
            fpn_conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            self.lateral_convs.append(lateral_conv)
            self.fpn_convs.append(fpn_conv)

        # extra convolutions for additional levels if needed
        extra_levels = num_outs - len(self.in_channels)
        if extra_levels > 0:
            self.extra_convs = nn.ModuleList()
            for i in range(extra_levels):
                in_channels = out_channels
                extra_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
                self.extra_convs.append(extra_conv)
        else:
            self.extra_convs = None

    def forward(self, inputs):
        # inputs: list of feature maps from backbone (예: level 1,2,3)
        # lateral features
        lateral_feats = [l_conv(fea) for l_conv, fea in zip(self.lateral_convs, inputs)]
        # build top-down pathway
        num_levels = len(lateral_feats)
        for i in range(num_levels - 1, 0, -1):
            lateral_feats[i - 1] += F.interpolate(lateral_feats[i],
                                                   size=lateral_feats[i - 1].shape[-2:],
                                                   mode='nearest')
        # smooth with 3x3 conv
        outs = [fpn_conv(lateral_feats[i]) for i, fpn_conv in enumerate(self.fpn_convs)]
        # extra levels
        if self.extra_convs is not None:
            last_feat = outs[-1]
            for extra_conv in self.extra_convs:
                if self.relu_before_extra_convs:
                    last_feat = F.relu(last_feat)
                last_feat = extra_conv(last_feat)
                outs.append(last_feat)
        return outs