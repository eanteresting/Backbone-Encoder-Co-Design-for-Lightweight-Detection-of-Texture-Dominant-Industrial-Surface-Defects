import torch
import torch.nn as nn
import torch.nn.functional as F


class AMS_MultiScaleConv(nn.Module):

    def __init__(self, in_channels, out_channels, groups=8):
        super().__init__()

        assert in_channels > 0 and out_channels > 0 and groups > 0
        assert in_channels % groups == 0 and out_channels % groups == 0

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups
        self.num_scales = 3

        self.ams_conv_branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1, groups=groups, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 5, padding=2, groups=in_channels, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 7, padding=6, dilation=2, groups=groups, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        ])

        self.ams_prior_weights = nn.Parameter(torch.ones(self.num_scales))
        self.ams_branch_attn = nn.Conv2d(
            out_channels * self.num_scales,
            self.num_scales,
            1,
            bias=False
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.ams_conv_branches.modules():
            if isinstance(m, nn.Conv2d):
                if m.groups == self.in_channels:
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                else:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        nn.init.constant_(self.ams_branch_attn.weight, 0)

    def forward(self, x):
        ams_scale_feats = [branch(x) for branch in self.ams_conv_branches]

        ams_global_feats = [feat.mean([2, 3], keepdim=True) for feat in ams_scale_feats]
        ams_global_feat = torch.cat(ams_global_feats, dim=1)

        ams_branch_weight = F.softmax(self.ams_branch_attn(ams_global_feat), dim=1)

        ams_prior_weights = self.ams_prior_weights.view(1, -1, 1, 1)
        ams_final_weight = F.softmax(ams_prior_weights * ams_branch_weight, dim=1)

        ams_fused_feat = sum(
            feat * weight.unsqueeze(1)
            for feat, weight in zip(ams_scale_feats, ams_final_weight.unbind(1))
        )
        return ams_fused_feat


class AMS_Stem(nn.Module):

    def __init__(self, input_channels: int = 3, stem_channels: int = 32,
                 reduction_ratio: int = 4, groups: int = 8):
        super().__init__()

        self.stem_channels = stem_channels

        self.ams_init_proj = nn.Sequential(
            nn.Conv2d(input_channels, stem_channels // 2, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(stem_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(stem_channels // 2, stem_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(stem_channels),
            nn.ReLU(inplace=True)
        )

        self.ams_multi_scale = AMS_MultiScaleConv(stem_channels, stem_channels, groups=groups)

        self.ams_base_transform = nn.Sequential(
            nn.Conv2d(stem_channels, stem_channels, 1, bias=False)
        )

        self.ams_channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(stem_channels, stem_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(stem_channels // reduction_ratio, stem_channels, 1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Sigmoid()
        )

        self.ams_spatial_attn = nn.Sequential(
            nn.Conv2d(stem_channels, 1, 3, padding=1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Sigmoid()
        )

        self.ams_cross_channel_fusion = nn.Conv2d(stem_channels, stem_channels, 1, groups=1, bias=False)
        self.ams_bn = nn.BatchNorm2d(stem_channels)

        self._initialize_weights()

    def _initialize_weights(self):
        for name, m in self.named_modules():
            if "ams_multi_scale" in name and isinstance(m, (nn.Conv2d, nn.BatchNorm2d)):
                continue

            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert (x.shape[2] % 4 == 0) and (x.shape[3] % 4 == 0), "输入尺寸H/W必须为4的倍数"

        ams_base_feat = self.ams_init_proj(x)

        ams_base_transformed = self.ams_base_transform(ams_base_feat)

        ams_ms_feat = self.ams_multi_scale(ams_base_feat)

        ams_channel_mask = self.ams_channel_attn(ams_ms_feat)
        ams_spatial_mask = self.ams_spatial_attn(ams_ms_feat)

        ams_refined_feat = ams_ms_feat * (ams_channel_mask + ams_spatial_mask) / 2

        ams_cross_feat = self.ams_cross_channel_fusion(ams_refined_feat)

        ams_output = self.ams_bn(ams_cross_feat + ams_base_transformed)

        return F.relu(ams_output, inplace=True)
