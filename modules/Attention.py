import torch
import torch.nn as nn

__all__ = ['BlockShuffleAttn']

class BlockShuffleAttn(nn.Module):
    def __init__(self, in_features, group=4, tau=0.4):
        super().__init__()
        self.block_group = group
        self.in_features = in_features
        self.tau = float(tau)

        assert group > 0
        assert in_features % self.block_group == 0
        assert in_features >= 8
        groups_for_refine = in_features // 4
        assert in_features % groups_for_refine == 0

        self.group_evaluator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_features, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

        self.block_norm = nn.GroupNorm(num_groups=self.block_group, num_channels=in_features)
        self.block_conv = nn.Conv2d(in_features, in_features, kernel_size=1, groups=self.block_group)
        self.block_act = nn.GELU()

        self.attention_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_features, in_features // 2, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(in_features // 2, in_features, groups=self.block_group, kernel_size=1),
            nn.Sigmoid()
        )

        self.feature_refine = nn.Sequential(
            nn.Conv2d(in_features, in_features, kernel_size=3, padding=1, groups=in_features // 4),
            nn.BatchNorm2d(in_features),
            nn.GELU()
        )

    def block_shuffle(self, x):
        b, c, h, w = x.shape
        gc = c // self.block_group
        x = x.reshape(b, gc, self.block_group, h, w).permute(0, 2, 1, 3, 4)
        return x.reshape(b, c, h, w)

    def block_rearrange(self, x):
        b, c, h, w = x.shape
        gc = c // self.block_group
        x = x.reshape(b, self.block_group, gc, h, w).permute(0, 2, 1, 3, 4)
        return x.reshape(b, c, h, w)

    def forward(self, x):
        residual = x

        alpha = self.group_evaluator(x).view(x.size(0), 1, 1, 1)

        x = self.block_shuffle(x)
        x = self.block_act(self.block_conv(self.block_norm(x)))

        x = x * self.attention_gate(x)

        x = self.block_rearrange(x)

        if self.training:
            refined_x = self.feature_refine(x)
            x_mix = x * (1 - alpha) + refined_x * alpha
        else:
            refine_mask = (alpha > self.tau)

            if not bool(refine_mask.any().item()):
                x_mix = x
            else:
                refined_x = self.feature_refine(x)
                x_mix = torch.where(refine_mask, refined_x, x)

        out = residual * (1 - alpha) + x_mix * alpha
        return out
