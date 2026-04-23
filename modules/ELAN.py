import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

__all__ = ['C2f_MSAM', 'C3k2_MSAM']


def autopad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Conv(nn.Module):
    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Shift_channel_mix(nn.Module):
    def __init__(self, shifts=[1, 2]):
        super(Shift_channel_mix, self).__init__()
        self.shifts = shifts

    def forward(self, x):
        B, C, H, W = x.shape
        if C < 8 or C % 8 != 0:
            return x

        channel_groups = torch.chunk(x, 8, dim=1)
        shifted_groups = []

        shifted_groups.append(torch.roll(channel_groups[0], self.shifts[0], dims=2))
        shifted_groups.append(torch.roll(channel_groups[1], -self.shifts[0], dims=2))
        shifted_groups.append(torch.roll(channel_groups[2], self.shifts[0], dims=3))
        shifted_groups.append(torch.roll(channel_groups[3], -self.shifts[0], dims=3))
        shifted_groups.append(torch.roll(channel_groups[4], self.shifts[1], dims=2))
        shifted_groups.append(torch.roll(channel_groups[5], -self.shifts[1], dims=2))
        shifted_groups.append(torch.roll(channel_groups[6], self.shifts[1], dims=3))
        shifted_groups.append(torch.roll(channel_groups[7], -self.shifts[1], dims=3))

        x = torch.cat(shifted_groups, 1)
        return x


class MSAM(nn.Module):
    def __init__(self, in_channels, att_channels=16, lk_size=9, sk_size=5, reduction=2):
        super().__init__()
        self.att_channels = min(att_channels, in_channels // 2) if in_channels > 1 else in_channels
        self.idt_channels = in_channels - self.att_channels

        self.lk_size = lk_size
        self.sk_size = sk_size

        self.kernel_gen = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.att_channels, self.att_channels // reduction, 1),
            nn.GELU(),
            nn.Conv2d(self.att_channels // reduction, self.att_channels * sk_size * sk_size, 1)
        )

        for m in self.kernel_gen.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None: nn.init.zeros_(m.bias)
        nn.init.zeros_(self.kernel_gen[-1].weight)
        nn.init.zeros_(self.kernel_gen[-1].bias)

        self.lk_conv = nn.Conv2d(self.att_channels, self.att_channels,
                                 kernel_size=lk_size, padding=lk_size // 2, bias=False)
        nn.init.kaiming_normal_(self.lk_conv.weight, mode='fan_out', nonlinearity='relu')

        self.fusion = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.scm = Shift_channel_mix()

    def forward(self, x):
        B, C, H, W = x.shape
        F_att, F_idt = torch.split(x, [self.att_channels, self.idt_channels], dim=1)

        kernel_weight = self.kernel_gen(F_att)
        kernel = kernel_weight.reshape(B * self.att_channels, 1, self.sk_size, self.sk_size)
        F_att_re = rearrange(F_att, 'b c h w -> 1 (b c) h w')
        out_dk = F.conv2d(F_att_re, kernel, padding=self.sk_size // 2, groups=B * self.att_channels)
        out_dk = rearrange(out_dk, '1 (b c) h w -> b c h w', b=B, c=self.att_channels)

        out_lk = self.lk_conv(F_att)

        out_att = out_lk + out_dk
        out = torch.cat([out_att, F_idt], dim=1)
        out = self.scm(out)
        out = self.fusion(out)
        return out


class Bottleneck_MSAM(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2
        self.Attention = MSAM(c2)

    def forward(self, x):
        return x + self.Attention(self.cv2(self.cv1(x))) if self.add else self.Attention(self.cv2(self.cv1(x)))


class C3k_MSAM(nn.Module):

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)

        kernel_tuple = ((k, k), (k, k))
        self.m = nn.Sequential(*(Bottleneck_MSAM(c_, c_, shortcut, g, k=kernel_tuple, e=1.0) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C2f_MSAM(nn.Module):

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            Bottleneck_MSAM(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3k2_MSAM(C2f_MSAM):

    def __init__(self, c1, c2, n=1, c3k=True, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k_MSAM(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck_MSAM(self.c, self.c, shortcut, g,
                                                                                 k=((3, 3), (3, 3)), e=1.0) for _ in
            range(n)
        )
