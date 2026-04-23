import torch
import torch.nn.functional as F
from torch import nn
import math

try:
    import pywt
except ImportError:
    raise ImportError("PMSWaveletConv requires 'PyWavelets'. Please install it via 'pip install PyWavelets'.")

__all__ = ['PMSWaveletConv']


def create_wavelet_filter(wave, in_size, out_size, dtype=torch.float):
    w = pywt.Wavelet(wave)

    dec_hi = torch.tensor(w.dec_hi[::-1], dtype=dtype)
    dec_lo = torch.tensor(w.dec_lo[::-1], dtype=dtype)

    dec_filters = torch.stack([
        dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1),
        dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),
        dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),
        dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1),
    ], dim=0)
    dec_filters = dec_filters[:, None].repeat(in_size, 1, 1, 1)

    rec_hi = torch.tensor(w.rec_hi[::-1], dtype=dtype)
    rec_lo = torch.tensor(w.rec_lo[::-1], dtype=dtype)

    rec_filters = torch.stack([
        rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1),
        rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1),
        rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1),
        rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1),
    ], dim=0)
    rec_filters = rec_filters[:, None].repeat(in_size, 1, 1, 1)

    return dec_filters, rec_filters


def wavelet_transform(x, filters):
    b, c, h, w = x.shape
    pad_h = (filters.shape[2] - 1) // 2
    pad_w = (filters.shape[3] - 1) // 2
    x = F.conv2d(x, filters, stride=2, groups=c, padding=(pad_h, pad_w))
    x = x.reshape(b, c, 4, h // 2, w // 2)
    return x


def inverse_wavelet_transform(x, filters, output_size=None):
    b, c4, h_half, w_half = x.shape
    c = c4 // 4
    pad_h = (filters.shape[2] - 1) // 2
    pad_w = (filters.shape[3] - 1) // 2
    stride = 2

    if output_size is not None:
        expected_h = output_size[0]
        expected_w = output_size[1]
        output_padding_h = expected_h - (h_half - 1) * stride - filters.shape[2] + 2 * pad_h
        output_padding_w = expected_w - (w_half - 1) * stride - filters.shape[3] + 2 * pad_w
        output_padding_h = max(0, min(output_padding_h, stride - 1))
        output_padding_w = max(0, min(output_padding_w, stride - 1))
        output_padding = (output_padding_h, output_padding_w)
    else:
        output_padding = (0, 0)

    x = F.conv_transpose2d(x, filters, stride=stride, groups=c,
                           padding=(pad_h, pad_w), output_padding=output_padding)
    return x


class ImprovedFusion(nn.Module):

    def __init__(self, channels, levels):
        super(ImprovedFusion, self).__init__()
        self.fusion_weights = nn.Parameter(torch.ones(levels + 1))

    def forward(self, base_feat, wavelet_feats):
        normalized_weights = F.softmax(self.fusion_weights, dim=0)

        weighted_sum = normalized_weights[0] * base_feat

        for i, feat in enumerate(wavelet_feats):
            if feat.shape[2:] != base_feat.shape[2:]:
                feat = F.interpolate(feat, size=base_feat.shape[2:], mode='nearest')
            weighted_sum += normalized_weights[i + 1] * feat

        return weighted_sum


class PMSWaveletConv2d(nn.Module):

    def __init__(self, in_channels, kernel_size=3, stride=1,
                 bias=False, wt_levels=3):
        super(PMSWaveletConv2d, self).__init__()
        self.in_channels = in_channels
        self.wt_levels = wt_levels
        self.stride = stride
        self.min_input_size = 2 ** wt_levels

        self.pms_wt_filters, self.pms_iwt_filters = self._create_pms_multilevel_filters(in_channels)

        self.pms_base_conv = nn.Conv2d(in_channels, in_channels, kernel_size,
                                       padding='same', stride=1, dilation=1,
                                       groups=in_channels, bias=bias)
        self.pms_base_scale = _PMSScaleModule([1, in_channels, 1, 1])

        self.pms_wavelet_convs = nn.ModuleList([
            nn.Conv2d(in_channels * 4, in_channels * 4, kernel_size,
                      padding='same', stride=1, dilation=1,
                      groups=in_channels * 4, bias=False)
            for _ in range(self.wt_levels)
        ])
        self.pms_wavelet_scales = nn.ModuleList([
            _PMSScaleModule([1, in_channels * 4, 1, 1], init_scale=0.9)
            for _ in range(self.wt_levels)
        ])

        self.fusion_module = ImprovedFusion(in_channels, wt_levels)

        if self.stride > 1:
            self.downsample = nn.AvgPool2d(kernel_size=stride, stride=stride)
        else:
            self.downsample = None

    def _create_pms_multilevel_filters(self, channels):
        pms_wt_filters = []
        pms_iwt_filters = []

        wavelet_candidates = ['haar', 'db2', 'db4', 'sym4', 'coif2']

        for level in range(self.wt_levels):
            current_wt_name = wavelet_candidates[min(level, len(wavelet_candidates) - 1)]
            print(f"  > Level {level}: Using wavelet basis '{current_wt_name}'")

            wt_filter, iwt_filter = create_wavelet_filter(current_wt_name, channels, channels, torch.float)
            pms_wt_filters.append(nn.Parameter(wt_filter, requires_grad=False))
            pms_iwt_filters.append(nn.Parameter(iwt_filter, requires_grad=False))

        return nn.ParameterList(pms_wt_filters), nn.ParameterList(pms_iwt_filters)

    def forward(self, x):
        if x.shape[2] < self.min_input_size or x.shape[3] < self.min_input_size:
            raise ValueError(
                f"Input size too small. Minimum required size is {self.min_input_size}x{self.min_input_size}")

        pyramid_features = self._build_wavelet_pyramid(x)

        processed_pyramid = self._process_pyramid_features(pyramid_features)

        reconstructed_features = self._reconstruct_and_fuse(processed_pyramid, x.shape[2:])

        base_output = self.pms_base_scale(self.pms_base_conv(x))

        output = self.fusion_module(base_output, reconstructed_features)

        if self.downsample is not None:
            output = self.downsample(output)
        return output

    def _build_wavelet_pyramid(self, x):
        pyramid = []
        current = x
        for level in range(self.wt_levels):
            original_shape = current.shape

            if (current.shape[2] % 2 > 0) or (current.shape[3] % 2 > 0):
                pad_h = current.shape[2] % 2
                pad_w = current.shape[3] % 2
                pad_left = pad_w // 2
                pad_right = pad_w - pad_left
                pad_top = pad_h // 2
                pad_bottom = pad_h - pad_top
                current = F.pad(current, (pad_left, pad_right, pad_top, pad_bottom), mode='reflect')

            padded_shape = current.shape

            coeffs = wavelet_transform(current, self.pms_wt_filters[level])

            pyramid.append((coeffs, padded_shape, original_shape))
            current = coeffs[:, :, 0, :, :]
        return pyramid

    def _process_pyramid_features(self, pyramid):
        processed_features = []
        for level, (coeffs, padded_shape, original_shape) in enumerate(pyramid):
            b, c, _, h, w = coeffs.shape
            reshaped = coeffs.reshape(b, c * 4, h, w)

            conv_out = self.pms_wavelet_convs[level](reshaped)
            scaled_out = self.pms_wavelet_scales[level](conv_out)

            restored = scaled_out.reshape(b, c, 4, h, w)
            processed_features.append((restored, padded_shape, original_shape))
        return processed_features

    def _reconstruct_and_fuse(self, processed_features, original_input_size):
        if not processed_features:
            return []

        reconstructed_features = []
        reconstructed = None

        for level in reversed(range(len(processed_features))):
            feature, padded_shape, original_shape = processed_features[level]

            if reconstructed is not None:
                target_size = feature[:, :, 0, :, :].shape[2:]
                if reconstructed.shape[2:] != target_size:
                    reconstructed = F.interpolate(reconstructed,
                                                  size=target_size,
                                                  mode='nearest')
                ll_component = feature[:, :, 0, :, :] + reconstructed
                feature[:, :, 0, :, :] = ll_component

            b, c, _, h, w = feature.shape
            feature_reshaped = feature.reshape(b, c * 4, h, w)

            current_recon = inverse_wavelet_transform(
                feature_reshaped,
                self.pms_iwt_filters[level],
                output_size=padded_shape[2:]
            )

            h_crop = min(current_recon.shape[2], original_shape[2])
            w_crop = min(current_recon.shape[3], original_shape[3])
            reconstructed = current_recon[:, :, :h_crop, :w_crop]

            reconstructed_features.insert(0, reconstructed)

        return reconstructed_features


class _PMSScaleModule(nn.Module):

    def __init__(self, dims, init_scale=1.0):
        super(_PMSScaleModule, self).__init__()
        self.pms_weight = nn.Parameter(torch.ones(*dims) * init_scale)

    def forward(self, x):
        return torch.mul(self.pms_weight, x)


class PMSWaveletConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(PMSWaveletConv, self).__init__()
        self.pms_depthwise = PMSWaveletConv2d(in_channels, kernel_size=kernel_size)
        self.pms_pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1,
                                       stride=1, padding=0, bias=False)

    def forward(self, x):
        x = self.pms_depthwise(x)
        x = self.pms_pointwise(x)
        return x
