import torch
import torch.nn as nn

from ..base_model import BaseModel
from ..modules import (
    HorizontalPoolingPyramid,
    PackSequenceWrapper,
    SeparateBNNecks,
    SeparateFCs,
    SetBlockWrapper,
)

if hasattr(torch, "amp") and hasattr(torch.amp, "custom_fwd"):
    amp_custom_fwd = torch.amp.custom_fwd(device_type="cuda")
    amp_custom_bwd = torch.amp.custom_bwd(device_type="cuda")
else:  # PyTorch 1.x compatibility
    amp_custom_fwd = torch.cuda.amp.custom_fwd
    amp_custom_bwd = torch.cuda.amp.custom_bwd


# Reference for MultiSpike:
# M. Yao et al., "Scaling Spike-Driven Transformer With Efficient Spike
# Firing Approximation Training," IEEE Transactions on Pattern Analysis and
# Machine Intelligence, vol. 47, no. 4, pp. 2973-2990, 2025.
# DOI: https://doi.org/10.1109/TPAMI.2025.3530246
# Official code: https://github.com/BICLab/Spike-Driven-Transformer-V3
class Quant(torch.autograd.Function):
    @staticmethod
    @amp_custom_fwd
    def forward(ctx, i, min_value, max_value):
        ctx.min = min_value
        ctx.max = max_value
        ctx.save_for_backward(i)
        return torch.round(torch.clamp(i, min=min_value, max=max_value))

    @staticmethod
    @amp_custom_bwd
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        (i,) = ctx.saved_tensors
        grad_input[i < ctx.min] = 0
        grad_input[i > ctx.max] = 0
        return grad_input, None, None


class MultiSpike(nn.Module):
    def __init__(self, min_value=0, max_value=4, Norm=None):
        super().__init__()
        self.Norm = max_value if Norm is None else Norm
        self.min_value = min_value
        self.max_value = max_value

    @staticmethod
    def spike_function(x, min_value, max_value):
        return Quant.apply(x, min_value, max_value)

    def __repr__(self):
        return (
            f"MultiSpike(Max_Value={self.max_value}, "
            f"Min_Value={self.min_value}, Norm={self.Norm})"
        )

    def forward(self, x):
        return self.spike_function(
            x, min_value=self.min_value, max_value=self.max_value
        ) / self.Norm


def create_multispike_node(min_value=0, max_value=4, norm=None):
    return MultiSpike(
        min_value=min_value,
        max_value=max_value,
        Norm=norm,
    )


class ConvBN2D(nn.Module):
    def __init__(self, in_c, out_c, k=3, s=1, p=1, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, k, s, p, bias=bias)
        self.bn = nn.BatchNorm2d(out_c)

    def forward(self, x):
        return self.bn(self.conv(x))


class ConvBN3D(nn.Module):
    def __init__(self, in_c, out_c, k=3, s=1, p=1, bias=False):
        super().__init__()
        self.conv = nn.Conv3d(in_c, out_c, k, s, p, bias=bias)
        self.bn = nn.BatchNorm3d(out_c)

    def forward(self, x):
        return self.bn(self.conv(x))


class SpikingStructuralResidualBlock(nn.Module):
    def __init__(
        self,
        in_c,
        out_c,
        stride,
        min_value,
        max_value,
        norm,
    ):
        super().__init__()

        self.conv1 = ConvBN2D(in_c, out_c, 3, stride, 1, bias=False)
        self.spike1 = create_multispike_node(
            min_value=min_value, max_value=max_value, norm=norm
        )
        self.conv2 = ConvBN2D(out_c, out_c, 3, 1, 1, bias=False)
        self.spike2 = create_multispike_node(
            min_value=min_value, max_value=max_value, norm=norm
        )

        if stride != 1 or in_c != out_c:
            self.shortcut = ConvBN2D(in_c, out_c, 1, stride, 0, bias=False)
        else:
            self.shortcut = None

    def forward(self, x):
        residual = x

        out = self.spike1(x)
        out = self.conv1(out)

        out = self.spike2(out)
        out = self.conv2(out)

        if self.shortcut is not None:
            residual = self.shortcut(residual)

        out = out + residual
        return out


class SpikingVolumetricResidualBlock(nn.Module):
    def __init__(
        self,
        in_c,
        out_c,
        stride,
        min_value,
        max_value,
        norm,
    ):
        super().__init__()

        if isinstance(stride, int):
            stride = (1, stride, stride)
        elif len(stride) == 2:
            stride = (1, stride[0], stride[1])

        self.conv1 = ConvBN3D(
            in_c, out_c, k=(3, 3, 3), s=stride, p=(1, 1, 1), bias=False
        )
        self.spike1 = create_multispike_node(
            min_value=min_value, max_value=max_value, norm=norm
        )
        self.conv2 = ConvBN3D(
            out_c, out_c, k=(3, 3, 3), s=(1, 1, 1), p=(1, 1, 1), bias=False
        )
        self.spike2 = create_multispike_node(
            min_value=min_value, max_value=max_value, norm=norm
        )

        if stride != (1, 1, 1) or in_c != out_c:
            self.shortcut = ConvBN3D(in_c, out_c, k=1, s=stride, p=0, bias=False)
        else:
            self.shortcut = None

    def forward(self, x):
        residual = x

        out = self.spike1(x)
        out = self.conv1(out)

        out = self.spike2(out)
        out = self.conv2(out)

        if self.shortcut is not None:
            residual = self.shortcut(residual)

        out = out + residual
        return out


class LightTemporalFusion(nn.Module):
    def __init__(self, channels, spike_cfg, res_scale=0.3):
        super().__init__()
        self.res_scale = res_scale
        self.spike = MultiSpike(**spike_cfg)

        self.temporal_conv = nn.Conv3d(
            channels,
            channels,
            kernel_size=(3, 1, 1),
            stride=(1, 1, 1),
            padding=(1, 0, 0),
            groups=channels,
            bias=False,
        )
        self.bn = nn.BatchNorm3d(channels)

    def forward(self, x):
        residual = x
        out = self.spike(x)
        out = self.temporal_conv(out)
        out = self.bn(out)
        return residual + out * self.res_scale


class Stage1WithLightTemporalFusion(nn.Module):
    def __init__(self, block2d_list, channels, spike_cfg, res_scale=0.3):
        super().__init__()

        self.blocks = nn.ModuleList([SetBlockWrapper(block) for block in block2d_list])

        self.temporal_fusions = nn.ModuleList(
            [
                LightTemporalFusion(
                    channels=channels,
                    spike_cfg=spike_cfg,
                    res_scale=res_scale,
                )
                for _ in block2d_list
            ]
        )

    def forward(self, x):
        for block, fusion in zip(self.blocks, self.temporal_fusions):
            x = block(x)
            x = fusion(x)
        return x


class GlobalSpikingProjector(nn.Module):
    def __init__(self, in_channels, out_channels, min_value, max_value, norm):
        super().__init__()
        self.spike = create_multispike_node(
            min_value=min_value,
            max_value=max_value,
            norm=norm,
        )
        self.project = ConvBN3D(
            in_channels, out_channels, k=1, s=1, p=0, bias=False
        )

    def forward(self, x):
        x = self.spike(x)
        x = self.project(x)
        return x


class SpikingBackbone(nn.Module):
    def __init__(
        self,
        in_channels,
        base_channels,
        min_value,
        max_value,
        norm,
    ):
        super().__init__()

        self.stem = nn.Sequential(
            ConvBN2D(in_channels, base_channels, k=3, s=1, p=1, bias=False),
        )
        self.stem = SetBlockWrapper(self.stem)

        spike_cfg = dict(
            min_value=min_value,
            max_value=max_value,
            Norm=max_value if norm is None else norm,
        )

        block_list = [
            SpikingStructuralResidualBlock(
                base_channels,
                base_channels,
                stride=1,
                min_value=min_value,
                max_value=max_value,
                norm=norm,
            )
        ]

        self.stage1 = nn.Sequential(
            Stage1WithLightTemporalFusion(
                block2d_list=block_list,
                channels=base_channels,
                spike_cfg=spike_cfg,
                res_scale=0.3,
            )
        )

        self.stage2 = nn.Sequential(
            SpikingVolumetricResidualBlock(
                base_channels,
                base_channels * 2,
                stride=(1, 2, 2),
                min_value=min_value,
                max_value=max_value,
                norm=norm,
            )
        )

        self.stage3 = nn.Sequential(
            SpikingVolumetricResidualBlock(
                base_channels * 2,
                base_channels * 4,
                stride=(1, 2, 2),
                min_value=min_value,
                max_value=max_value,
                norm=norm,
            )
        )

        self.gsp = GlobalSpikingProjector(
            in_channels=base_channels * 4,
            out_channels=base_channels * 8,
            min_value=min_value,
            max_value=max_value,
            norm=norm,
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.gsp(x)
        return x


class SpikingGait(BaseModel):
    def build_network(self, model_cfg):
        min_value = model_cfg.get("min_value", 0)
        max_value = model_cfg.get("max_value", 4)
        norm = model_cfg.get("norm", None)
        base_channels = model_cfg.get("base_channels", 128)
        self.Backbone = SpikingBackbone(
            in_channels=1,
            base_channels=base_channels,
            min_value=min_value,
            max_value=max_value,
            norm=norm,
        )

        self.TP = PackSequenceWrapper(torch.max)
        self.HPP = HorizontalPoolingPyramid(bin_num=model_cfg.get("bin_num", [16]))
        self.FCs = SeparateFCs(**model_cfg["SeparateFCs"])
        self.BNNecks = SeparateBNNecks(**model_cfg["SeparateBNNecks"])

    def forward(self, inputs):
        ipts, labs, _, _, seqL = inputs
        sils = ipts[0]

        if len(sils.size()) == 4:
            sils = sils.unsqueeze(1)

        outs = self.Backbone(sils)
        outs = self.TP(outs, seqL, options={"dim": 2})[0]

        feat = self.HPP(outs)
        embed = self.FCs(feat)
        _, logits = self.BNNecks(embed)

        n, _, s, h, w = sils.size()

        retval = {
            "training_feat": {
                "triplet": {"embeddings": embed, "labels": labs},
                "softmax": {"logits": logits, "labels": labs},
            },
            "visual_summary": {
                "image/sils": sils.reshape(n * s, 1, h, w),
            },
            "inference_feat": {"embeddings": embed},
        }
        return retval
