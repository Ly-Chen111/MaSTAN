from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
import fvcore.nn.weight_init as weight_init


def get_norm(norm, out_channels): # only support GN or LN
    """
    Args:
        norm (str or callable): either one of BN, SyncBN, FrozenBN, GN;
            or a callable that takes a channel number and returns
            the normalization layer as a nn.Module.

    Returns:
        nn.Module or None: the normalization layer
    """
    if norm is None:
        return None
    if isinstance(norm, str):
        if len(norm) == 0:
            return None
        norm = {
            "GN": lambda channels: nn.GroupNorm(8, channels),
            "LN": lambda channels: nn.LayerNorm(channels)
        }[norm]
    return norm(out_channels)

class Conv2d(torch.nn.Conv2d):
    """
    A wrapper around :class:`torch.nn.Conv2d` to support empty inputs and more features.
    """

    def __init__(self, *args, **kwargs):
        """
        Extra keyword arguments supported in addition to those in `torch.nn.Conv2d`:

        Args:
            norm (nn.Module, optional): a normalization layer
            activation (callable(Tensor) -> Tensor): a callable activation function

        It assumes that norm layer is used before activation.
        """
        norm = kwargs.pop("norm", None)
        activation = kwargs.pop("activation", None)
        super().__init__(*args, **kwargs)

        self.norm = norm
        self.activation = activation

    def forward(self, x):
        # torchscript does not support SyncBatchNorm yet
        # https://github.com/pytorch/pytorch/issues/40507
        # and we skip these codes in torchscript since:
        # 1. currently we only support torchscript in evaluation mode
        # 2. features needed by exporting module to torchscript are added in PyTorch 1.6 or
        # later version, `Conv2d` in these PyTorch versions has already supported empty inputs.
        if not torch.jit.is_scripting():
            if x.numel() == 0 and self.training:
                # https://github.com/pytorch/pytorch/issues/12013
                assert not isinstance(
                    self.norm, torch.nn.SyncBatchNorm
                ), "SyncBatchNorm does not support empty inputs!"

        x = F.conv2d(
            x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x

# FPN structure
class DGDecoder(nn.Module):
    def __init__(self, feature_channels: List, conv_dim: int,  norm=None):
        """
        Args:
            feature_channels: list of fpn feature channel numbers.
            conv_dim: number of output channels for the intermediate conv layers.
            mask_dim: number of output channels for the final conv layer.
            norm (str or callable): normalization for all conv layers
        """
        super().__init__()

        self.feature_channels = feature_channels

        lateral_convs_normal = []
        lateral_convs_occ = []
        output_convs_normal = []
        output_convs_occ = []
        upsample_convs_normal = []
        upsample_convs_occ = []

        use_bias = norm == ""
        for idx, in_channels in enumerate(feature_channels):
            # in_channels: 4x -> 32x
            lateral_norm = get_norm(norm, conv_dim)
            output_norm = get_norm(norm, conv_dim)

            lateral_conv_normal = Conv2d(
                in_channels, conv_dim, kernel_size=1, bias=use_bias, norm=lateral_norm
            )
            lateral_conv_occ = Conv2d(
                in_channels, conv_dim, kernel_size=1, bias=use_bias, norm=lateral_norm
            )
            output_conv_normal = Conv2d(
                conv_dim,
                conv_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=use_bias,
                norm=output_norm,
                activation=F.relu,
            )
            output_conv_occ = Conv2d(
                conv_dim,
                conv_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=use_bias,
                norm=output_norm,
                activation=F.relu,
            )
            upsample_conv_normal = nn.Sequential(
                nn.ConvTranspose2d(conv_dim, conv_dim, 4, stride=2, padding=1, bias=False),
                nn.GroupNorm(8, 256, eps=1e-05, affine=False), nn.ReLU(inplace=True))
            upsample_conv_occ = nn.Sequential(
                nn.ConvTranspose2d(conv_dim, conv_dim, 4, stride=2, padding=1, bias=False),
                nn.GroupNorm(8, 256, eps=1e-05, affine=False), nn.ReLU(inplace=True))
            weight_init.c2_xavier_fill(upsample_conv_normal[0])
            weight_init.c2_xavier_fill(upsample_conv_occ[0])
            stage = idx + 1
            self.add_module("normal_adapter_{}".format(stage), lateral_conv_normal)
            self.add_module("occ_adapter_{}".format(stage), lateral_conv_occ)
            self.add_module("normal_layer_{}".format(stage), output_conv_normal)
            self.add_module("occ_layer_{}".format(stage), output_conv_occ)
            if stage > 1:
                self.add_module('normal_upsample_{}'.format(stage), upsample_conv_normal)
                self.add_module('occ_upsample_{}'.format(stage), upsample_conv_occ)
                upsample_convs_normal.append(upsample_conv_normal)
                upsample_convs_occ.append(upsample_conv_occ)

            lateral_convs_normal.append(lateral_conv_normal)
            lateral_convs_occ.append(lateral_conv_occ)
            output_convs_normal.append(output_conv_normal)
            output_convs_occ.append(output_conv_occ)

        # Place convs into top-down order (from low to high resolution)
        # to make the top-down computation in forward clearer.
        self.lateral_convs_normal = lateral_convs_normal[::-1]
        self.lateral_convs_occ = lateral_convs_occ[::-1]
        self.output_convs_normal = output_convs_normal[::-1]
        self.output_convs_occ = output_convs_occ[::-1]
        self.upsample_convs_normal = nn.ModuleList(upsample_convs_normal[::-1])
        self.upsample_convs_occ = nn.ModuleList(upsample_convs_occ[::-1])

        self.output_upsample_normal = nn.Sequential(
            nn.ConvTranspose2d(conv_dim, conv_dim, 8, stride=4, padding=2, bias=False),
            nn.GroupNorm(8, 256, eps=1e-05, affine=False), nn.ReLU(inplace=True))
        self.output_upsample_occ = nn.Sequential(
            nn.ConvTranspose2d(conv_dim, conv_dim, 8, stride=4, padding=2, bias=False),
            nn.GroupNorm(8, 256, eps=1e-05, affine=False), nn.ReLU(inplace=True))
        weight_init.c2_xavier_fill(self.output_upsample_normal[0])
        weight_init.c2_xavier_fill(self.output_upsample_occ[0])

        self.mask_features_normal = Conv2d(
            conv_dim,
            1,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.mask_features_occ = Conv2d(
            conv_dim,
            1,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        weight_init.c2_xavier_fill(self.mask_features_normal)
        weight_init.c2_xavier_fill(self.mask_features_occ)

    def forward_features(self, memory, features_img, features_flow):
        for idx in range(len(self.feature_channels) - 1):  # 32x -> 8x
            lateral_conv_normal = self.lateral_convs_normal[idx]
            lateral_conv_occ = self.lateral_convs_occ[idx]
            output_conv_normal = self.output_convs_normal[idx]
            output_conv_occ = self.output_convs_occ[idx]

            # NOTE: here the (h, w) is the size for current fpn layer
            cur_fpn_normal = lateral_conv_normal(memory[idx])
            cur_fpn_occ = lateral_conv_occ(memory[idx])

            if idx == 0:  # top layer
                normal_ln = output_conv_normal(cur_fpn_normal)
                occ_ln = output_conv_occ(cur_fpn_occ)
            else:
                upsample_conv_normal = self.upsample_convs_normal[idx - 1]
                upsample_conv_occ = self.upsample_convs_occ[idx - 1]
                gate_normal_feature_up = upsample_conv_normal((1-torch.tanh(occ_ln)) * normal_ln)
                gate_occ_feature_up = upsample_conv_occ((1-torch.tanh(normal_ln)) * occ_ln)
                normal_ln = output_conv_normal(cur_fpn_normal + gate_normal_feature_up)
                occ_ln = output_conv_occ(cur_fpn_occ + gate_occ_feature_up)

        lateral_conv_normal = self.lateral_convs_normal[-1]
        lateral_conv_occ = self.lateral_convs_occ[-1]
        output_conv_normal = self.output_convs_normal[-1]
        output_conv_occ = self.output_convs_occ[-1]
        upsample_conv_normal = self.upsample_convs_normal[-1]
        upsample_conv_occ = self.upsample_convs_occ[-1]

        x_img, x_mask = features_img[0].decompose()
        cur_fpn_normal = lateral_conv_normal(x_img)

        x_flow, x_mask = features_flow[0].decompose()
        cur_fpn_occ = lateral_conv_occ(x_flow)

        gate_normal_feature_up = upsample_conv_normal((1 - torch.tanh(occ_ln)) * normal_ln)
        gate_occ_feature_up = upsample_conv_occ((1 - torch.tanh(normal_ln)) * occ_ln)

        gate_normal_feature_up = output_conv_normal(gate_normal_feature_up + cur_fpn_normal)
        gate_occ_feature_up = output_conv_occ(gate_occ_feature_up + cur_fpn_occ)

        normal_edge = self.output_upsample_normal(gate_normal_feature_up)

        occ_edge = self.output_upsample_occ(gate_occ_feature_up)

        return normal_edge, occ_edge  # [b*t, c, h, w], the spatial stride is 4x, now need to meet ori shape

    def forward(self, memory, features_img, features_flow):
        normal_edge, occ_edge = self.forward_features(memory, features_img, features_flow)
        return self.mask_features_normal(normal_edge), self.mask_features_occ(occ_edge)


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")