import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn import (build_conv_layer, build_norm_layer, build_plugin_layer,
                      constant_init, kaiming_init)
from mmcv.runner import load_checkpoint
from torch.nn.modules.batchnorm import _BatchNorm

from mmdet.utils import get_root_logger
from ..builder import BACKBONES
from ..utils import ResLayer


__all__ = ['ResNetPf']


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, cfg, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        """
        Args:
            inplanes: (int) num of feature maps in the input.
            planes: (int) num of feature maps in the output.
            cfg: (int) num of filters in the first layer(only prune the first layer).
        """
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, cfg, stride)
        self.bn1 = norm_layer(cfg)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(cfg, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class BottleneckPf(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, cfg, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        """
        Args:
            inplanes: (int) num of feature maps in the input.
            planes: (int) num of feature maps in the output.
            cfg: (int).
        """
        super(BottleneckPf, self).__init__()

        # 默认的初始化(pytorch原始)
        if (cfg is None) | (len(cfg) == 0):
            if norm_layer is None:
                norm_layer = nn.BatchNorm2d
            width = int(planes * (base_width / 64.)) * groups
            self.conv1 = conv1x1(inplanes, width)
            self.bn1 = norm_layer(width)
            self.conv2 = conv3x3(width, width, stride, groups, dilation)
            self.bn2 = norm_layer(width)
            self.conv3 = conv1x1(width, planes * self.expansion)
            self.bn3 = norm_layer(planes * self.expansion)
            self.relu = nn.ReLU(inplace=True)
            self.downsample = downsample
            self.stride = stride
        else:
            assert len(cfg) == 3, "Wrong cfg!"
            if norm_layer is None:
                norm_layer = nn.BatchNorm2d
            self.conv1 = conv1x1(inplanes, cfg[0])
            self.bn1 = norm_layer(cfg[0])
            self.conv2 = conv3x3(cfg[0], cfg[1], stride, groups, dilation)
            self.bn2 = norm_layer(cfg[1])
            self.conv3 = conv1x1(cfg[1], cfg[2])
            self.bn3 = norm_layer(cfg[2])
            self.relu = nn.ReLU(inplace=True)
            self.downsample = downsample
            self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


@BACKBONES.register_module()
class ResNetPf(nn.Module):

    arch_settings = {
        18: (BasicBlock, (2, 2, 2, 2)),
        34: (BasicBlock, (3, 4, 6, 3)),
        50: (BottleneckPf, (3, 4, 6, 3)),
        101: (BottleneckPf, (3, 4, 23, 3)),
        152: (BottleneckPf, (3, 8, 36, 3))
    }

    def __init__(self, depth, pf_cfg=None, groups=1, width_per_group=64,
                 replace_stride_with_dilation=None, norm_layer=None,
                 zero_init_residual=True):
        """
        Args:
            pf_cfg: (list of int) num of filters of the first layer in each block, len(cfg) == num of blocks.
        """
        super(ResNetPf, self).__init__()
        if depth not in self.arch_settings:
            raise KeyError(f'invalid depth {depth} for resnet')
        block, layers = self.arch_settings[depth]

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        if pf_cfg is None:
            # 空的三级列表
            pf_cfg = [[[] for _ in range(layers[i])] for i in range(4)]
        else:
            # cfg: 一级列表 -> 三级列表
            num_layers = 2 if isinstance(block, BasicBlock) else 3
            assert len(pf_cfg) == sum(layers) * num_layers, "Wrong cfg length!"
            cnt = 0
            new_cfg = []
            for num_blocks in layers:
                sub_cfg = []
                for _ in range(num_blocks):
                    sub_sub_cfg = []
                    for _ in range(num_layers):
                        sub_sub_cfg.append(pf_cfg[cnt])
                        cnt += 1
                    sub_cfg.append(sub_sub_cfg)
                new_cfg.append(sub_cfg)
            pf_cfg = new_cfg
            self.cfg = pf_cfg

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], cfg=pf_cfg[0])
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=2, cfg=pf_cfg[1],
            dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(
            block, 256, layers[2], cfg=pf_cfg[2], stride=2,
            dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=2, cfg=pf_cfg[3],
            dilate=replace_stride_with_dilation[2])

        # 为了适配init_weights而增加的变量
        self.dcn = None
        self.zero_init_residual = zero_init_residual

    def _make_layer(self, block, planes, blocks, cfg, stride=1, dilate=False):
        """
        Args:
            block: (class type).
            planes: (int) num of feature maps of the output in this stage.
            blocks: (int) num of block in this stage.
            cfg: (list of list of int) 二级列表.
            len(cfg) == blocks.
            stride: (int).
            dilate: (bool).

        Returns: the created stage
        """
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        # the first block may need a downsampler
        layers.append(
            block(self.inplanes, planes, cfg[0], stride, downsample, self.groups,
                  self.base_width, previous_dilation, norm_layer)
        )
        self.inplanes = planes * block.expansion
        # the rest blocks
        for i in range(1, blocks):
            layers.append(
                block(self.inplanes, planes, cfg[i], groups=self.groups,
                      base_width=self.base_width, dilation=self.dilation,
                      norm_layer=norm_layer)
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        outs = []
        for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
            x = layer(x)
            outs.append(x)
        return tuple(outs)

    def init_weights(self, pretrained=None):
        # copy from torch.resnet
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if self.zero_init_residual:
            for m in self.modules():
                if isinstance(m, BottleneckPf):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)
