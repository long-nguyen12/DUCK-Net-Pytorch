from torch import nn
import torch
import math
from functools import reduce
from operator import __add__
import torch.nn.functional as F
from models.layers.same_conv2d import Conv2dSame as Conv2dSamePadding


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return Conv2dSamePadding(in_planes, out_planes, kernel_size=1, padding=0)


def conv3x3(
    in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1
) -> nn.Conv2d:
    return Conv2dSamePadding(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        dilation=dilation,
    )


class Conv_Block(nn.Module):
    def __init__(
        self, in_channels, out_channels, block_type, layers=1, kernel=3, padding=1
    ):
        super(Conv_Block, self).__init__()
        self.wide = WideScope_Conv(in_channels, out_channels)
        self.mid = MidScope_Conv(in_channels, out_channels)
        self.res = ResNet_Conv(in_channels, out_channels)
        self.sep = Separated_Conv(in_channels, out_channels, kernel)
        self.duck = Duck_Block(in_channels, out_channels)
        self.double_conv = DoubleConv(in_channels, out_channels)
        self.block_type = block_type
        self.layers = layers

    def forward(self, x):
        result = x
        for i in range(0, self.layers):
            if self.block_type == "separated":
                result = self.sep(result)
            elif self.block_type == "duckv2":
                result = self.duck(result)
            elif self.block_type == "midscope":
                result = self.mid(result)
            elif self.block_type == "widescope":
                result = self.wide(result)
            elif self.block_type == "resnet":
                result = self.res(result)
            elif self.block_type == "double_convolution":
                result = self.double_conv(result)
            else:
                return None

        return result

class ConvBlock2D(nn.Module):
    def __init__(self, filters, block_type, repeat=1):
        super(ConvBlock2D, self).__init__()
        self.repeat = repeat
        self.layers = nn.ModuleList()

        for i in range(repeat):
            if block_type == 'duckv2':
                self.layers.append(Duck_Block(filters))
            elif block_type == 'resnet':
                self.layers.append(ResNet_Conv(filters))
            else:
                raise ValueError(f"Unsupported block_type: {block_type}")

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class Duck_Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Duck_Block, self).__init__()
        self.norm = nn.BatchNorm2d(in_channels)
        self.wide = WideScope_Conv(in_channels, out_channels)
        self.mid = MidScope_Conv(in_channels, out_channels)

        self.res_1 = ResNet_Conv(in_channels, out_channels)

        self.res_2 = ResNet_Conv(in_channels, out_channels)
        self.res_2_1 = ResNet_Conv(out_channels, out_channels)

        self.res_3 = ResNet_Conv(in_channels, out_channels)
        self.res_3_1 = ResNet_Conv(out_channels, out_channels)
        self.res_3_2 = ResNet_Conv(out_channels, out_channels)

        self.sep = Separated_Conv(in_channels, out_channels, 6)
        self.norm_out = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.norm(x)
        x1 = self.wide(x)
        x2 = self.mid(x)
        x3 = self.res_1(x)
        x4 = self.res_2_1(self.res_2(x))
        x5 = self.res_3_2(self.res_3_1(self.res_3(x)))
        x6 = self.sep(x)

        x = x1 + x2 + x3 + x4 + x5 + x6
        x = self.norm_out(x)
        return x


class Separated_Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel=3):
        super(Separated_Conv, self).__init__()
        self.conv1n = Conv2dSamePadding(
            in_channels, out_channels, kernel_size=(1, kernel), stride=1
        )
        self.convn1 = Conv2dSamePadding(
            out_channels,
            out_channels,
            kernel_size=(kernel, 1),
            stride=1,
        )
        self.act = nn.ReLU(inplace=True)
        self.norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.norm(self.act(self.conv1n(x)))
        x = self.norm(self.act(self.convn1(x)))

        return x


class MidScope_Conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MidScope_Conv, self).__init__()
        self.conv33_1 = Conv2dSamePadding(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            dilation=1,
        )
        self.conv33_2 = Conv2dSamePadding(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            dilation=2,
        )
        self.act = nn.ReLU(inplace=True)
        self.norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.norm(self.act(self.conv33_1(x)))
        x = self.norm(self.act(self.conv33_2(x)))

        return x


class WideScope_Conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(WideScope_Conv, self).__init__()
        self.conv33_1 = Conv2dSamePadding(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            dilation=1,
        )
        self.conv33_2 = Conv2dSamePadding(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            dilation=2,
        )
        self.conv33_3 = Conv2dSamePadding(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            dilation=3,
        )
        self.act = nn.ReLU(inplace=True)
        self.norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.norm(self.act(self.conv33_1(x)))
        x = self.norm(self.act(self.conv33_2(x)))
        x = self.norm(self.act(self.conv33_3(x)))

        return x


class ResNet_Conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResNet_Conv, self).__init__()
        self.conv11 = conv1x1(in_channels, out_channels)
        self.conv33_1 = conv3x3(in_channels, out_channels)
        self.conv33_2 = conv3x3(out_channels, out_channels)
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        skip = self.act(self.conv11(x))

        x = self.norm(self.act(self.conv33_1(x)))
        x = self.norm(self.act(self.conv33_2(x)))

        return self.norm(x + skip)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            Conv2dSamePadding(in_channels, out_channels, kernel_size=3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            Conv2dSamePadding(out_channels, out_channels, kernel_size=3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)