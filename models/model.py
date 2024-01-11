from torch import nn
from models.layers.blocks import *
import torch.nn.functional as F
import torch
import math
import warnings

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        tensor.uniform_(2 * l - 1, 2 * u - 1)

        tensor.erfinv_()

        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        tensor.clamp_(min=a, max=b)
        return tensor

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


class DUCK_Net(nn.Module):
    def __init__(self, in_channels):
        super(DUCK_Net, self).__init__()

        # self.conv1 = nn.Conv2d(
        #     3, in_channels * 2, kernel_size=2, stride=2, bias=False, padding="valid"
        # )
        # self.conv2 = nn.Conv2d(
        #     in_channels * 2,
        #     in_channels * 4,
        #     kernel_size=2,
        #     stride=2,
        #     bias=False,
        #     padding="valid",
        # )
        # self.conv3 = nn.Conv2d(
        #     in_channels * 4,
        #     in_channels * 8,
        #     kernel_size=2,
        #     stride=2,
        #     bias=False,
        #     padding="valid",
        # )
        # self.conv4 = nn.Conv2d(
        #     in_channels * 8,
        #     in_channels * 16,
        #     kernel_size=2,
        #     stride=2,
        #     bias=False,
        #     padding="valid",
        # )
        # self.conv5 = nn.Conv2d(
        #     in_channels * 16,
        #     in_channels * 32,
        #     kernel_size=2,
        #     stride=2,
        #     bias=False,
        # )
        self.conv1 = Conv2dSamePadding(3, in_channels * 2, kernel_size=2, stride=2)
        self.conv2 = Conv2dSamePadding(in_channels * 2, in_channels * 4, kernel_size=2, stride=2)
        self.conv3 = Conv2dSamePadding(in_channels * 4, in_channels * 8, kernel_size=2, stride=2)
        self.conv4 = Conv2dSamePadding(in_channels * 8, in_channels * 16, kernel_size=2, stride=2)
        self.conv5 = Conv2dSamePadding(in_channels * 16, in_channels * 32, kernel_size=2, stride=2)

        self.t0 = Conv_Block(3, in_channels, "duckv2", layers=1)

        self.l1i = Conv2dSamePadding(in_channels, in_channels * 2, kernel_size=2, stride=2)
        self.l2i = Conv2dSamePadding(in_channels * 2, in_channels * 4, kernel_size=2, stride=2)
        self.l3i = Conv2dSamePadding(in_channels * 4, in_channels * 8, kernel_size=2, stride=2)
        self.l4i = Conv2dSamePadding(in_channels * 8, in_channels * 16, kernel_size=2, stride=2)
        self.l5i = Conv2dSamePadding(in_channels * 16, in_channels * 32, kernel_size=2, stride=2)

        # self.l1i = nn.Conv2d(
        #     in_channels, in_channels * 2, kernel_size=2, stride=2, padding="valid"
        # )
        # self.l2i = nn.Conv2d(
        #     in_channels * 2, in_channels * 4, kernel_size=2, stride=2, padding="valid"
        # )
        # self.l3i = nn.Conv2d(
        #     in_channels * 4, in_channels * 8, kernel_size=2, stride=2, padding="valid"
        # )
        # self.l4i = nn.Conv2d(
        #     in_channels * 8, in_channels * 16, kernel_size=2, stride=2, padding="valid"
        # )
        # self.l5i = nn.Conv2d(
        #     in_channels * 16, in_channels * 32, kernel_size=2, stride=2
        # )

        self.t1 = Conv_Block(in_channels * 2, in_channels * 2, "duckv2", layers=1)
        self.t2 = Conv_Block(in_channels * 4, in_channels * 4, "duckv2", layers=1)
        self.t3 = Conv_Block(in_channels * 8, in_channels * 8, "duckv2", layers=1)
        self.t4 = Conv_Block(in_channels * 16, in_channels * 16, "duckv2", layers=1)
        self.t5_1 = Conv_Block(in_channels * 32, in_channels * 32, "resnet", layers=2)
        self.t5_3 = Conv_Block(in_channels * 32, in_channels * 16, "resnet", layers=1)
        self.t5_2 = Conv_Block(in_channels * 16, in_channels * 16, "resnet", layers=1)

        self.q4 = Conv_Block(in_channels * 16, in_channels * 8, "duckv2", layers=1)
        self.q3 = Conv_Block(in_channels * 8, in_channels * 4, "duckv2", layers=1)
        self.q2 = Conv_Block(in_channels * 4, in_channels * 2, "duckv2", layers=1)
        self.q1 = Conv_Block(in_channels * 2, in_channels, "duckv2", layers=1)
        self.z1 = Conv_Block(in_channels, in_channels, "duckv2", layers=1)

        self.out = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.upsample = nn.UpsamplingNearest2d(scale_factor=2)

        # self.apply(self._init_weights)

    def forward(self, x):
        p1 = self.conv1(x)
        p2 = self.conv2(p1)
        p3 = self.conv3(p2)
        p4 = self.conv4(p3)
        p5 = self.conv5(p4)

        t_0 = self.t0(x)

        l1_i = self.l1i(t_0)
        s_1 = p1 + l1_i
        t_1 = self.t1(s_1)

        l2_i = self.l2i(t_1)
        s_2 = p2 + l2_i
        t_2 = self.t2(s_2)

        l3_i = self.l3i(t_2)
        s_3 = p3 + l3_i
        t_3 = self.t3(s_3)

        l4_i = self.l4i(t_3)
        s_4 = p4 + l4_i
        t_4 = self.t4(s_4)

        l5_i = self.l5i(t_4)
        s_5 = p5 + l5_i
        t_51 = self.t5_1(s_5)
        t_53 = self.t5_3(t_51)
        t_52 = self.t5_2(t_53)

        l5_o = self.upsample(t_52)
        c4 = l5_o + t_4
        q_4 = self.q4(c4)

        l4_o = self.upsample(q_4)
        c3 = l4_o + t_3
        q_3 = self.q3(c3)

        l3_o = self.upsample(q_3)
        c2 = l3_o + t_2
        q_2 = self.q2(c2)

        l2_o = self.upsample(q_2)
        c1 = l2_o + t_1
        q_1 = self.q1(c1)

        l1_o = self.upsample(q_1)
        c0 = l1_o + t_0
        z_1 = self.z1(c0)

        x = torch.sigmoid(self.out(z_1))

        return x
    
    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            fan_in = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
            fan_in // m.groups
            std = math.sqrt(2.0 / fan_in)
            m.weight.data.normal_(0, std)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)