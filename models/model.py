from torch import nn
from models.layers.blocks import *
import torch.nn.functional as F
import torch
import math
import warnings


def UpsamplingNearest2d(x, scale_factor=2, mode='nearest'):
    return F.interpolate(x, scale_factor=scale_factor, mode=mode)

class DUCK_Net(nn.Module):
    def __init__(self, in_channels):
        super(DUCK_Net, self).__init__()

        self.conv1 = Conv2dSamePadding(3, in_channels * 2, kernel_size=2, stride=2)
        self.conv2 = Conv2dSamePadding(
            in_channels * 2, in_channels * 4, kernel_size=2, stride=2
        )
        self.conv3 = Conv2dSamePadding(
            in_channels * 4, in_channels * 8, kernel_size=2, stride=2
        )
        self.conv4 = Conv2dSamePadding(
            in_channels * 8, in_channels * 16, kernel_size=2, stride=2
        )
        self.conv5 = Conv2dSamePadding(
            in_channels * 16, in_channels * 32, kernel_size=2, stride=2
        )

        self.t0 = Conv_Block(3, in_channels, "duckv2", layers=1)

        self.l1i = Conv2dSamePadding(
            in_channels, in_channels * 2, kernel_size=2, stride=2
        )
        self.l2i = Conv2dSamePadding(
            in_channels * 2, in_channels * 4, kernel_size=2, stride=2
        )
        self.l3i = Conv2dSamePadding(
            in_channels * 4, in_channels * 8, kernel_size=2, stride=2
        )
        self.l4i = Conv2dSamePadding(
            in_channels * 8, in_channels * 16, kernel_size=2, stride=2
        )
        self.l5i = Conv2dSamePadding(
            in_channels * 16, in_channels * 32, kernel_size=2, stride=2
        )

        self.t1 = Conv_Block(in_channels * 2, in_channels * 2, "duckv2", layers=1)
        self.t2 = Conv_Block(in_channels * 4, in_channels * 4, "duckv2", layers=1)
        self.t3 = Conv_Block(in_channels * 8, in_channels * 8, "duckv2", layers=1)
        self.t4 = Conv_Block(in_channels * 16, in_channels * 16, "duckv2", layers=1)
        self.t5_1 = Conv_Block(in_channels * 32, in_channels * 32, "resnet", layers=2)
        # self.t5_3 = Conv_Block(in_channels * 32, in_channels * 16, "resnet", layers=1)
        self.t5_2 = Conv_Block(in_channels * 16, in_channels * 16, "resnet", layers=1)

        self.q4 = Conv_Block(in_channels * 16, in_channels * 8, "duckv2", layers=1)
        self.q3 = Conv_Block(in_channels * 8, in_channels * 4, "duckv2", layers=1)
        self.q2 = Conv_Block(in_channels * 4, in_channels * 2, "duckv2", layers=1)
        self.q1 = Conv_Block(in_channels * 2, in_channels, "duckv2", layers=1)
        self.z1 = Conv_Block(in_channels, in_channels, "duckv2", layers=1)

        self.out = nn.Conv2d(in_channels, 1, kernel_size=1)

        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in')
                m.bias.detach().zero_()
            elif isinstance(m, torch.nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in')
                m.bias.detach().zero_()

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
        # t_53 = self.t5_3(t_51)
        t_52 = self.t5_2(t_51)

        l5_o = UpsamplingNearest2d(t_52)
        c4 = l5_o + t_4
        q_4 = self.q4(c4)

        l4_o = UpsamplingNearest2d(q_4)
        c3 = l4_o + t_3
        q_3 = self.q3(c3)

        l3_o = UpsamplingNearest2d(q_3)
        c2 = l3_o + t_2
        q_2 = self.q2(c2)

        l2_o = UpsamplingNearest2d(q_2)
        c1 = l2_o + t_1
        q_1 = self.q1(c1)

        l1_o = UpsamplingNearest2d(q_1)
        c0 = l1_o + t_0
        z_1 = self.z1(c0)

        x = torch.sigmoid(self.out(z_1))

        return x
