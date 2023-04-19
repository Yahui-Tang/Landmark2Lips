import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from models.base import *


class InpaintGenerator(BaseNetwork):
    def __init__(self, residual_blocks=7, init_weights=True):
        super(InpaintGenerator, self).__init__()

        self.encoder1 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=4, out_channels=64, kernel_size=7, padding=0),
            nn.InstanceNorm2d(64, track_running_stats=False),
            nn.ReLU(True))

        self.encoder2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128, track_running_stats=False),
            nn.ReLU(True),
        )

        self.encoder3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256, track_running_stats=False),
            nn.ReLU(True)
        )

        self.fushion1 = nn.Sequential(
            nn.Conv2d(in_channels=256,out_channels=256,kernel_size=1),
            nn.InstanceNorm2d(256,track_running_stats=False),
            nn.ReLU(True)
        )

        self.fushion2 = nn.Sequential(
            nn.Conv2d(in_channels=128,out_channels=128,kernel_size=1),
            nn.InstanceNorm2d(128,track_running_stats=False),
            nn.ReLU(True)
        )

        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128, track_running_stats=False),
            nn.ReLU(True),
        )

        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(64, track_running_stats=False),
            nn.ReLU(True),
        )

        self.decoder3 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=128, out_channels=3, kernel_size=7, padding=0),
        )

        blocks = []
        for _ in range(residual_blocks):
            block = ResnetBlock(256, 2)
            blocks.append(block)

        self.middle = nn.Sequential(*blocks)

        self.auto_attn = Auto_Attn(input_nc=256, norm_layer=None)

        if init_weights:
            self.init_weights()


    def forward(self, x, mask_whole, mask_half, mask_quarter):

        f_e1 = self.encoder1(x)
        f_e2 = self.encoder2(f_e1)
        f_e3 = self.encoder3(f_e2)
        x = self.middle(f_e3)
        x, _ = self.auto_attn(x, f_e3, mask_quarter)
        x = self.decoder1(x)
        x = self.fushion1(torch.cat((f_e2*(1-mask_half),x),dim=1))
        x = self.decoder2(x)
        x = self.fushion2(torch.cat((f_e1*(1-mask_whole),x),dim=1))
        x = self.decoder3(x)
        x = (torch.tanh(x) + 1) / 2

        return x
