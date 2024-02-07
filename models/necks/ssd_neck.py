import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning
from models.layers.network_blocks import BaseConv, SPPCSPC


class SSDNECK(pytorch_lightning.LightningModule):
    """
    Need 2 inputs and 6 outputs.
    """

    def __init__(
            self,
            depths=(1024, 512, 256, 128),
            in_channels=(512, 512),
            norm='bn',
            act="silu",
    ):
        super().__init__()
        self.h1conv0 = torch.nn.Conv2d(in_channels[0], depths[0], kernel_size=3, padding=1)
        self.h1bn0 = torch.nn.BatchNorm2d(depths[0])
        self.h1act0 = torch.nn.ReLU()
        self.h1conv1 = torch.nn.Conv2d(depths[0], depths[0], kernel_size=1)
        self.h1bn1 = torch.nn.BatchNorm2d(depths[0])
        self.h1act1 = torch.nn.ReLU()

        self.h2conv0 = torch.nn.Conv2d(depths[0], depths[1], kernel_size=3, stride=2, padding=1)
        self.h2bn0 = torch.nn.BatchNorm2d(depths[1])
        self.h2act0 = torch.nn.ReLU()
        self.h2conv1 = torch.nn.Conv2d(depths[1], depths[2], kernel_size=1)
        self.h2bn1 = torch.nn.BatchNorm2d(depths[2])
        self.h2act1 = torch.nn.ReLU()

        self.h3conv0 = torch.nn.Conv2d(depths[2], depths[2], kernel_size=3, stride=2, padding=1)
        self.h3bn0 = torch.nn.BatchNorm2d(depths[2])
        self.h3act0 = torch.nn.ReLU()
        self.h3conv1 = torch.nn.Conv2d(depths[2], depths[3], kernel_size=1)
        self.h3bn1 = torch.nn.BatchNorm2d(depths[3])
        self.h3act1 = torch.nn.ReLU()

        self.h4conv0 = torch.nn.Conv2d(depths[3], depths[2], kernel_size=3, padding=0)
        self.h4bn0 = torch.nn.BatchNorm2d(depths[2])
        self.h4act0 = torch.nn.ReLU()
        self.h4conv1 = torch.nn.Conv2d(depths[2], depths[3], kernel_size=1)
        self.h4bn1 = torch.nn.BatchNorm2d(depths[3])
        self.h4act1 = torch.nn.ReLU()

        self.h5conv0 = torch.nn.Conv2d(depths[3], depths[2], kernel_size=3, padding=0)
        self.h5bn0 = torch.nn.BatchNorm2d(depths[2])
        self.h5act0 = torch.nn.ReLU()
        self.h5conv1 = torch.nn.Conv2d(depths[2], depths[3], kernel_size=1)
        self.h5bn1 = torch.nn.BatchNorm2d(depths[3])
        self.h5act1 = torch.nn.ReLU()

    def forward(self, inputs):
        #  backbone
        [x0, x1] = inputs
        h0 = x0
        h1 = self.h1act0(self.h1bn0(self.h1conv0(x1)))
        h1 = self.h1act1(self.h1bn1(self.h1conv1(h1)))
        h2 = self.h2act0(self.h2bn0(self.h2conv0(h1)))
        h2 = self.h2act1(self.h2bn1(self.h2conv1(h2)))
        h3 = self.h3act0(self.h3bn0(self.h3conv0(h2)))
        h3 = self.h3act1(self.h3bn1(self.h3conv1(h3)))
        h4 = self.h4act0(self.h4bn0(self.h4conv0(h3)))
        h4 = self.h4act1(self.h4bn1(self.h4conv1(h4)))
        h5 = self.h5act0(self.h5bn0(self.h5conv0(h4)))
        h5 = self.h5act1(self.h5bn1(self.h5conv1(h5)))

        outputs = (h0, h1, h2, h3, h4, h5)
        return outputs

