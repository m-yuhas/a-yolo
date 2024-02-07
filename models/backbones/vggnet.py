"""Vggnet Backbone for SSD detector"""

import torch
from torch import nn
import pytorch_lightning


class VGGNET(pytorch_lightning.LightningModule):
    """
    VGGNET Encoder without classification head
    """
    def __init__(
        self,
        depths=(2, 2, 3, 3, 3),
        channels=(64, 128, 256, 512, 512),
        out_features=("block0", "block1", "block2", "block3", "block4"),
        norm='bn',
        act="relu",
    ):
        super().__init__()

        # parameters of the network
        self.out_features = out_features

        # block0
        self.block0_quant = torch.ao.quantization.QuantStub()
        self.block0 = VggnetEncBlock(3, channels[0], depths[0])
        self.block0_dequant = torch.ao.quantization.DeQuantStub()

        # block1
        self.block1_quant = torch.ao.quantization.QuantStub()
        self.block1 = VggnetEncBlock(channels[0], channels[1], depths[1])
        self.block1_dequant = torch.ao.quantization.DeQuantStub()

        # block2
        self.block2_quant = torch.ao.quantization.QuantStub()
        self.block2 = VggnetEncBlock(channels[1], channels[2], depths[2])
        self.block2_dequant = torch.ao.quantization.DeQuantStub()

        # block3
        self.block3_quant = torch.ao.quantization.QuantStub()
        self.block3 = VggnetEncBlock(channels[2], channels[3], depths[3])
        self.block3_dequant = torch.ao.quantization.DeQuantStub()

        # block4
        self.block4_quant = torch.ao.quantization.QuantStub()
        self.block4 = VggnetEncBlock(channels[3], channels[4], depths[4], pool=False)
        self.block4_dequant = torch.ao.quantization.DeQuantStub()

    def forward(self, x):
        outputs = {}
        x = self.block0(x)
        outputs["block0"] = x
        x = self.block1(x)
        outputs["block1"] = x
        x = self.block2(x)
        outputs["block2"] = x
        x = self.block3(x)
        outputs["block3"] = x
        x = self.block4(x)
        outputs["block4"] = x
        return [v for k, v in outputs.items() if k in self.out_features]


class VggnetEncBlock(pytorch_lightning.LightningModule):

    def __init__(self, in_channels, out_channels, n_conv, pool=True):
        super().__init__()
        self.conv = torch.nn.Sequential()
        self.conv.add_module(
            f'conv0',
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        )
        self.conv.add_module(
            f'bn0',
            torch.nn.BatchNorm2d(out_channels),
        )
        self.conv.add_module(
            f'act0',
            torch.nn.ReLU()
        )
        for layer in range(1, n_conv):
            self.conv.add_module(f'conv{layer}', torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
            self.conv.add_module(f'bn{layer}', torch.nn.BatchNorm2d(out_channels))
            self.conv.add_module(f'act{layer}', torch.nn.ReLU())
        if pool:
            self.conv.add_module('maxpool', torch.nn.MaxPool2d(2))
    
    def forward(self, x):
        return self.conv(x)


class VggnetSsdHead(pytorch_lightning.LightningModule):
    def __init__(self, in_channels = [512, 512], n_classes=2):
        super().__init__()
        self.head1conv = torch.nn.Conv2d(in_channels[0], 4*(n_classes+4), kernel_size=3, padding=1)
        self.head1bn = torch.nn.BatchNorm2d(4*(n_classes+4))
        self.head1act = torch.nn.Sigmoid()

        self.head2conv0 = torch.nn.Conv2d(in_channels[1], 1024, kernel_size=3, padding=1)
        self.head2bn0 = torch.nn.BatchNorm2d(1024)
        self.head2act0 = torch.nn.ReLU()
        self.head2conv1 = torch.nn.Conv2d(1024, 1024, kernel_size=1)
        self.head2bn1 = torch.nn.BatchNorm2d(1024)
        self.head2act1 = torch.nn.ReLU()
        self.head2conv2 = torch.nn.Conv2d(1024, 6*(n_classes+4), kernel_size=3, pading=1)
        self.head2bn2 = torch.nn.BatchNorm2d(6*(n_classes+4))
        self.head2act2 = torch.nn.Sigmoid()

        self.head3conv0 = torch.nn.Conv2d(1024, 512, kernel_size=3, stride=2, padding=1)
        self.head3bn0 = torch.nn.BatchNorm2d(512)
        self.head3act0 = torch.nn.ReLU()
        self.head3conv1 = torch.nn.Conv2d(512, 256, kernel_size=1)
        self.head3bn1 = torch.nn.BatchNorm2d(256)
        self.head3act1 = torch.nn.ReLU()
        self.head3conv2 = torch.nn.Conv2d(256, 6*(n_classes+4), kernel_size=3, padding=1)
        self.head3bn2 = torch.nn.BatchNorm2d(6*(n_classes+4))
        self.head3act2 = torch.nn.Sigmoid()
        
        self.head4conv0 = torch.nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)
        self.head4bn0 = torch.nn.BatchNorm2d(256)
        self.head4act0 = torch.nn.ReLU()
        self.head4conv1 = torch.nn.Conv2d(256, 128, kernel_size=1)
        self.head4bn1 = torch.nn.BatchNorm2d(128)
        self.head4act1 = torch.nn.ReLU()
        self.head4conv2 = torch.nn.Conv2d(128, 6*(n_classes+4), kernel_size=3, padding=1)
        self.head4bn2 = torch.nn.BatchNorm2d(6*(n_classes+4))
        self.head4act2 = torch.nn.Sigmoid()

        self.head5conv0 = torch.nn.Conv2d(128, 256, kernel_size=3, padding=0)
        self.head5bn0 = torch.nn.BatchNorm2d(256)
        self.head5act1 = torch.nn.ReLU()
        self.head5conv1 = torch.nn.Conv2d(256, 128, kernel_size=1)
        self.head5bn1 = torch.nn.BatchNorm2d(128)
        self.head5act1 = torch.nn.ReLU()
        self.head5conv2 = torch.nn.Conv2d(128, 4*(n_classes+4), kernel_size=3, padding=1)
        self.head5bn2 = torch.nn.BatchNorm2d(4*(n_classes+4))
        self.head5act2 = torch.nn.Sigmoid()

        self.head6conv0 = torch.nn.Conv2d(128, 256, kernel_size=3, padding=0)
        self.head6bn0 = torch.nn.BatchNorm2d(256)
        self.head6act0 = torch.nn.ReLU()
        self.head6conv1 = torch.nn.Conv2d(256, 128, kernel_size=1)
        self.head6bn1 = torch.nn.BatchNorm2d(128)
        self.head6act1 = torch.nn.ReLU()
        self.head6conv2 = torch.nn.Conv2d(128, 4*(n_classes+4), kernel_size=1)
        self.head6bn2 = torch.nn.BatchNorm2d(4 * (n_classes+4))
        self.head6act2 = torch.nn.Sigmoid()

    def forward(self, x):
        c3, c4 = x
        h1 = self.head1act0(self.head1bn0(self.head1conv0(c3)))
        
        c4 = self.head2act0(self.head2bn0(self.head2conv0(c4)))
        c4 = self.head2act1(self.head2bn1(self.head2conv1(c4)))
        h2 = self.head2act2(self.head2bn2(self.head2conv2(c4)))

        c4 = self.head3act0(self.head3bn0(self.head3conv0(c4)))
        c4 = self.head3act1(self.head3bn1(self.head3conv1(c4)))
        h3 = self.head3act2(self.head3bn2(self.head3conv2(c4)))

        c4 = self.head4act0(self.head4bn0(self.head4conv0(c4)))
        c4 = self.head4act1(self.head4bn1(self.head4conv1(c4)))
        h4 = self.head4act2(self.head4bn2(self.head4conv2(c4)))

        c4 = self.head5act0(self.head5bn0(self.head5conv0(c4)))
        c4 = self.head5act1(self.head5bn1(self.head5conv1(c4)))
        h5 = self.head5act2(self.head5bn2(self.head5conv2(c4)))

        c4 = self.head6act0(self.head6bn0(self.head6conv0(c4)))
        c4 = self.head6act1(self.head6bn1(self.head6conv1(c4)))
        h6 = self.head6act2(self.head6bn2(self.head6conv2(c4)))

        return h1, h2, h3, h4, h5, h6


class Vggnet11Enc(pytorch_lightning.LightningModule):

    def __init__(self):
        super().__init__()
        self.block0_quant = torch.ao.quantization.QuantStub()
        self.block0 = VggnetEncBlock(3, 64, 1)
        self.block0_dequant = torch.ao.quantization.DeQuantStub()

        self.block1_quant = torch.ao.quantization.QuantStub()
        self.block1 = VggnetEncBlock(64, 128, 1)
        self.block1_dequant = torch.ao.quantization.DeQuantStub()

        self.block2_quant = torch.ao.quantization.QuantStub()
        self.block2 = VggnetEncBlock(128, 256, 2)
        self.block2_dequant = torch.ao.quantization.DeQuantStub()

        self.block3_quant = torch.ao.quantization.QuantStub()
        self.block3 = VggnetEncBlock(256, 512, 2)
        self.block3_dequant = torch.ao.quantization.DeQuantStub()

        self.block4_quant = torch.ao.quantization.QuantStub()
        self.block4 = VggnetEncBlock(512, 512, 2)
        self.block4_dequant = torch.ao.quantization.DeQuantStub()


    def forward(self, x):
        y = self.block0_quant(x)
        y = self.block0(y)
        y = self.block0_dequant(y)

        y = self.block1_quant(y)
        y = self.block1(y)
        y = self.block1_dequant(y)

        y = self.block2_quant(y)
        y = self.block2(y)
        y = self.block2_dequant(y)

        y = self.block3_quant(y)
        y = self.block3(y)
        y = self.block3_dequant(y)

        y = self.block4_quant(y)
        y = self.block4(y)
        y = self.block4_dequant(y)

        return y 

class Vggnet13Enc(pytorch_lightning.LightningModule):

    def __init__(self):
        super().__init__()
        self.block0_quant = torch.ao.quantization.QuantStub()
        self.block0 = VggnetEncBlock(3, 64, 2)
        self.block0_dequant = torch.ao.quantization.DeQuantStub()

        self.block1_quant = torch.ao.quantization.QuantStub()
        self.block1 = VggnetEncBlock(64, 128, 2)
        self.block1_dequant = torch.ao.quantization.DeQuantStub()

        self.block2_quant = torch.ao.quantization.QuantStub()
        self.block2 = VggnetEncBlock(128, 256, 2)
        self.block2_dequant = torch.ao.quantization.DeQuantStub()

        self.block3_quant = torch.ao.quantization.QuantStub()
        self.block3 = VggnetEncBlock(256, 512, 2)
        self.block3_dequant = torch.ao.quantization.DeQuantStub()

        self.block4_quant = torch.ao.quantization.QuantStub()
        self.block4 = VggnetEncBlock(512, 512, 2)
        self.block4_dequant = torch.ao.quantization.DeQuantStub()


    def forward(self, x):
        y = self.block0_quant(x)
        y = self.block0(y)
        y = self.block0_dequant(y)

        y = self.block1_quant(y)
        y = self.block1(y)
        y = self.block1_dequant(y)

        y = self.block2_quant(y)
        y = self.block2(y)
        y = self.block2_dequant(y)

        y = self.block3_quant(y)
        y = self.block3(y)
        y = self.block3_dequant(y)

        y = self.block4_quant(y)
        y = self.block4(y)
        y = self.block4_dequant(y)

        return y 


class Vggnet16Enc(pytorch_lightning.LightningModule):

    def __init__(self):
        super().__init__()
        self.block0_quant = torch.ao.quantization.QuantStub()
        self.block0 = VggnetEncBlock(3, 64, 2)
        self.block0_dequant = torch.ao.quantization.DeQuantStub()

        self.block1_quant = torch.ao.quantization.QuantStub()
        self.block1 = VggnetEncBlock(64, 128, 2)
        self.block1_dequant = torch.ao.quantization.DeQuantStub()

        self.block2_quant = torch.ao.quantization.QuantStub()
        self.block2 = VggnetEncBlock(128, 256, 3)
        self.block2_dequant = torch.ao.quantization.DeQuantStub()

        self.block3_quant = torch.ao.quantization.QuantStub()
        self.block3 = VggnetEncBlock(256, 512, 3)
        self.block3_dequant = torch.ao.quantization.DeQuantStub()

        self.block4_quant = torch.ao.quantization.QuantStub()
        self.block4 = VggnetEncBlock(512, 512, 3)
        self.block4_dequant = torch.ao.quantization.DeQuantStub()


    def forward(self, x):
        y = self.block0_quant(x)
        y = self.block0(y)
        y = self.block0_dequant(y)

        y = self.block1_quant(y)
        y = self.block1(y)
        y = self.block1_dequant(y)

        y = self.block2_quant(y)
        y = self.block2(y)
        y = self.block2_dequant(y)

        y = self.block3_quant(y)
        y = self.block3(y)
        y = self.block3_dequant(y)

        y = self.block4_quant(y)
        y = self.block4(y)
        y = self.block4_dequant(y)

        return y 


class Vggnet19Enc(pytorch_lightning.LightningModule):

    def __init__(self):
        super().__init__()
        self.block0_quant = torch.ao.quantization.QuantStub()
        self.block0 = VggnetEncBlock(3, 64, 2)
        self.block0_dequant = torch.ao.quantization.DeQuantStub()

        self.block1_quant = torch.ao.quantization.QuantStub()
        self.block1 = VggnetEncBlock(64, 128, 2)
        self.block1_dequant = torch.ao.quantization.DeQuantStub()

        self.block2_quant = torch.ao.quantization.QuantStub()
        self.block2 = VggnetEncBlock(128, 256, 4)
        self.block2_dequant = torch.ao.quantization.DeQuantStub()

        self.block3_quant = torch.ao.quantization.QuantStub()
        self.block3 = VggnetEncBlock(256, 512, 4)
        self.block3_dequant = torch.ao.quantization.DeQuantStub()

        self.block4_quant = torch.ao.quantization.QuantStub()
        self.block4 = VggnetEncBlock(512, 512, 4)
        self.block4_dequant = torch.ao.quantization.DeQuantStub()


    def forward(self, x):
        y = self.block0_quant(x)
        y = self.block0(y)
        y = self.block0_dequant(y)

        y = self.block1_quant(y)
        y = self.block1(y)
        y = self.block1_dequant(y)

        y = self.block2_quant(y)
        y = self.block2(y)
        y = self.block2_dequant(y)

        y = self.block3_quant(y)
        y = self.block3(y)
        y = self.block3_dequant(y)

        y = self.block4_quant(y)
        y = self.block4(y)
        y = self.block4_dequant(y)

        return y 
