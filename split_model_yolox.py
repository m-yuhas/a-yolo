# Neeeded for Conversion
import argparse
import os
import torch
import torchvision
import pytorch_lightning
from typing import Callable, List, Tuple


# Needed for YOLO
import models
from pytorch_lightning import Trainer, seed_everything
from utils.defaults import train_argument_parser, load_config
from utils.build_data import build_data
from utils.build_logger import build_logger
from pytorch_lightning.callbacks import ModelCheckpoint
from PL_Modules.build_detection import build_model
from PL_Modules.pl_detection import LitDetection


class YoloBlock0(pytorch_lightning.LightningModule):
    def __init__(self, full, ee4, ee3, ee2):
        super().__init__()
        self.resize = torchvision.transforms.Resize((608, 608))
        self.stem = full.backbone.stem
        self.stage1 = full.backbone.stage1
        self.stage2 = full.backbone.stage2
        self.head = ee2.head

    def forward(self, x):
        x = self.resize(x)
        x0 = self.stem(x)
        x1 = self.stage1(x0)
        x2 = self.stage2(x1)
        y0, y1, y2 = self.head((x0, x1, x2))
        return y0, y1, y2, x1, x2


class YoloBlock1(pytorch_lightning.LightningModule):
    def __init__(self, full, ee4, ee3, ee2):
        super().__init__()
        self.stage3 = full.backbone.stage3
        self.head = ee3.head

    def forward(self, x1, x2):
        x3 = self.stage3(x2)
        y1, y2, y3 = self.head((x1, x2, x3))
        return y1, y2, y3, x2, x3


class YoloBlock2(pytorch_lightning.LightningModule):
    def __init__(self, full, ee4, ee3, ee2):
        super().__init__()
        self.stage4 = full.backbone.stage4
        self.head = ee4.head

    def forward(self, x2, x3):
        x4 = self.stage4(x3)
        y2, y3, y4 = self.head((x2, x3, x4))
        return y2, y3, y4, x2, x3, x4
    

class YoloBlock3(pytorch_lightning.LightningModule):
    def __init__(self, full, ee4, ee3, ee2):
        super().__init__()
        self.neck = full.neck
        self.head = full.head

    def forward(self, x2, x3, x4):
        xf = self.neck((x2, x3, x4))
        return self.head(xf)


class Yolox(pytorch_lightning.LightningModule):
    def __init__(self, full, ee4, ee3, ee2):
        super().__init__()
        self.model = full

    def forward(self, x):
        return self.model(x)

def split_model(full: str, ee4: str, ee3: str, ee2: str) -> List[torch.nn.Module]:
    full = torch.load(full)
    ee4 = torch.load(ee4)
    ee3 = torch.load(ee3)
    ee2 = torch.load(ee2)
    return [
        YoloBlock0(full, ee4, ee3, ee2),
        YoloBlock1(full, ee4, ee3, ee2),
        YoloBlock2(full, ee4, ee3, ee2),
        YoloBlock3(full, ee4, ee3, ee2),
        Yolox(full, ee4, ee3, ee2),
    ]


def strip_lightning_info(blocks: List[torch.nn.Module]):
    out_blocks = []
    for block in blocks:
        for param in block.modules():
            if isinstance(param, models.layers.network_blocks.BaseConv):
                param._should_prevent_trainer_and_dataloaders_deepcopy = False
            if isinstance(param, models.layers.network_blocks.Focus):
                param._should_prevent_trainer_and_dataloaders_deepcopy = False
            if isinstance(param, models.layers.network_blocks.CSPLayer):
                param._should_prevent_trainer_and_dataloaders_deepcopy = False
            if isinstance(param, models.layers.network_blocks.SPPBottleneck):
                param._should_prevent_trainer_and_dataloaders_deepcopy = False
            if isinstance(param, models.backbones.darknet_csp.CSPDarkNet):
                param._should_prevent_trainer_and_dataloaders_deepcopy = False
            if isinstance(param, models.necks.pafpn_csp.CSPPAFPN):
                param._should_prevent_trainer_and_dataloaders_deepcopy = False
            if isinstance(param, models.heads.decoupled_head.DecoupledHead):
                param._should_prevent_trainer_and_dataloaders_deepcopy = False
        out_blocks.append(block)
    return out_blocks


def save_gpu_blocks(blocks: List[torch.nn.Module], name: str):
    blocks = [b.to(torch.device('cuda')).eval() for b in blocks]
    block0, block1, block2, block3, full = blocks
    x = torch.randn(1,3,640,480).to(torch.device('cuda'))
    _, _, _, x1, x2 = block0(x)
    _, _, _, x2, x3 = block1(x1, x2)
    _, _, _, x2, x3, x4 = block2(x2, x3)
    _ = block3(x2, x3, x4)
    block0.to_torchscript(f'{name}-b0.pt', method='trace', example_inputs=(x))
    block1.to_torchscript(f'{name}-b1.pt', method='trace', example_inputs=(x1, x2))
    block2.to_torchscript(f'{name}-b2.pt', method='trace', example_inputs=(x2, x3))
    block3.to_torchscript(f'{name}-b3.pt', method='trace', example_inputs=(x2, x3, x4))
    full.to_torchscript('yolox.pt', method='trace', example_inputs=(x))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Quantize a Beta-VAE model')
    parser.add_argument(
        '--full',
        help='Weights for full model',
    )
    parser.add_argument(
        '--ee4',
        help='Weights for block3'
    )
    parser.add_argument(
        '--ee3',
        help='Weights for block2'
    )
    parser.add_argument(
        '--ee2',
        help='Weights for block1'
    )
    parser.add_argument(
        '--name',
        help='Name to call split model'
    )
    args = parser.parse_args()
    print('Splitting model...')
    blocks = split_model(args.full, args.ee4, args.ee3, args.ee2)
    #print('Stripping Trainer Details...')
    #blocks = strip_lightning_info(blocks)
    print('Saving GPU blocks...')
    save_gpu_blocks(blocks, args.name)