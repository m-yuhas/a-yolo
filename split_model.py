# Neeeded for Conversion
import argparse
import os
import torch
import torchvision
from typing import Callable, List, Tuple


# Needed for YOLO
from pytorch_lightning import Trainer, seed_everything
from utils.defaults import train_argument_parser, load_config
from utils.build_data import build_data
from utils.build_logger import build_logger
from pytorch_lightning.callbacks import ModelCheckpoint
from PL_Modules.build_detection import build_model
from PL_Modules.pl_detection import LitDetection


class YoloBlock0(torch.nn.Module):
    def __init__(self, full, b3, b2, b1, b0):
        super().__init__()
        self.backbone_q = full.backbone.stem_q
        self.backbone = full.backbone.stem
        self.backbone_dq_fwd = full.backbone.stem_dq
        self.downsample_b0h3 = b0.backbone.stem_exit0
        self.downsample_b0h2 = b0.backbone.stem_exit1
        self.downsample_dq_h2 = b0.backbone.stem_dq2
        self.downsample_b0h1 = b0.backbone.stem_exit2
        self.downsample_dq_h1 = b0.backbone.stem_dq3
        self.downsample_b0h0 = b0.backbone.stem_exit
        self.downsample_dq_h0 = b0.backbone.stem_dq4

        self.neck = b0.neck
        self.head = b0.head


    def forward(self, x):
        x0 = self.backbone_q(x)
        x0 = self.backbone(x0)
        x0h3 = self.downsample_b0h3(x0)
        x0h2 = self.downsample_b0h2(x0h3)
        x0h1 = self.downsample_b0h1(x0h2)
        x0h0 = self.downsample_b0h0(x0h1)
        x0h0 = self.downsample_dq_h0(x0h0)
        x0h1 = self.downsample_dq_h1(x0h1)
        x0h2 = self.downsample_dq_h2(x0h2)
        x0 = self.backbone_dq_fwd(x0)

        y = self.neck((x0,))
        det1, = self.head(y)
        return det1, x0, x0h1, x0h2


class YoloBlock1(torch.nn.Module):
    def __init__(self, full, b3, b2, b1, b0):
        super().__init__()
        self.backbone_q = full.backbone.stage1_q
        self.backbone = full.backbone.stage1
        self.backbone_dq_fwd = full.backbone.stage1_dq
        self.downsample_b1h3 = b1.backbone.block1_exit0
        self.downsample_dq_h3 = b1.backbone.block1_dq1
        self.downsample_b1h2 = b1.backbone.block1_exit1
        self.downsample_dq_h2 = b1.backbone.block1_dq2
        self.downsample_b1h1 = b1.backbone.block1_exit
        self.downsample_dq_h1 = b1.backbone.block1_dq3

        self.neck = b1.neck
        self.head = b2.head

    def forward(self, x0, x0h1, x0h2):
        x1 = self.backbone_q(x0)
        x1 = self.backbone(x1)
        x1h3 = self.downsample_b1h3(x1)
        x1h2 = self.downsample_b1h2(x1h3)
        x1h1 = self.downsample_b1h1(x1h2)
        x1h1 = self.downsample_dq_h1(x1h1)
        x1h2 = self.downsample_dq_h2(x1h2)
        x1h3 = self.downsample_dq_h3(x1h3)
        x1 = self.backbone_dq_fwd(x1)

        y = self.neck((x0h1, x1h1))
        det1, det2 = self.head(y)
        return det1, det2, x0h2, x1, x1h2, x1h3

class YoloBlock2(torch.nn.Module):
    def __init__(self, full, b3, b2, b1, b0):
        super().__init__()
        self.backbone_q = full.backbone.stage2_q
        self.backbone = full.backbone.stage2
        self.backbone_dq_fwd = full.backbone.stage2_dq
        self.downsample_b2h3 = b2.backbone.block2_exit0
        self.downsample_dq_h3 = b2.backbone.block2_dq1
        self.downsample_b2h2 = b2.backbone.block2_exit
        self.downsample_dq_h2 = b2.backbone.block2_dq2

        self.neck = b2.neck
        self.head = b2.head

    def forward(self, x0h2, x1, x1h2, x1h3):
        x2 = self.backbone_q(x1)
        x2 = self.backbone(x2)
        x2h3 = self.downsample_b2h3(x2)
        x2h2 = self.downsample_b2h2(x2h3)
        x2h2 = self.downsample_dq_h2(x2h2)
        x2h3 = self.downsample_dq_h3(x2h3)
        x2 = self.backbone_dq_fwd(x2)

        y = self.neck((x0h2, x1h2, x2h2))
        det1, det2, det3 = self.head(y)
        return det1, det2, det3, x1h3, x2, x2h3

    
class YoloBlock3(torch.nn.Module):
    def __init__(self, full, b3, b2, b1, b0):
        super().__init__()
        self.backbone_q = full.backbone.stage3_q
        self.backbone = full.backbone.stage3
        self.backbone_dq_fwd = full.backbone.stage3_dq
        self.downsample_b3 = b3.backbone.self.block3_exit
        self.downsample_dq_h3 = b3.backbone.stage3_dq1

        self.neck = b3.neck
        self.head = b3.head

    def forward(self, x1h3, x2, x2h3):
        x3 = self.backbone_q(x2)
        x3 = self.backbone(x3)
        x3h3 = self.downsample_b3(x3)
        x3h3 = self.downsample_dq_h3(x3h3)
        x3 = self.backbone_dq_fwd(x3)

        y = self.neck((x1h3, x2h3, x3h3))
        det1, det2, det3 = self.head(y)
        return det1, det2, det3, x2, x3


class YoloBlock4(torch.nn.Module):
    def __init__(self, full, b3, b2, b1, b0):
        super().__init__()
        self.backbone_q = full.backbone.stage4_q
        self.backbone = full.backbone.stage4
        self.backbone_dq = full.backbone.stage4_dq

        self.neck = full.neck
        self.head = full.head

    def forward(self, x2, x3):
        x4 = self.backbone_q(x3)
        x4 = self.backbone(x4)
        x4 = self.backbone_dq(x4)
        y = self.neck((x2, x3, x4))
        det1, det2, det3 = self.head(y)
        return det1, det2, det3


def split_model(full: str, b3: str, b2: str, b1: str, b0: str) -> List[torch.nn.Module]:
    full = torch.load(full)
    b3 = torch.load(b3)
    b2 = torch.load(b2)
    b1 = torch.load(b1)
    b0 = torch.load(b0)
    return [
        YoloBlock0(full, b3, b2, b1, b0),
        YoloBlock1(full, b3, b2, b1, b0),
        YoloBlock2(full, b3, b2, b1, b0),
        YoloBlock3(full, b3, b2, b1, b0),
        YoloBlock4(full, b3, b2, b1, b0),
    ]


def static_quantize(cal_set: str, blocks: Tuple[torch.nn.Module]) -> Tuple[torch.nn.Module]:
    block0, block1, block2, block3, block4 = blocks
    block0 = block0.to('cpu')
    block1 = block1.to('cpu')
    block2 = block2.to('cpu')
    block3 = block3.to('cpu')
    block4 = block4.to('cpu')
    block0.eval()
    block1.eval()
    block2.eval()
    block3.eval()
    block4.eval()

    # Fuse Conv, bn and relu
    block0.fuse_model()
    block1.fuse_model()
    block2.fuse_model()
    block3.fuse_model()
    block4.fuse_model()

    # Specify quantization configuration
    # Start with simple min/max range estimation and per-tensor quantization of weights
    #block0.qconfig = torch.quantization.default_qconfig
    #block1.qconfig = torch.quantization.default_qconfig
    #block2.qconfig = torch.quantization.default_qconfig
    #block3.qconfig = torch.quantization.default_qconfig
    #block4.qconfig = torch.quantization.default_qconfig
    block0.qconfig = torch.quantization.get_default_qconfig('x86')
    block1.qconfig = torch.quantization.get_default_qconfig('x86')
    block2.qconfig = torch.quantization.get_default_qconfig('x86')
    block3.qconfig = torch.quantization.get_default_qconfig('x86')
    block4.qconfig = torch.quantization.get_default_qconfig('x86')
    block0 = torch.quantization.prepare(block0)
    block1 = torch.quantization.prepare(block1)
    block1 = torch.quantization.prepare(block2)
    block1 = torch.quantization.prepare(block3)
    block1 = torch.quantization.preapre(block4)

    # Calibrate with the training set
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((416, 416))
    ])
    cal_data = torchvision.datasets.ImageFolder(
        cal_set,
        transforms=transform,
        is_valid_file=lambda x: True if x.endswith('.png') else False
    )
    data_loader = torch.utils.data.DataLoader(cal_data, batch_size=64, shuffle=True)
    with torch.no_grad():
        for x, _ in data_loader:
            det1, x0, x0h1, x0h2 = block0(x)
            det1, det2, x0h2, x1, x1h2, x1h3 = block1(x0, x0h1, x0h2)
            det1, det2, det3, x1h3, x2, x2h3 = block2(x0h2, x1, x1h2, x1h3)
            det1, det2, det3, x2, x3 = block3(x1h3, x2, x2h3)
            det1, det2, det3 = block4(x2, x3)

    # Convert to quantized model
    block0 = torch.quantization.convert(block0)
    block1 = torch.quantization.convert(block1)
    block2 = torch.quantization.convert(block2)
    block3 = torch.quantization.convert(block3)
    block4 = torch.quantization.convert(block4)

    return [block0, block1, block2, block3, block4]


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Quantize a Beta-VAE model')
    parser.add_argument(
        '--full',
        help='Weights for full model',
    )
    parser.add_argument(
        '--b3',
        help='Weights for block3'
    )
    parser.add_argument(
        '--b2',
        help='Weights for block2'
    )
    parser.add_argument(
        '--b1',
        help='Weights for block1'
    )
    parser.add_argument(
        '--b0',
        help='Weights for block0'
    )
    parser.add_argument(
        '--calset',
        help='Calibration set for static quantization'
    )
    args = parser.parse_args()
    blocks = split_model(
        args.full,
        args.b3,
        args.b2,
        args.b1,
        args.b0,
    )
    for idx, block in enumerate(blocks):
        torch.save(block, f'yoloblock{idx}.pt')
    q_blocks = static_quantize(args.calset, blocks)
    for idx, block in enumerate(blocks):
        torch.save(block, f'yoloblock{idx}q.pt')
    #model_scripted = torch.jit.script(model) # Export to TorchScript
    #model_scripted.save('model_scripted.pt') # Save