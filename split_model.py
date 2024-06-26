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

class QBaseConv(torch.nn.Module):
    def __init__(self, base):
        super().__init__()
        self.conv = base.conv
        self.norm = base.norm
        self.act = base.act

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))

class QCSPLayer(torch.nn.Module):
    def __init__(self,base):
        super().__init__()
        self.conv1 = QBaseConv(base.conv1)
        self.conv2 = QBaseConv(base.conv2)
        self.conv3 = torch.nn.Sequential(
            *[QBaseConv(b) for b in base.conv3]
        )
        self.conv4 = torch.nn.Sequential(
            *[QBaseConv(b) for b in base.conv4]
        )
        self.conv5 = QBaseConv(base.conv5)

    def forward(self,x):
        x_1 = self.conv1(x)
        x_2 = self.conv2(x)
        x_3 = self.conv3(x_2)
        x_4 = self.conv4(x_3)
        x_all = [x_1, x_2, x_3, x_4]
        x = torch.cat(x_all, dim=1)
        return self.conv5(x)

class QTransition(torch.nn.Module):
    def __init__(self, base):
        super().__init__()
        self.mp = base.mp
        self.conv1 = QBaseConv(base.conv1)
        self.conv2 = QBaseConv(base.conv2)
        self.conv3 = QBaseConv(base.conv3)

    def forward(self, x):
        x_1 = self.mp(x)
        x_1 = self.conv1(x_1)

        x_2 = self.conv2(x)
        x_2 = self.conv3(x_2)

        return torch.cat([x_2, x_1], 1)

class QSPPBottleneck(torch.nn.Module):
    def __init__(self, base):
        super().__init__()
        self.conv1 = QBaseConv(base.conv1)
        self.m = base.m
        self.conv2 = QBaseConv(base.conv2)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.cat([x] + [m(x) for m in self.m], dim=1)
        x = self.conv2(x)
        return x

class QSPPCSPC(torch.nn.Module):
    def __init__(self, base):
        super().__init__()
        self.cv1 = QBaseConv(base.cv1)
        self.cv2 = QBaseConv(base.cv2)
        self.cv3 = QBaseConv(base.cv3)
        self.cv4 = QBaseConv(base.cv4)
        self.m = base.m
        self.cv5 = QBaseConv(base.cv5)
        self.cv6 = QBaseConv(base.cv6)
        self.cv7 = QBaseConv(base.cv7)

    def forward(self, x):
        x1 = self.cv4(self.cv3(self.cv1(x)))
        y1 = self.cv6(self.cv5(torch.cat([x1] + [m(x1) for m in self.m], 1)))
        y2 = self.cv2(x)
        return self.cv7(torch.cat((y1, y2), dim=1))

class QNeckTran(torch.nn.Module):
    def __init__(self, base):
        super().__init__()
        self.mp = base.mp
        self.conv1 = QBaseConv(base.conv1)
        self.conv2 = QBaseConv(base.conv2)
        self.conv3 = QBaseConv(base.conv3)

    def forward(self, x):
        x_1 = self.mp(x)
        x_1 = self.conv1(x_1)

        x_2 = self.conv2(x)
        x_2 = self.conv3(x_2)

        return torch.cat([x_2, x_1], 1)

class QNeck(torch.nn.module):
    def __init__(self, base):
        super().__init__()
        self.spp = QSPPCSPC(base.spp)
        self.conv_for_P5 = QBaseConv(base.conv_for_P5)
        self.upsample = base.upsample
        self.conv_for_C4 = QBaseConv(base.conv_for_C4)
        self.p5_p4 = QCSPLayer(base.p5_p4)
        self.conv_for_P4 = QBaseConv(base.conv_for_P4)
        self.conv_for_C3 = QBaseConv(base.conv_for_C3)
        self.p4_p3 = QCSPLayer(base.p4_p3)
        self.downsample_conv1 = QNeckTran(base.downsample_conv1)
        self.n3_n4 = QCSPLayer(base.n3_n4)
        self.downsample_conv2 = QNeckTran(base.downsample_conv2)
        self.n4_n5 = QCSPLayer(base.n4_n5)
        self.n3 = QBaseConv(base.n3)
        self.n4 = QBaseConv(base.n4)
        self.n5 = QBaseConv(base.n5)
        self.q3 = base.q3
        self.q4 = base.q4
        self.q5 = base.q5
        self.dq3 = base.dq3
        self.dq4 = base.dq4
        self.dq5 = base.dq5

    def forward(self, inputs):
        #  backbone
        [c3, c4, c5] = inputs
        c3 = self.q3(c3)
        c4 = self.q4(c4)
        c5 = self.q5(c5)
        # top-down
        p5 = self.spp(c5)
        p5_shrink = self.conv_for_P5(p5)
        p5_upsample = self.upsample(p5_shrink)
        p4 = torch.cat([p5_upsample, self.conv_for_C4(c4)], 1)
        p4 = self.p5_p4(p4)

        p4_shrink = self.conv_for_P4(p4)
        p4_upsample = self.upsample(p4_shrink)
        p3 = torch.cat([p4_upsample, self.conv_for_C3(c3)], 1)
        p3 = self.p4_p3(p3)

        # down-top
        n3 = p3
        n3_downsample = self.downsample_conv1(n3)
        n4 = torch.cat([n3_downsample, p4], 1)
        n4 = self.n3_n4(n4)

        n4_downsample = self.downsample_conv2(n4)
        n5 = torch.cat([n4_downsample, p5], 1)
        n5 = self.n4_n5(n5)

        n3 = self.n3(n3)
        n4 = self.n4(n4)
        n5 = self.n5(n5)

        n3 = self.dq3(n3)
        n4 = self.dq4(n4)
        n5 = self.dq5(n5)
        outputs = (n3, n4, n5)
        return outputs

class QNeck2(torch.nn.Module):
    def __init__(self, base):
        super().__init__()
        self.upsample = base.upsample
        self.conv_for_C4 = QBaseConv(base.conv_for_C4)
        self.p5_p4 = QCSPLayer(base.p5_p4)
        self.conv_for_P4 = QBaseConv(base.conv_for_P4)
        self.conv_for_C3 = QBaseConv(base.conv_for_C3)
        self.p4_p3 = QCSPLayer(base.p4_p3)
        self.downsample_conv1 = QNeckTran(base.downsample_conv1)
        self.n3_n4 = QCSPLayer(base.n3_n4)
        self.downsample_conv2 = QNeckTran(base.downsample_conv2)
        self.n3 = QBaseConv(base.n3)
        self.n4 = QBaseConv(base.n4)
        
        self.q3 = base.q3
        self.q4 = base.q4
        self.dq3 = base.dq3
        self.dq4 = base.dq4

    def forward(self, inputs):
        #  backbone
        [c3, c4] = inputs
        c3 = self.q3(c3)
        c4 = self.q4(c4)
        p4 = c4
        p4 = self.p5_p4(p4)

        p4_shrink = self.conv_for_P4(p4)
        p4_upsample = self.upsample(p4_shrink)
        p3 = torch.cat([p4_upsample, self.conv_for_C3(c3)], 1)
        p3 = self.p4_p3(p3)

        # down-top
        n3 = p3
        n3_downsample = self.downsample_conv1(n3)
        n4 = torch.cat([n3_downsample, p4], 1)
        n4 = self.n3_n4(n4)

        n4_downsample = self.downsample_conv2(n4)

        n3 = self.n3(n3)
        n4 = self.n4(n4)

        n3 = self.dq3(n3)
        n4 = self.dq4(n4)
        outputs = (n3, n4)
        return outputs

class QNeck1(torch.nn.Module):
    def __init__(self,base):
        super().__init__()
        self.n3 = QBaseConv(base.n3)
        self.q3 = base.q3
        self.dq3 = base.dq3

    def forward(self, inputs):
        #  backbone
        c3 = inputs
        c3 = self.q3(c3)
        n3 = self.n3(c3)
        n3 = self.dq3(n3)
        outputs = (n3,)
        return outputs        



class YoloBlock0(torch.nn.Module):
    def __init__(self, full, b3, b2, b1, b0):
        super().__init__()
        self.backbone_q = full.backbone.stem_q
        self.backbone = torch.nn.Sequential(
            *[QBaseConv(b) for b in full.backbone.stem]
        )
        self.backbone_dq_fwd = full.backbone.stem_dq
        self.downsample_b0h3 = QTransition(b0.backbone.stem_exit0)
        self.downsample_b0h2 = QTransition(b0.backbone.stem_exit1)
        self.downsample_dq_h2 = b0.backbone.stem_dq1
        self.downsample_b0h1 = QTransition(b0.backbone.stem_exit2)
        self.downsample_dq_h1 = b0.backbone.stem_dq2
        self.downsample_b0h0 = QTransition(b0.backbone.stem_exit)
        self.downsample_dq_h0 = b0.backbone.stem_dq3

        self.neck = QNeck1(b0.neck)
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

        #y = self.neck((x0,))
        y = self.neck(x0)
        det1, = self.head(y)
        return det1, x0, x0h1, x0h2


class YoloBlock1(torch.nn.Module):
    def __init__(self, full, b3, b2, b1, b0):
        super().__init__()
        self.backbone_q = full.backbone.stage1_q
        self.backbone = torch.nn.Sequential(*[
                QBaseConv(full.backbone.stage1[0]),
                QCSPLayer(full.backbone.stage1[1]),
        ])
        self.backbone_dq_fwd = full.backbone.stage1_dq
        self.downsample_b1h3 = QTransition(b1.backbone.block1_exit0)
        self.downsample_dq_h3 = b1.backbone.block1_dq1
        self.downsample_b1h2 = QTransition(b1.backbone.block1_exit1)
        self.downsample_dq_h2 = b1.backbone.block1_dq2
        self.downsample_b1h1 = QTransition(b1.backbone.block1_exit)
        self.downsample_dq_h1 = b1.backbone.block1_dq3

        self.neck = QNeck2(b1.neck)
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
        self.backbone = torch.nn.Sequential(*[
            QTransition(full.backbone.stage2[0]),
            QCSPLayer(full.backbone.stage2[1]),
        ])
        self.backbone_dq_fwd = full.backbone.stage2_dq
        self.downsample_b2h3 = QTransition(b2.backbone.block2_exit0)
        self.downsample_dq_h3 = b2.backbone.block2_dq1
        self.downsample_b2h2 = QTransition(b2.backbone.block2_exit)
        self.downsample_dq_h2 = b2.backbone.block2_dq2

        self.neck = QNeck(b2.neck)
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
        self.backbone = torch.nn.Sequential(*[
            QTransition(full.backbone.stage3[0]),
            QCSPLayer(full.backbone.stage3[1]),
        ])
        self.backbone_dq_fwd = full.backbone.stage3_dq
        self.downsample_b3 = QTransition(b3.backbone.block3_exit)
        self.downsample_dq_h3 = b3.backbone.block3_dq1

        self.neck = QNeck(b3.neck)
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
        self.backbone = torch.nn.Sequential(*[
            QTransition(full.backbone.stage4[0]),
            QSPPBottleneck(full.backbone.stage4[1]),
            QCSPLayer(full.backbone.stage4[2]),
        ])
        self.backbone_dq = full.backbone.stage4_dq

        self.neck = QNeck(full.neck)
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
    block0_fuse_list = [
        ['backbone.0.conv', 'backbone.0.norm'],
        ['backbone.1.conv', 'backbone.1.norm'],
        ['backbone.2.conv', 'backbone.2.norm'],
        ['downsample_b0h3.conv1.conv', 'downsample_b0h3.conv1.norm'],
        ['downsample_b0h3.conv2.conv', 'downsample_b0h3.conv2.norm'],
        ['downsample_b0h3.conv3.conv', 'downsample_b0h3.conv3.norm'],
        ['downsample_b0h2.conv1.conv', 'downsample_b0h2.conv1.norm'],
        ['downsample_b0h2.conv2.conv', 'downsample_b0h2.conv2.norm'],
        ['downsample_b0h2.conv3.conv', 'downsample_b0h2.conv3.norm'],
        ['downsample_b0h1.conv1.conv', 'downsample_b0h1.conv1.norm'],
        ['downsample_b0h1.conv2.conv', 'downsample_b0h1.conv2.norm'],
        ['downsample_b0h1.conv3.conv', 'downsample_b0h1.conv3.norm'],
        ['downsample_b0h0.conv1.conv', 'downsample_b0h0.conv1.norm'],
        ['downsample_b0h0.conv2.conv', 'downsample_b0h0.conv2.norm'],
        ['downsample_b0h0.conv3.conv', 'downsample_b0h0.conv3.norm'],
        ['neck.n3.conv', 'neck.n3.norm'],
    ]
    block0 = torch.quantization.fuse_modules(block0, block0_fuse_list)
    block1_fuse_list = [
        ['backbone.0.conv', 'backbone.0.norm'],
        ['backbone.1.conv1.conv', 'backbone.1.conv1.norm'],
        ['backbone.1.conv2.conv', 'backbone.1.conv2.norm'],
        ['backbone.1.conv3.0.conv', 'backbone.1.conv3.0.norm'],
        ['backbone.1.conv3.1.conv', 'backbone.1.conv3.1.norm'],
        ['backbone.1.conv4.0.conv', 'backbone.1.conv4.0.norm'],
        ['backbone.1.conv4.1.conv', 'backbone.1.conv4.1.norm'],
        ['backbone.1.conv5.conv', 'backbone.1.conv5.norm'],
        ['downsample_b1h3.conv1.conv', 'downsample_b1h3.conv1.norm'],
        ['downsample_b1h3.conv2.conv', 'downsample_b1h3.conv2.norm'],
        ['downsample_b1h3.conv3.conv', 'downsample_b1h3.conv3.norm'],
        ['downsample_b1h2.conv1.conv', 'downsample_b1h2.conv1.norm'],
        ['downsample_b1h2.conv2.conv', 'downsample_b1h2.conv2.norm'],
        ['downsample_b1h2.conv3.conv', 'downsample_b1h2.conv3.norm'],
        ['downsample_b1h1.conv1.conv', 'downsample_b1h1.conv1.norm'],
        ['downsample_b1h1.conv2.conv', 'downsample_b1h1.conv2.norm'],
        ['downsample_b1h1.conv3.conv', 'downsample_b1h1.conv3.norm'],
        ['neck.conv_for_C4.conv', 'neck.conv_for_C4.norm'],
        ['neck.p5_p4.conv1.conv', 'neck.p5_p4.conv1.norm'],
        ['neck.p5_p4.conv2.conv', 'neck.p5_p4.conv2.norm'],
        ['neck.p5_p4.conv3.conv', 'neck.p5_p4.conv3.norm'],
        ['neck.p5_p4.conv4.0.conv', 'neck.p5_p4.conv4.0.norm'],
        ['neck.p5_p4.conv4.1.conv', 'neck.p5_p4.conv4.1.norm'],
        ['neck.p5_p4.conv4.2.conv', 'neck.p5_p4.conv4.2.norm'],
        ['neck.p5_p4.conv4.3.conv', 'neck.p5_p4.conv4.3.norm'],
        ['neck.conv_for_P4.conv', 'neck.conv_for_P4.norm'],
        ['neck.conv_for_C3.conv', 'neck.conv_for_C3.norm'],
        ['neck.p4_p3.conv1.conv', 'neck.p4_p3.conv1.norm'],
        ['neck.p4_p3.conv2.conv', 'neck.p4_p3.conv2.norm'],
        ['neck.p4_p3.conv3.conv', 'neck.p4_p3.conv3.norm'],
        ['neck.p4_p3.conv4.0.conv', 'neck.p4_p3.conv4.0.norm'],
        ['neck.p4_p3.conv4.1.conv', 'neck.p4_p3.conv4.1.norm'],
        ['neck.p4_p3.conv4.2.conv', 'neck.p4_p3.conv4.2.norm'],
        ['neck.p4_p3.conv4.3.conv', 'neck.p4_p3.conv4.3.norm'],
        ['neck.p4_p3.conv5.conv', 'neck.p4_p3.conv5.norm'],
        ['neck.downsample_conv1.conv1.conv', 'neck.downsample_conv1.conv1.norm'],
        ['neck.downsample_conv1.conv2.conv', 'neck.downsample_conv1.conv2.norm'],
        ['neck.downsample_conv1.conv3.conv', 'neck.downsample_conv1.conv3.norm'],
        ['neck.n3_n4.conv1.conv', 'neck.n3_n4.conv1.norm'],
        ['neck.n3_n4.conv2.conv', 'neck.n3_n4.conv2.norm'],
        ['neck.n3_n4.conv3.conv', 'neck.n3_n4.conv3.norm'],
        ['neck.n3_n4.conv4.0.conv', 'neck.n3_n4.conv4.0.norm'],
        ['neck.n3_n4.conv4.1.conv', 'neck.n3_n4.conv4.1.norm'],
        ['neck.n3_n4.conv4.2.conv', 'neck.n3_n4.conv4.2.norm'],
        ['neck.n3_n4.conv4.3.conv', 'neck.n3_n4.conv4.3.norm'],
        ['neck.n3_n4.conv5.conv', 'neck.n3_n4.conv5.norm'],
        ['neck.downsample_conv2.conv1.conv', 'neck.downsample_conv2.conv1.norm'],
        ['neck.downsample_conv2.conv2.conv', 'neck.downsample_conv2.conv2.norm'],
        ['neck.downsample_conv2.conv3.conv', 'neck.downsample_conv2.conv3.norm'],
        ['neck.n3.conv', 'neck.n3.norm'],
        ['neck.n4.conv', 'neck.n4.norm'],
    ]
    block1 = torch.quantization.fuse_modules(block1, block1_fuse_list)
    block2_fuse_list = [
        ['backbone.0.conv1.conv', 'backbone.0.conv1.norm'],
        ['backbone.0.conv2.conv', 'backbone.0.conv2.norm'],
        ['backbone.0.conv3.conv', 'backbone.0.conv3.norm'],
        ['backbone.1.conv1.conv', 'backbone.1.conv1.norm'],
        ['backbone.1.conv2.conv', 'backbone.1.conv2.norm'],
        ['backbone.1.conv3.0.conv', 'backbone.1.conv3.0.norm'],
        ['backbone.1.conv3.1.conv', 'backbone.1.conv3.1.norm'],
        ['backbone.1.conv4.0.conv', 'backbone.1.conv4.0.norm'],
        ['backbone.1.conv4.1.conv', 'backbone.1.conv4.1.norm'],
        ['backbone.1.conv5.conv', 'backbone.1.conv5.norm'],
        ['downsample_b2h3.conv1.conv', 'downsample_b2h3.conv1.norm'],
        ['downsample_b2h3.conv2.conv', 'downsample_b2h3.conv2.norm'],
        ['downsample_b2h3.conv3.conv', 'downsample_b2h3.conv3.norm'],
        ['downsample_b2h2.conv1.conv', 'downsample_b2h2.conv1.norm'],
        ['downsample_b2h2.conv2.conv', 'downsample_b2h2.conv2.norm'],
        ['downsample_b2h2.conv3.conv', 'downsample_b2h2.conv3.norm'],
        ['neck.spp.cv1.conv', 'neck.spp.cv1.norm'],
        ['neck.spp.cv2.conv', 'neck.spp.cv2.norm'],
        ['neck.spp.cv3.conv', 'neck.spp.cv3.norm'],
        ['neck.spp.cv4.conv', 'neck.spp.cv4.norm'],
        ['neck.spp.cv5.conv', 'neck.spp.cv5.norm'],
        ['neck.spp.cv6.conv', 'neck.spp.cv6.norm'],
        ['neck.spp.cv7.conv', 'neck.spp.cv7.norm'],
        ['neck.conv_for_P5.conv', 'neck.conv_for_P5.norm'],
        ['neck.conv_for_C4.conv', 'neck.conv_for_C4.norm'],
        ['neck.p5_p4.conv1.conv', 'neck.p5_p4.conv1.norm'],
        ['neck.p5_p4.conv2.conv', 'neck.p5_p4.conv2.norm'],
        ['neck.p5_p4.conv3.conv', 'neck.p5_p4.conv3.norm'],
        ['neck.p5_p4.conv4.0.conv', 'neck.p5_p4.conv4.0.norm'],
        ['neck.p5_p4.conv4.1.conv', 'neck.p5_p4.conv4.1.norm'],
        ['neck.p5_p4.conv4.2.conv', 'neck.p5_p4.conv4.2.norm'],
        ['neck.p5_p4.conv4.3.conv', 'neck.p5_p4.conv4.3.norm'],
        ['neck.p5_p4.conv5.conv', 'neck.p5_p4.conv5.norm'],
        ['neck.conv_for_P4.conv', 'neck.conv_for_P4.norm'],
        ['neck.conv_for_C3.conv', 'neck.conv_for_C3.norm'],
        ['neck.p4_p3.conv1.conv', 'neck.p4_p3.conv1.norm'],
        ['neck.p4_p3.conv2.conv', 'neck.p4_p3.conv2.norm'],
        ['neck.p4_p3.conv3.conv', 'neck.p4_p3.conv3.norm'],
        ['neck.p4_p3.conv4.0.conv', 'neck.p4_p3.conv4.0.norm'],
        ['neck.p4_p3.conv4.1.conv', 'neck.p4_p3.conv4.1.norm'],
        ['neck.p4_p3.conv4.2.conv', 'neck.p4_p3.conv4.2.norm'],
        ['neck.p4_p3.conv4.3.conv', 'neck.p4_p3.conv4.3.norm'],
        ['neck.p4_p3.conv5.conv', 'neck.p4_p3.conv5.norm'],
        ['neck.downsample_conv1.conv1.conv', 'neck.downsample_conv1.conv1.norm'],
        ['neck.downsample_conv1.conv2.conv', 'neck.downsample_conv1.conv2.norm'],
        ['neck.downsample_conv1.conv3.conv', 'neck.downsample_conv1.conv3.norm'],
        ['neck.n3_n4.conv1.conv', 'neck.n3_n4.conv1.norm'],
        ['neck.n3_n4.conv2.conv', 'neck.n3_n4.conv2.norm'],
        ['neck.n3_n4.conv3.conv', 'neck.n3_n4.conv3.norm'],
        ['neck.n3_n4.conv4.0.conv', 'neck.n3_n4.conv4.0.norm'],
        ['neck.n3_n4.conv4.1.conv', 'neck.n3_n4.conv4.1.norm'],
        ['neck.n3_n4.conv4.2.conv', 'neck.n3_n4.conv4.2.norm'],
        ['neck.n3_n4.conv4.3.conv', 'neck.n3_n4.conv4.3.norm'],
        ['neck.n3_n4.conv5.conv', 'neck.n3_n4.conv5.norm'],
        ['neck.downsample_conv2.conv1.conv', 'neck.downsample_conv2.conv1.norm'],
        ['neck.downsample_conv2.conv2.conv', 'neck.downsample_conv2.conv2.norm'],
        ['neck.downsample_conv2.conv3.conv', 'neck.downsample_conv2.conv3.norm'],
        ['neck.n4_n5.conv1.conv', 'neck.n4_n5.conv1.norm'],
        ['neck.n4_n5.conv2.conv', 'neck.n4_n5.conv2.norm'],
        ['neck.n4_n5.conv3.conv', 'neck.n4_n5.conv3.norm'],
        ['neck.n4_n5.conv4.0.conv', 'neck.n4_n5.conv4.0.norm'],
        ['neck.n4_n5.conv4.1.conv', 'neck.n4_n5.conv4.1.norm'],
        ['neck.n4_n5.conv4.2.conv', 'neck.n4_n5.conv4.2.norm'],
        ['neck.n4_n5.conv4.3.conv', 'neck.n4_n5.conv4.3.norm'],
        ['neck.n4_n5.conv5.conv', 'neck.n4_n5.conv5.norm'],
        ['neck.n3.conv', 'neck.n3.norm'],
        ['neck.n4.conv', 'neck.n4.norm'],
        ['neck.n5.conv', 'neck.n5.norm'],
    ]
    block2 = torch.quantization.fuse_modules(block2, block2_fuse_list)
    block3_fuse_list = [
        ['backbone.0.conv1.conv', 'backbone.0.conv1.norm'],
        ['backbone.0.conv2.conv', 'backbone.0.conv2.norm'],
        ['backbone.0.conv3.conv', 'backbone.0.conv3.norm'],
        ['backbone.1.conv1.conv', 'backbone.1.conv1.norm'],
        ['backbone.1.conv2.conv', 'backbone.1.conv2.norm'],
        ['backbone.1.conv3.0.conv', 'backbone.1.conv3.0.norm'],
        ['backbone.1.conv3.1.conv', 'backbone.1.conv3.1.norm'],
        ['backbone.1.conv4.0.conv', 'backbone.1.conv4.0.norm'],
        ['backbone.1.conv4.1.conv', 'backbone.1.conv4.1.norm'],
        ['backbone.1.conv5.conv', 'backbone.1.conv5.norm'],
        ['downsample_b3.conv1.conv', 'downsample_b3.conv1.norm'],
        ['downsample_b3.conv2.conv', 'downsample_b3.conv2.norm'],
        ['downsample_b3.conv3.conv', 'downsample_b3.conv3.norm'],
        ['neck.spp.cv1.conv', 'neck.spp.cv1.norm'],
        ['neck.spp.cv2.conv', 'neck.spp.cv2.norm'],
        ['neck.spp.cv3.conv', 'neck.spp.cv3.norm'],
        ['neck.spp.cv4.conv', 'neck.spp.cv4.norm'],
        ['neck.spp.cv5.conv', 'neck.spp.cv5.norm'],
        ['neck.spp.cv6.conv', 'neck.spp.cv6.norm'],
        ['neck.spp.cv7.conv', 'neck.spp.cv7.norm'],
        ['neck.conv_for_P5.conv', 'neck.conv_for_P5.norm'],
        ['neck.conv_for_C4.conv', 'neck.conv_for_C4.norm'],
        ['neck.p5_p4.conv1.conv', 'neck.p5_p4.conv1.norm'],
        ['neck.p5_p4.conv2.conv', 'neck.p5_p4.conv2.norm'],
        ['neck.p5_p4.conv3.conv', 'neck.p5_p4.conv3.norm'],
        ['neck.p5_p4.conv4.0.conv', 'neck.p5_p4.conv4.0.norm'],
        ['neck.p5_p4.conv4.1.conv', 'neck.p5_p4.conv4.1.norm'],
        ['neck.p5_p4.conv4.2.conv', 'neck.p5_p4.conv4.2.norm'],
        ['neck.p5_p4.conv4.3.conv', 'neck.p5_p4.conv4.3.norm'],
        ['neck.p5_p4.conv5.conv', 'neck.p5_p4.conv5.norm'],
        ['neck.conv_for_P4.conv', 'neck.conv_for_P4.norm'],
        ['neck.conv_for_C3.conv', 'neck.conv_for_C3.norm'],
        ['neck.p4_p3.conv1.conv', 'neck.p4_p3.conv1.norm'],
        ['neck.p4_p3.conv2.conv', 'neck.p4_p3.conv2.norm'],
        ['neck.p4_p3.conv3.conv', 'neck.p4_p3.conv3.norm'],
        ['neck.p4_p3.conv4.0.conv', 'neck.p4_p3.conv4.0.norm'],
        ['neck.p4_p3.conv4.1.conv', 'neck.p4_p3.conv4.1.norm'],
        ['neck.p4_p3.conv4.2.conv', 'neck.p4_p3.conv4.2.norm'],
        ['neck.p4_p3.conv4.3.conv', 'neck.p4_p3.conv4.3.norm'],
        ['neck.p4_p3.conv5.conv', 'neck.p4_p3.conv5.norm'],
        ['neck.downsample_conv1.conv1.conv', 'neck.downsample_conv1.conv1.norm'],
        ['neck.downsample_conv1.conv2.conv', 'neck.downsample_conv1.conv2.norm'],
        ['neck.downsample_conv1.conv3.conv', 'neck.downsample_conv1.conv3.norm'],
        ['neck.n3_n4.conv1.conv', 'neck.n3_n4.conv1.norm'],
        ['neck.n3_n4.conv2.conv', 'neck.n3_n4.conv2.norm'],
        ['neck.n3_n4.conv3.conv', 'neck.n3_n4.conv3.norm'],
        ['neck.n3_n4.conv4.0.conv', 'neck.n3_n4.conv4.0.norm'],
        ['neck.n3_n4.conv4.1.conv', 'neck.n3_n4.conv4.1.norm'],
        ['neck.n3_n4.conv4.2.conv', 'neck.n3_n4.conv4.2.norm'],
        ['neck.n3_n4.conv4.3.conv', 'neck.n3_n4.conv4.3.norm'],
        ['neck.n3_n4.conv5.conv', 'neck.n3_n4.conv5.norm'],
        ['neck.downsample_conv2.conv1.conv', 'neck.downsample_conv2.conv1.norm'],
        ['neck.downsample_conv2.conv2.conv', 'neck.downsample_conv2.conv2.norm'],
        ['neck.downsample_conv2.conv3.conv', 'neck.downsample_conv2.conv3.norm'],
        ['neck.n4_n5.conv1.conv', 'neck.n4_n5.conv1.norm'],
        ['neck.n4_n5.conv2.conv', 'neck.n4_n5.conv2.norm'],
        ['neck.n4_n5.conv3.conv', 'neck.n4_n5.conv3.norm'],
        ['neck.n4_n5.conv4.0.conv', 'neck.n4_n5.conv4.0.norm'],
        ['neck.n4_n5.conv4.1.conv', 'neck.n4_n5.conv4.1.norm'],
        ['neck.n4_n5.conv4.2.conv', 'neck.n4_n5.conv4.2.norm'],
        ['neck.n4_n5.conv4.3.conv', 'neck.n4_n5.conv4.3.norm'],
        ['neck.n4_n5.conv5.conv', 'neck.n4_n5.conv5.norm'],
        ['neck.n3.conv', 'neck.n3.norm'],
        ['neck.n4.conv', 'neck.n4.norm'],
        ['neck.n5.conv', 'neck.n5.norm'],
    ]
    block3 = torch.quantization.fuse_modules(block3, block3_fuse_list)
    #for name, param in block4.named_parameters():
    #    print(name)
    block4_fuse_list = [
        ['backbone.0.conv1.conv', 'backbone.0.conv1.norm'],
        ['backbone.0.conv2.conv', 'backbone.0.conv2.norm'],
        ['backbone.0.conv3.conv', 'backbone.0.conv3.norm'],
        ['backbone.1.conv1.conv', 'backbone.1.conv1.norm'],
        ['backbone.1.conv2.conv', 'backbone.1.conv2.norm'],
        ['backbone.2.conv1.conv', 'backbone.2.conv1.norm'],
        ['backbone.2.conv2.conv', 'backbone.2.conv2.norm'],
        ['backbone.2.conv3.0.conv', 'backbone.2.conv3.0.norm'],
        ['backbone.2.conv3.1.conv', 'backbone.2.conv3.1.norm'],
        ['backbone.2.conv4.0.conv', 'backbone.2.conv4.0.norm'],
        ['backbone.2.conv4.1.conv', 'backbone.2.conv4.1.norm'],
        ['backbone.2.conv5.conv', 'backbone.2.conv5.norm'],
        ['neck.spp.cv1.conv', 'neck.spp.cv1.norm'],
        ['neck.spp.cv2.conv', 'neck.spp.cv2.norm'],
        ['neck.spp.cv3.conv', 'neck.spp.cv3.norm'],
        ['neck.spp.cv4.conv', 'neck.spp.cv4.norm'],
        ['neck.spp.cv5.conv', 'neck.spp.cv5.norm'],
        ['neck.spp.cv6.conv', 'neck.spp.cv6.norm'],
        ['neck.spp.cv7.conv', 'neck.spp.cv7.norm'],
        ['neck.conv_for_P5.conv', 'neck.conv_for_P5.norm'],
        ['neck.conv_for_C4.conv', 'neck.conv_for_C4.norm'],
        ['neck.p5_p4.conv1.conv', 'neck.p5_p4.conv1.norm'],
        ['neck.p5_p4.conv2.conv', 'neck.p5_p4.conv2.norm'],
        ['neck.p5_p4.conv3.conv', 'neck.p5_p4.conv3.norm'],
        ['neck.p5_p4.conv4.0.conv', 'neck.p5_p4.conv4.0.norm'],
        ['neck.p5_p4.conv4.1.conv', 'neck.p5_p4.conv4.1.norm'],
        ['neck.p5_p4.conv4.2.conv', 'neck.p5_p4.conv4.2.norm'],
        ['neck.p5_p4.conv4.3.conv', 'neck.p5_p4.conv4.3.norm'],
        ['neck.p5_p4.conv5.conv', 'neck.p5_p4.conv5.norm'],
        ['neck.conv_for_P4.conv', 'neck.conv_for_P4.norm'],
        ['neck.conv_for_C3.conv', 'neck.conv_for_C3.norm'],
        ['neck.p4_p3.conv1.conv', 'neck.p4_p3.conv1.norm'],
        ['neck.p4_p3.conv2.conv', 'neck.p4_p3.conv2.norm'],
        ['neck.p4_p3.conv3.conv', 'neck.p4_p3.conv3.norm'],
        ['neck.p4_p3.conv4.0.conv', 'neck.p4_p3.conv4.0.norm'],
        ['neck.p4_p3.conv4.1.conv', 'neck.p4_p3.conv4.1.norm'],
        ['neck.p4_p3.conv4.2.conv', 'neck.p4_p3.conv4.2.norm'],
        ['neck.p4_p3.conv4.3.conv', 'neck.p4_p3.conv4.3.norm'],
        ['neck.p4_p3.conv5.conv', 'neck.p4_p3.conv5.norm'],
        ['neck.downsample_conv1.conv1.conv', 'neck.downsample_conv1.conv1.norm'],
        ['neck.downsample_conv1.conv2.conv', 'neck.downsample_conv1.conv2.norm'],
        ['neck.downsample_conv1.conv3.conv', 'neck.downsample_conv1.conv3.norm'],
        ['neck.n3_n4.conv1.conv', 'neck.n3_n4.conv1.norm'],
        ['neck.n3_n4.conv2.conv', 'neck.n3_n4.conv2.norm'],
        ['neck.n3_n4.conv3.conv', 'neck.n3_n4.conv3.norm'],
        ['neck.n3_n4.conv4.0.conv', 'neck.n3_n4.conv4.0.norm'],
        ['neck.n3_n4.conv4.1.conv', 'neck.n3_n4.conv4.1.norm'],
        ['neck.n3_n4.conv4.2.conv', 'neck.n3_n4.conv4.2.norm'],
        ['neck.n3_n4.conv4.3.conv', 'neck.n3_n4.conv4.3.norm'],
        ['neck.n3_n4.conv5.conv', 'neck.n3_n4.conv5.norm'],
        ['neck.downsample_conv2.conv1.conv', 'neck.downsample_conv2.conv1.norm'],
        ['neck.downsample_conv2.conv2.conv', 'neck.downsample_conv2.conv2.norm'],
        ['neck.downsample_conv2.conv3.conv', 'neck.downsample_conv2.conv3.norm'],
        ['neck.n4_n5.conv1.conv', 'neck.n4_n5.conv1.norm'],
        ['neck.n4_n5.conv2.conv', 'neck.n4_n5.conv2.norm'],
        ['neck.n4_n5.conv3.conv', 'neck.n4_n5.conv3.norm'],
        ['neck.n4_n5.conv4.0.conv', 'neck.n4_n5.conv4.0.norm'],
        ['neck.n4_n5.conv4.1.conv', 'neck.n4_n5.conv4.1.norm'],
        ['neck.n4_n5.conv4.2.conv', 'neck.n4_n5.conv4.2.norm'],
        ['neck.n4_n5.conv4.3.conv', 'neck.n4_n5.conv4.3.norm'],
        ['neck.n4_n5.conv5.conv', 'neck.n4_n5.conv5.norm'],
        ['neck.n3.conv', 'neck.n3.norm'],
        ['neck.n4.conv', 'neck.n4.norm'],
        ['neck.n5.conv', 'neck.n5.norm'],
    ]
    block4 = torch.quantization.fuse_modules(block4, block4_fuse_list)

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
    block2 = torch.quantization.prepare(block2)
    block3 = torch.quantization.prepare(block3)
    block4 = torch.quantization.prepare(block4)

    # Calibrate with the training set
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize((416, 416))
    ])
    cal_data = torchvision.datasets.ImageFolder(
        cal_set,
        transform=transforms,
        is_valid_file=lambda x: True if x.endswith('.png') else False
    )
    data_loader = torch.utils.data.DataLoader(cal_data, batch_size=16, shuffle=True)
    with torch.no_grad():
        count = 0
        for x, _ in data_loader:
            print(f'Calibration Batch: {count}')
            count += 1
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

def save_gpu_blocks(blocks):
    blocks = [b.to(torch.device('cuda')).eval() for b in blocks]
    block0, block1, block2, block3, block4 = blocks
    x = torch.randn(1,3,416,416).to(torch.device('cuda'))
    _, x0, x0h1, x0h2 = block0(x)
    _, _, x0h2, x1, x1h2, x1h3 = block1(x0, x0h1, x0h2)
    _, _, _, x1h3, x2, x2h3 = block2(x0h2, x1, x1h2, x1h3)
    _, _, _, x2, x3 = block3(x1h3, x2, x2h3)
    torch.jit.save(torch.jit.trace(block0, (x)), 'yoloblock0.pt')
    torch.jit.save(torch.jit.trace(block1, (x0, x0h1, x0h2)), 'yoloblock1.pt')
    torch.jit.save(torch.jit.trace(block2, (x0h2, x1, x1h2, x1h3)), 'yoloblock2.pt')
    torch.jit.save(torch.jit.trace(block3, (x1h3, x2, x2h3)), 'yoloblock3.pt')
    torch.jit.save(torch.jit.trace(block4, (x2, x3)), 'yoloblock4.pt')


def save_cpu_blocks(blocks):
    blocks = [b.to(torch.device('cpu')).eval() for b in blocks]
    block0, block1, block2, block3, block4 = blocks
    x = torch.randn(1,3,416,416).to(torch.device('cpu'))
    _, x0, x0h1, x0h2 = block0(x)
    _, _, x0h2, x1, x1h2, x1h3 = block1(x0, x0h1, x0h2)
    _, _, _, x1h3, x2, x2h3 = block2(x0h2, x1, x1h2, x1h3)
    _, _, _, x2, x3 = block3(x1h3, x2, x2h3)
    torch.jit.save(torch.jit.trace(block0, (x)), 'yoloblock0q.pt')
    torch.jit.save(torch.jit.trace(block1, (x0, x0h1, x0h2)), 'yoloblock1q.pt')
    torch.jit.save(torch.jit.trace(block2, (x0h2, x1, x1h2, x1h3)), 'yoloblock2q.pt')
    torch.jit.save(torch.jit.trace(block3, (x1h3, x2, x2h3)), 'yoloblock3q.pt')
    torch.jit.save(torch.jit.trace(block4, (x2, x3)), 'yoloblock4q.pt')


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
    print('Splitting model...')
    blocks = split_model(
        args.full,
        args.b3,
        args.b2,
        args.b1,
        args.b0,
    )
    print('Saving blocks...')
    for idx, block in enumerate(blocks):
        for param in block.modules():
            if isinstance(param, models.layers.network_blocks.BaseConv):
                param._should_prevent_trainer_and_dataloaders_deepcopy = False
            if isinstance(param, models.backbones.eelan_b0.Transition):
                param._should_prevent_trainer_and_dataloaders_deepcopy = False
            if isinstance(param, models.backbones.eelan_b1.Transition):
                param._should_prevent_trainer_and_dataloaders_deepcopy = False
            if isinstance(param, models.backbones.eelan_b2.Transition):
                param._should_prevent_trainer_and_dataloaders_deepcopy = False
            if isinstance(param, models.backbones.eelan_b3.Transition):
                param._should_prevent_trainer_and_dataloaders_deepcopy = False
            if isinstance(param, models.backbones.eelan_full.Transition):
                param._should_prevent_trainer_and_dataloaders_deepcopy = False
            if isinstance(param, models.necks.yolov7_neck2.CSPLayer):
                param._should_prevent_trainer_and_dataloaders_deepcopy = False
            if isinstance(param, models.necks.yolov7_neck.CSPLayer):
                param._should_prevent_trainer_and_dataloaders_deepcopy = False
            if isinstance(param, models.backbones.eelan_full.CSPLayer):
                param._should_prevent_trainer_and_dataloaders_deepcopy = False
            if isinstance(param, models.layers.network_blocks.SPPCSPC):
                param._should_prevent_trainer_and_dataloaders_deepcopy = False
            if isinstance(param, models.layers.network_blocks.SPPBottleneck):
                param._should_prevent_trainer_and_dataloaders_deepcopy = False
        torch.save(block, f'yoloblock{idx}.pt')
    print('Quantizing...')
    q_blocks = static_quantize(args.calset, blocks)
    print('Saving quantized blocks...')
    for idx, block in enumerate(blocks):
        torch.save(block, f'yoloblock{idx}q.pt')
    #model_scripted = torch.jit.script(model) # Export to TorchScript
    #model_scripted.save('model_scripted.pt') # Save