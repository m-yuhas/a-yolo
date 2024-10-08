import torch.nn as nn
# backbones
from models.backbones.darknet_csp import CSPDarkNet
from models.backbones.mobilenext_csp import CSPMobileNext
from models.backbones.eelan import EELAN
from models.backbones.eelan_full import EELANFull
from models.backbones.eelan_b3  import EELANBlock3
from models.backbones.eelan_b2 import EELANBlock2
from models.backbones.eelan_b1 import EELANBlock1
from models.backbones.eelan_b0 import EELANBlock0
from models.backbones.ecmnet import ECMNet
from models.backbones.shufflenetv2 import ShuffleNetV2_Plus
from models.backbones.mobilenetv3 import MobileNetV3_Small, MobileNetV3_Large
from models.backbones.vision_transformer.vision_transformer import VisionTransformer
from models.backbones.vision_transformer.swin_transformer import SwinTransformer
from models.backbones.vggnet import VGGNET
# necks
from models.necks.pafpn_csp import CSPPAFPN
from models.necks.pafpn_al import AL_PAFPN
from models.necks.yolov7_neck import YOLOv7NECK
from models.necks.yolov7_neck1 import YOLOv7NECK1
from models.necks.yolov7_neck2 import YOLOv7NECK2
from models.necks.ssd_neck import SSDNECK
# heads
from models.heads.decoupled_head import DecoupledHead
from models.heads.implicit_head import ImplicitHead
from models.heads.ssd_head import SsdHead
# heads
from models.heads.decoupled_head import DecoupledHead
from models.heads.implicit_head import ImplicitHead
# loss
from models.losses.yolox.yolox_loss import YOLOXLoss
from models.losses.yolov7.yolov7_loss import YOLOv7Loss


def build_model(cfg_models, num_classes):
    cb = cfg_models['backbone']
    cn = [neck for neck in cfg_models['neck']]
    ch = [head for head in cfg_models['head']]
    cl = cfg_models['loss']

    backbone = eval(cb['name'])(cb)
    neck = [eval(n['name'])(n) if n is not None else None for n in cn]
    head = [eval(h['name'])(h, num_classes) if h is not None else None for h in ch]
    loss = eval(cl['name'])(cl, num_classes)
    model = OneStageD(backbone, neck, head, loss)
    return model


class OneStageD(nn.Module):

    def __init__(self, backbone=None, neck=None, head=None, loss=None):
        super().__init__()
        self.backbone = backbone
        self.neck = neck
        self.head = head
        self.loss = loss

    def forward(self, x, labels=None):
        x = self.backbone(x)
        y = []
        for idx, n in enumerate(self.neck):
            if n is not None:
                y.append(n(x[idx - len(n.in_channels) + 1:idx + 1]))
            else:
                y.append(x[idx])
        z = []
        for idx, h in enumerate(self.head):
            if h is not None:
                print('#######')
                print(len(x[idx - len(n.in_channels):idx]))
                print('#######')
                z.append(h(y[idx - len(h.in_channels):idx]))
        if labels is not None:
            w = self.loss(z, labels)
        return w


# Backbones
def cspdarknet(cfg):
    backbone = CSPDarkNet(cfg['depths'], cfg['channels'], cfg['outputs'], cfg['norm'], cfg['act'])
    return backbone


def cspmobilenext(cfg):
    backbone = CSPMobileNext(cfg['depths'], cfg['channels'], cfg['outputs'], cfg['norm'], cfg['act'])
    return backbone


def eelan(cfg):
    backbone = EELAN(cfg['depths'], cfg['channels'], cfg['outputs'], cfg['norm'], cfg['act'])
    return backbone

def eelan_full(cfg):
    backbone = EELANFull(cfg['depths'], cfg['channels'], cfg['outputs'], cfg['norm'], cfg['act'])
    return backbone

def eelan_b3(cfg):
    backbone = EELANBlock3(cfg['depths'], cfg['channels'], cfg['outputs'], cfg['norm'], cfg['act'], cfg['weights'])
    return backbone

def eelan_b2(cfg):
    backbone = EELANBlock2(cfg['depths'], cfg['channels'], cfg['outputs'], cfg['norm'], cfg['act'], cfg['weights'])
    return backbone

def eelan_b1(cfg):
    backbone = EELANBlock1(cfg['depths'], cfg['channels'], cfg['outputs'], cfg['norm'], cfg['act'], cfg['weights'])
    return backbone

def eelan_b0(cfg):
    backbone = EELANBlock0(cfg['depths'], cfg['channels'], cfg['outputs'], cfg['norm'], cfg['act'], cfg['weights'])
    return backbone


def vggnet(cfg):
    backbone = VGGNET(cfg['depths'], cfg['channels'], cfg['outputs'], cfg['norm'], cfg['act'])
    return backbone

def ecmnet(cfg):
    backbone = ECMNet(cfg['depths'], cfg['channels'], cfg['outputs'], cfg['norm'], cfg['act'])
    return backbone


def shufflenetv2(cfg):
    backbone = ShuffleNetV2_Plus(cfg['channels'], cfg['outputs'], cfg['norm'], cfg['act'])
    return backbone


def mobilenetv3s(cfg):
    backbone = MobileNetV3_Small(cfg['outputs'])
    return backbone


def mobilenetv3l(cfg):
    backbone = MobileNetV3_Large(cfg['outputs'])
    return backbone


def vision_transformer(cfg):
    backbone = VisionTransformer(patch_size=cfg['patch_size'], embed_dim=cfg['embed_dim'], depth=cfg['depth'],
                                 num_heads=cfg['num_heads'], mlp_ratio=cfg['mlp_ratio'])
    return backbone


def swin_transformer(cfg):
    backbone = SwinTransformer(embed_dim=cfg['embed_dim'], depths=cfg['depths'], num_heads=cfg['num_heads'],
                               window_size=cfg['window_size'], mlp_ratio=cfg['mlp_ratio'],
                               drop_path_rate=cfg['drop_path_rate'])
    return backbone


# Necks
def csppafpn(cfg):
    neck = CSPPAFPN(cfg['depths'], cfg['channels'], cfg['norm'], cfg['act'])
    return neck


def al_pafpn(cfg):
    neck = AL_PAFPN(cfg['depths'], cfg['channels'], cfg['norm'], cfg['act'])
    return neck


def yolov7neck(cfg):
    neck = YOLOv7NECK(cfg['depths'], cfg['channels'], cfg['norm'], cfg['act'])
    return neck

def yolov7neck1(cfg):
    neck = YOLOv7NECK1(cfg['depths'], cfg['channels'], cfg['norm'], cfg['act'])
    return neck

def yolov7neck2(cfg):
    neck = YOLOv7NECK2(cfg['depths'], cfg['channels'], cfg['norm'], cfg['act'])
    return neck

def ssdneck(cfg):
    neck = SSDNECK(cfg['depths'], cfg['channels'], cfg['norm'], cfg['act'])
    return neck


def none(cfg):
    return None


# Heads
def decoupled_head(cfg, num_classes):
    head = DecoupledHead(num_classes, cfg['num_anchor'], cfg['channels'], cfg['norm'], cfg['act'])
    return head


def implicit_head(cfg, num_classes):
    head = ImplicitHead(num_classes, cfg['num_anchor'], cfg['channels'])
    return head

def ssdhead(cfg, num_classes):
    head = SsdHead(num_classes, cfg['num_anchor'], cfg['channels'])
    return head


# Losses
def yolox(cfg, num_classes):
    head = YOLOXLoss(num_classes, cfg['stride'])
    return head


def yolov7(cfg, num_classes):
    head = YOLOv7Loss(num_classes, cfg['stride'], cfg['anchors'])
    return head
