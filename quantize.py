# Neeeded for Conversion
import argparse
import os
import cv2
import numpy
import onnx
import tensorflow
import torch
import torchvision
from typing import Callable, List, Tuple
from onnx_tf.backend import prepare
from PIL import Image


# Needed for YOLO
from pytorch_lightning import Trainer, seed_everything
from utils.defaults import train_argument_parser, load_config
from utils.build_data import build_data
from utils.build_logger import build_logger
from pytorch_lightning.callbacks import ModelCheckpoint
from PL_Modules.build_detection import build_model
from PL_Modules.pl_detection import LitDetection


class YoloBlock1(torch.nn.Module):
    def __init__(self, base_model):
        super().__init__()

        # BACKBONE
        self.stem = base_model.backbone.stem
        self.stem_exit0 = base_model.backbone.stem_exit0
        self.stem_exit1 = base_model.backbone.stem_exit1
        self.stem_exit2 = base_model.backbone.stem_exit
        
        self.block1 = base_model.backbone.block1
        self.block1_exit0 = base_model.backbone.block1_exit0
        self.block1_exit1 = base_model.backbone.block1_exit1
        self.block1_exit2 = base_model.backbone.block1_exit

        # NECK
        self.neck = base_model.neck

        #HEAD
        self.head = base_model.head


    def forward(self, x):
        s = self.stem(x)
        se0 = self.stem_exit0(s)
        se1 = self.stem_exit1(se0)
        se2 = self.stem_exit2(se1)
        b1 = self.block1(s)
        b1e0 = self.block1_exit0(b1)
        b1e1 = self.block1_exit1(b1e0)
        b1e2 = self.block1_exit2(b1e1)

        y = self.neck((se2, b1e2))
        det0, det1 = self.head(y)
        return det0, det1, b1, se1, b1e0, b1e1


class YoloBlock2(torch.nn.Module):
    def __init__(self, base_model):
        super().__init__()

        # BACKBONE
        self.block2 = base_model.backbone.block2
        self.block2_exit0 = base_model.backbone.block2_exit0
        self.block2_exit1 = base_model.backbone.block2_exit

        # NECK
        self.neck = base_model.neck

        # HEAD
        self.head = base_model.head

<<<<<<< HEAD
    def forward(self, x):
        b1, se1, b1e0, b1e1 = x
=======
    def forward(self, b1, se1, b1e0, b1e1):
>>>>>>> a3cc840dbfce3776d754b9d5d2a34f61790aa792
        b2 = self.block2(b1)
        b2e0 = self.block2_exit0(b2)
        b2e1 = self.block2_exit1(b2e0)

        y = self.neck((se1, b1e1, b2e1))
        det0, det1, det2 = self.head(y)
        return det0, det1, det2, b2, b1e0, b2e0

    
class YoloBlock3(torch.nn.Module):
    def __init__(self, base_model):
        super().__init__()

        # BACKBONE
        self.block3 = base_model.backbone.block3
        self.block3_exit0 = base_model.backbone.block3_exit
        
        # NECK
        self.neck = base_model.neck

        # HEAD
        self.head = base_model.head

<<<<<<< HEAD
    def forward(self, x):
        b2, b1e0, b2e0 = x
=======
    def forward(self, b2, b1e0, b2e0):
>>>>>>> a3cc840dbfce3776d754b9d5d2a34f61790aa792
        b3 = self.block3(b2)
        b3e0 = self.block3_exit0(b3)
        
        y = self.neck((b1e0, b2e0, b3e0))
        det0, det1, det2 = self.head(y)
        return det0, det1, det2, b2, b3


class YoloBlock4(torch.nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.block4 = base_model.backbone.stage4
        self.neck = base_model.neck
        self.head = base_model.head

<<<<<<< HEAD
    def forward(self, x):
        b2, b3 = x
=======
    def forward(self, b2, b3):
>>>>>>> a3cc840dbfce3776d754b9d5d2a34f61790aa792
        b4 = self.block4(b3)
        y = self.neck((b2, b3, b4))
        det0, det1, det2 = self.head(y)
        return det0, det1, det2

        

def split_model(block1: str, block2: str, block3: str, block4: str) -> List[torch.nn.Module]:
    return [
        YoloBlock1(torch.load(block1)),
        YoloBlock2(torch.load(block2)),
        YoloBlock3(torch.load(block3)),
        YoloBlock4(torch.load(block4)),
    ]


def block1_to_tf(block: torch.nn.Module, name: str, dummy_input: Tuple[int]):
    block.eval()
    det0, det1, b1, se1, b1e0, b1e1 = block(dummy_input)
    torch.onnx.export(
        block,                          # PyTorch Model
        torch.rand(dummy_input.shape),  # Input tensor
        name + ".onnx",                 # Output file (eg. 'output_model.onnx')
        opset_version=14,               # Operator support version
        input_names=['x'],              # Input tensor name (arbitary)
        output_names=['det0', 'det1', 'b1', 'se1', 'b1e0', 'b1e1']      # Output tensor name (arbitary)
    )
    onnx_model = onnx.load(name + ".onnx")
    tf_rep = prepare(onnx_model)
    tf_rep.export_graph(name + ".tf")
    return b1, se1, b1e0, b1e1

def block2_to_tf(block: torch.nn.Module, name: str, dummy_input: Tuple[int]):
    block.eval()
<<<<<<< HEAD
    det0, det1, det2, b2, b1e0, b2e0 = block(dummy_input)
=======
    det0, det1, det2, b2, b1e0, b2e0 = block(dummy_input[0], dummy_input[1], dummy_input[2], dummy_input[3])
>>>>>>> a3cc840dbfce3776d754b9d5d2a34f61790aa792
    torch.onnx.export(
        block,                          # PyTorch Model
        [torch.rand(dummy_input[i].shape) for i in range(len(dummy_input))],  # Input tensor
        name + ".onnx",                 # Output file (eg. 'output_model.onnx')
        opset_version=14,               # Operator support version
<<<<<<< HEAD
        input_names=['b1in', 'se1in', 'b1e0in', 'b1e1in'],              # Input tensor name (arbitary)
        output_names=['det0', 'det1', 'det2', 'b2out', 'b1e0out', 'b2e0out']      # Output tensor name (arbitary)
=======
        input_names=['b1', 'se1', 'b1e0', 'b1e1'],              # Input tensor name (arbitary)
        output_names=['det0', 'det1', 'det2', 'b2', 'b1e0', 'b2e0']      # Output tensor name (arbitary)
>>>>>>> a3cc840dbfce3776d754b9d5d2a34f61790aa792
    )
    onnx_model = onnx.load(name + ".onnx")
    tf_rep = prepare(onnx_model)
    tf_rep.export_graph(name + ".tf")
    return b2, b1e0, b2e0

def block3_to_tf(block: torch.nn.Module, name: str, dummy_input: Tuple[int]):
    block.eval()
<<<<<<< HEAD
    det0, det1, det2, b2, b3 = block(dummy_input)
=======
    det0, det1, det2, b2, b3 = block(dummy_input[0], dummy_input[1], dummy_input[2])
>>>>>>> a3cc840dbfce3776d754b9d5d2a34f61790aa792
    torch.onnx.export(
        block,                          # PyTorch Model
        [torch.rand(dummy_input[i].shape) for i in range(len(dummy_input))],  # Input tensor
        name + ".onnx",                 # Output file (eg. 'output_model.onnx')
        opset_version=14,               # Operator support version
<<<<<<< HEAD
        input_names=['b2in', 'b1e0in', 'b2e0in'],              # Input tensor name (arbitary)
        output_names=['det0', 'det1', 'det2', 'b2out', 'b3out']      # Output tensor name (arbitary)
=======
        input_names=['b2', 'b1e0', 'b2e0'],              # Input tensor name (arbitary)
        output_names=['det0', 'det1', 'det2', 'b2', 'b3']      # Output tensor name (arbitary)
>>>>>>> a3cc840dbfce3776d754b9d5d2a34f61790aa792
    )
    onnx_model = onnx.load(name + ".onnx")
    tf_rep = prepare(onnx_model)
    tf_rep.export_graph(name + ".tf")
    return b2, b3

def block4_to_tf(block: torch.nn.Module, name: str, dummy_input: Tuple[int]):
    block.eval()
    det0, det1, det2 = block(dummy_input[0], dummy_input[1], dummy_input[2])
    torch.onnx.export(
        block,
        [torch.rand(dummy_input[i].shape) for i in range(len(dummy_input))],
        name + ".onnx",
        opset_version=14,
        input_names=['b2in', 'b3in'],
        output_names=['det0', 'det1', 'det2']
    )
    onnx_model = onnx.load(name + ".onnx")
    tf_rep = prepare(onnx_model)
    tf_rep.export_graph(name + ".tf")
    return


def calibration_set(path: str, model_blocks: List[torch.nn.Module], stop_block: int):
    def rep_set():
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize((640, 640), antialias=False)
        ])
        for c in os.listdir(path):
            for f in os.listdir(os.path.join(path, c)):
                if not (f.endswith('.png') or f.endswith('.jpg')):
                    continue
                x = Image.open(os.path.join(path, c, f)).convert('RGB')
                x = transform(x)
                x = x.unsqueeze(0)
                if stop_block == 0:
                    r0 = x.detach().numpy().astype(numpy.float32)
                    yield [r0]
                _, _, b1, se1, b1e0, b1e1 = model_blocks[0](x)
                if stop_block == 1:
                    r0 = b1.detach().numpy().astype(numpy.float32)
                    r1 = se1.detach().numpy().astype(numpy.float32)
                    r2 = b1e0.detach().numpy().astype(numpy.float32)
                    r3 = b1e1.detach().numpy().astype(numpy.float32)
                    #yield [r0, r1, r2, r3]
                    yield [r3, r2, r1, r0]
                _, _, _, b2, b1e0, b2e0 = model_blocks[1]((b1, se1, b1e0, b1e1))
                if stop_block == 2:
                    r0 = b2.detach().numpy().astype(numpy.float32)
                    r1 = b1e0.detach().numpy().astype(numpy.float32)
                    r2 = b2e0.detach().numpy().astype(numpy.float32)
                    yield [r0, r1, r2]
                _, _, _, b2, b3 = model_blocks[2]((b2, b1e0, b2e0))
                if stop_block == 3:
                    r0 = b2.detach().numpy().astype(numpy.float32)
                    r1 = b3.detach().numpy().astype(numpy.float32)
                    yield [r0, r1]
    return rep_set


def tf_to_tfliteq(name: str, representative_dataset: Callable):
    converter = tensorflow.lite.TFLiteConverter.from_saved_model(name + ".tf")
    converter.optimizations = [tensorflow.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tensorflow.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tensorflow.int8
    converter.inference_output_type = tensorflow.int8
    tflite_quant_model = converter.convert()
    with open(name + ".tflite", 'wb') as f:
        f.write(tflite_quant_model)
    interpreter = tensorflow.lite.Interpreter(model_path=name + ".tflite")
    interpreter.allocate_tensors()


def static_quantize(block1: str, block2: str, block3: str, block4, calset: str) -> None:
    #dummy_input = torch.rand((1, 3, 224, 224))
    dummy_input = torch.rand((1, 3, 640, 640))
    blocks = split_model(block1, block2, block3, block4)

    print(f'### Working on block 1... ###')
    torch.save(blocks[0].state_dict(), f'yolov7block1_sd.pt')
    dummy_input = block1_to_tf(blocks[0], f'yolov7block1', dummy_input)
    tf_to_tfliteq(f'yolov7block1', calibration_set(calset, blocks, 0))       
        
    print(f'### Working on block 2... ###')
    torch.save(blocks[1].state_dict(), f'yolov7block2_sd.pt')
    dummy_input = block2_to_tf(blocks[1], f'yolov7block2', dummy_input)
    tf_to_tfliteq(f'yolov7block2', calibration_set(calset, blocks, 1))

    print(f'### Working on block 3... ###')
    torch.save(blocks[2].state_dict(), f'yolov7block3_sd.pt')
    block3_to_tf(blocks[2], f'yolov7block3', dummy_input)
    tf_to_tfliteq(f'yolov7block3', calibration_set(calset, blocks, 2))

    print(f'### Working on block 4... ###')
    torch.save(blocks[3].state_dict(), f'yolov7block4_sd.pt')
    block4_to_tf(blocks[3], f'yolov7block4', dummy_input)
    tf_to_tfliteq(f'yolov7block4', calibration_set(calset, blocks, 3))

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Quantize a Beta-VAE model')
    parser.add_argument(
        '--block1',
        help='Weights for block1',
    )
    parser.add_argument(
        '--block2',
        help='Weights for block2'
    )
    parser.add_argument(
        '--block3',
        help='Weights for block3'
    )
    parser.add_argument(
        '--block4',
        help='Weights for block4'
    )
    parser.add_argument(
        '--calset',
        help='Calibration set for static quantization'
    )
    args = parser.parse_args()
    static_quantize(
        args.block1,
        args.block2,
        args.block3,
        args.block4,
        args.calset
    )
