import torch
import torch.nn as nn
import pytorch_lightning


class SsdHead(pytorch_lightning.LightningModule):
    def __init__(
            self,
            num_classes,
            num_anchors,
            in_channels,
    ):
        super().__init__()
        self.n_anchors = num_anchors
        self.num_classes = num_classes
        ch = self.n_anchors * (5 + num_classes)

        self.conv = nn.ModuleList()
        #self.ia = nn.ModuleList()
        #self.im = nn.ModuleList()
        # For each feature map we go through different convolution.
        for i in range(len(in_channels)):
            #self.ia.append(ImplicitA(in_channels[i]))
            self.conv.append(
                nn.Conv2d(in_channels[i], ch, kernel_size=3, padding=1)
            )
            #self.im.append(ImplicitM(ch))

    def forward(self, inputs):
        outputs = []
        #for k, (ia, head_conv, im, x) in enumerate(zip(self.ia, self.conv, self.im, inputs)):
        for k, (head_conv, x) in enumerate(zip(self.conv, inputs)):
            # x: [batch_size, n_ch, h, w]
            #x = ia(x)
            x = head_conv(x)
            #x = im(x)
            outputs.append(x)
        return outputs
