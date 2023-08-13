import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddleseg import utils
from paddleseg.cvlibs import manager
from paddleseg.models import layers


@manager.MODELS.add_component
class unet_more_layers_quadruple_conv(nn.Layer):
    def __init__(self,
                 num_classes,
                 in_channels=3):
        super().__init__()

        self.encode = Encoder(in_channels)
        self.decode = Decoder()
        self.cls = self.conv = nn.Conv2D(
            in_channels=64,
            out_channels=num_classes,
            kernel_size=3,
            stride=1,
            padding=1)

    def forward(self, x):
        logit_list = []
        x, short_cuts = self.encode(x)
        x = self.decode(x, short_cuts)
        logit = self.cls(x)
        logit_list.append(logit)
        logit_list.append(logit)
        return logit_list

class Encoder(nn.Layer):
    def __init__(self, in_channels=3):
        super().__init__()

        self.quadruple_conv = nn.Sequential(
            layers.ConvBNReLU(in_channels, 64, 3), layers.ConvBNReLU(64, 64, 3), layers.ConvBNReLU(64, 64, 3), layers.ConvBNReLU(64, 64, 3))
        down_channels = [[64, 128], [128, 256], [256, 512], [512, 1024], [1024, 2048], [2048, 4096]]
        self.down_sample_list = nn.LayerList([
            self.down_sampling(channel[0], channel[1])
            for channel in down_channels
        ])

    def down_sampling(self, in_channels, out_channels):
        modules = []
        modules.append(nn.MaxPool2D(kernel_size=2, stride=2))
        modules.append(layers.ConvBNReLU(in_channels, out_channels, 3))
        modules.append(layers.ConvBNReLU(out_channels, out_channels, 3))
        modules.append(layers.ConvBNReLU(out_channels, out_channels, 3))
        modules.append(layers.ConvBNReLU(out_channels, out_channels, 3))
        return nn.Sequential(*modules)

    def forward(self, x):
        short_cuts = []
        x = self.quadruple_conv(x)
        for down_sample in self.down_sample_list:
            short_cuts.append(x)
            x = down_sample(x)
        return x, short_cuts


class Decoder(nn.Layer):
    def __init__(self):
        super().__init__()

        up_channels = [[4096, 2048], [2048, 1024], [1024, 512], [512, 256], [256, 128], [128, 64]]
        self.up_sample_list = nn.LayerList([
            UpSampling(channel[0], channel[1])
            for channel in up_channels
        ])

    def forward(self, x, short_cuts):
        for i in range(len(short_cuts)):
            x = self.up_sample_list[i](x, short_cuts[-(i + 1)])
        return x


class UpSampling(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels):
        super().__init__()

        self.conv_2x2 = nn.Conv2D(in_channels=in_channels, out_channels=out_channels, kernel_size=2, padding='same')

        self.quadruple_conv = nn.Sequential(
            layers.ConvBNReLU(in_channels, out_channels, 3),
            layers.ConvBNReLU(out_channels, out_channels, 3), 
            layers.ConvBNReLU(out_channels, out_channels, 3), 
            layers.ConvBNReLU(out_channels, out_channels, 3)
        )
        
    def forward(self, x, short_cut):
        x = self.conv_2x2(x)
        x = F.interpolate(
            x,
            paddle.shape(short_cut)[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        x = paddle.concat([x, short_cut], axis=1)
        x = self.quadruple_conv(x)
        return x
