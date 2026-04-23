"""CycleGAN 风格迁移里使用的生成器。"""

import torch.nn as nn

from models.blocks import ResidualBlock


class ResnetGenerator(nn.Module):
    """一个适合 256x256 图片的 ResNet 生成器。"""

    def __init__(self, input_nc: int = 3, output_nc: int = 3, ngf: int = 64, n_blocks: int = 6):
        super().__init__()

        layers = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, stride=1, padding=0),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=True),
        ]

        in_channels = ngf
        out_channels = ngf * 2

        # 两次下采样：256 -> 128 -> 64
        for _ in range(2):
            layers += [
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
                nn.InstanceNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ]
            in_channels = out_channels
            out_channels *= 2

        # 在较小的特征图上做多次残差变换，学习风格纹理。
        for _ in range(n_blocks):
            layers.append(ResidualBlock(in_channels))

        out_channels = in_channels // 2

        # 两次上采样：64 -> 128 -> 256
        for _ in range(2):
            layers += [
                nn.ConvTranspose2d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                ),
                nn.InstanceNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ]
            in_channels = out_channels
            out_channels //= 2

        layers += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, output_nc, kernel_size=7, stride=1, padding=0),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
