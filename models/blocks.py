"""模型里会重复使用的小模块。"""

import torch.nn as nn


class ResidualBlock(nn.Module):
    """最基础的残差块。

    输入和输出的通道数不变，所以可以直接做残差相加：
    output = input + block(input)
    """

    def __init__(self, channels: int):
        super().__init__()

        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=0),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=0),
            nn.InstanceNorm2d(channels),
        )

    def forward(self, x):
        return x + self.block(x)


def init_weights(module: nn.Module) -> None:
    """给卷积层和归一化层做常见的 GAN 初始化。"""

    classname = module.__class__.__name__

    if "Conv" in classname:
        nn.init.normal_(module.weight.data, mean=0.0, std=0.02)
        if module.bias is not None:
            nn.init.constant_(module.bias.data, 0.0)

    elif "InstanceNorm" in classname:
        if getattr(module, "weight", None) is not None:
            nn.init.normal_(module.weight.data, mean=1.0, std=0.02)
        if getattr(module, "bias", None) is not None:
            nn.init.constant_(module.bias.data, 0.0)
