"""非配对训练用的数据集读取器。

CycleGAN 这类风格迁移方法不需要 A、B 两边的图片一一对应。
这里的做法是：A 域图片按索引顺序取，B 域图片随机抽一张，
这样每次返回的样本都是一张照片 + 一张风格图，但两者没有固定配对关系。
"""

import random
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


# 支持的图片后缀。这里只读取常见的栅格图片。
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


class UnpairedImageDataset(Dataset):
    """读取非配对图片对的数据集。

    返回结果不是“固定配对”的图片，而是：
    - A: 一张照片域图片
    - B: 一张风格域图片
    这样更符合无监督风格迁移的训练方式。
    """

    def __init__(self, dir_a, dir_b, resize_size=286, image_size=256, train=True):
        # 只扫描目录下一层文件，并按文件名排序，保证每次启动后的顺序稳定。
        self.files_a = sorted(
            [p for p in Path(dir_a).iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
        )
        self.files_b = sorted(
            [p for p in Path(dir_b).iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
        )

        # 提前检查空目录，避免训练刚开始时才报错，方便定位数据准备问题。
        if not self.files_a:
            raise RuntimeError(f"no images in {dir_a}")
        if not self.files_b:
            raise RuntimeError(f"no images in {dir_b}")

        if train:
            # 训练阶段启用随机裁剪和翻转，增强数据多样性。
            self.transform = transforms.Compose([
                # 先统一缩放到较大的正方形，再从中随机裁一块 256x256。
                transforms.Resize((resize_size, resize_size)),
                # 随机裁剪能让模型看到同一张图的不同局部。
                transforms.RandomCrop(image_size),
                # 随机水平翻转，增加左右方向变化。
                transforms.RandomHorizontalFlip(),
                # 转成张量后，像素值会从 [0, 255] 变成 [0, 1]。
                transforms.ToTensor(),
                # 再把值映射到 [-1, 1]，这和很多 GAN 生成器的输出范围一致。
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        else:
            # 测试阶段不做随机增强，尽量保证结果稳定可复现。
            self.transform = transforms.Compose([
                # 直接缩放到固定大小，方便统一评估。
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])

    def __len__(self):
        # 取更长的那个域作为长度，确保一个 epoch 内能够覆盖足够多的样本。
        return max(len(self.files_a), len(self.files_b))

    def __getitem__(self, index):
        # A 域按索引循环取图，保证遍历顺序稳定。
        path_a = self.files_a[index % len(self.files_a)]
        # B 域随机抽图，刻意打破固定配对关系，这就是“非配对”的关键。
        path_b = random.choice(self.files_b)

        # 读取图片后统一转成 RGB，避免灰度图或带透明通道的图片影响后续训练。
        image_a = Image.open(path_a).convert("RGB")
        image_b = Image.open(path_b).convert("RGB")

        # 返回字典，方便训练脚本同时拿到张量和原始文件路径，便于调试和保存可视化结果。
        return {
            "A": self.transform(image_a),
            "B": self.transform(image_b),
            "A_path": str(path_a),
            "B_path": str(path_b),
        }