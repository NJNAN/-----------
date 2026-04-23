"""把清洗后的照片域和风格域图片切分成训练集和测试集。

这个脚本不会去建立配对关系，而是分别处理两个域：
照片域（A）和风格域（B）。
最终会生成 trainA、testA、trainB、testB 四个目录，供非配对训练使用。
"""

import argparse
import random
import shutil
from pathlib import Path


# 支持的图片后缀。这里只处理常见的栅格图片格式。
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def list_images(path: Path):
    """列出目录下一层的图片文件，不递归子目录。"""
    return [p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]


def copy_split(files, train_dir: Path, test_dir: Path, test_ratio: float):
    """先打乱，再按比例切分，并复制到训练/测试目录中。

    这里使用复制而不是移动，是为了保留原始清洗后的数据不被破坏。
    """
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    # 先打乱文件顺序，避免原始文件排列顺序影响训练/测试划分。
    random.shuffle(files)
    # 至少保留 1 张测试图，避免小数据集切分后测试集为空。
    test_count = max(1, int(len(files) * test_ratio))
    test_files = files[:test_count]
    train_files = files[test_count:]

    for idx, file in enumerate(train_files, start=1):
        # 重新编号，让输出目录里的文件名连续整齐，便于后续读取。
        shutil.copy2(file, train_dir / f"{idx:06d}{file.suffix.lower()}")

    for idx, file in enumerate(test_files, start=1):
        shutil.copy2(file, test_dir / f"{idx:06d}{file.suffix.lower()}")

    return len(train_files), len(test_files)


def split_dataset(photo_dir, style_dir, output_dir, test_ratio, seed):
    """把照片域和风格域分别切分成训练集和测试集。"""
    random.seed(seed)
    output = Path(output_dir)

    # A 域通常是照片，B 域通常是目标风格图。
    photo_files = list_images(Path(photo_dir))
    style_files = list_images(Path(style_dir))

    # 先在入口处做空目录检查，尽早暴露数据准备问题。
    if not photo_files:
        raise RuntimeError(f"no photo images found: {photo_dir}")
    if not style_files:
        raise RuntimeError(f"no style images found: {style_dir}")

    # 分别生成 trainA/testA 和 trainB/testB。
    train_a, test_a = copy_split(photo_files, output / "trainA", output / "testA", test_ratio)
    train_b, test_b = copy_split(style_files, output / "trainB", output / "testB", test_ratio)

    print(f"trainA: {train_a}, testA: {test_a}")
    print(f"trainB: {train_b}, testB: {test_b}")


def main():
    # 命令行入口：指定照片目录、风格目录、输出目录和测试集比例。
    parser = argparse.ArgumentParser()
    parser.add_argument("--photo", required=True)
    parser.add_argument("--style", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    split_dataset(args.photo, args.style, args.output, args.test_ratio, args.seed)


if __name__ == "__main__":
    main()