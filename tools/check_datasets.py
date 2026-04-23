"""检查原始数据集是否放对位置、图片是否可读、目录结构是否正常。

这个脚本更像“数据体检工具”：
它不会修改任何文件，只会统计图片数量、读取一张样例图、
并打印顶层目录结构，方便你在正式清洗和训练前确认数据没问题。
"""

import argparse
from pathlib import Path

from PIL import Image


# 支持的图片后缀。这里只统计常见的图片格式。
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def iter_images(root: Path):
    """递归遍历目录中的所有图片文件。"""
    return [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS]


def first_image(root: Path):
    """返回目录中找到的第一张图片，用来做样例检查。"""
    for path in iter_images(root):
        return path
    return None


def resolve_photo_root(raw_photo_dir: Path) -> Path:
    """兼容不同照片数据来源的目录结构。

    有些照片数据会额外包一层目录，例如：
    - Caltech101: data/raw/photo/101_ObjectCategories
    - COCO val2017: data/raw/photo/val2017

    如果找到这类常见子目录，就返回更深一层的真实图片根目录。
    """
    for candidate in [
        raw_photo_dir / "101_ObjectCategories",
        raw_photo_dir / "val2017",
    ]:
        if candidate.exists():
            return candidate
    return raw_photo_dir


def inspect_dataset(name: str, root: Path) -> int:
    """检查一个数据集目录是否可读，并打印基础统计信息。"""
    if not root.exists():
        raise FileNotFoundError(f"{name} path not found: {root}")

    images = iter_images(root)
    if not images:
        raise RuntimeError(f"{name} has no images: {root}")

    sample = first_image(root)
    if sample is None:
        raise RuntimeError(f"{name} sample image not found: {root}")

    with Image.open(sample) as img:
        # 这里真正把图片读进来，确认文件不是坏的，也顺便打印尺寸和颜色模式。
        img.load()
        print(f"{name}: {len(images)} images")
        print(f"{name}: sample -> {sample}")
        print(f"{name}: sample size -> {img.size[0]}x{img.size[1]}, mode={img.mode}")

    # 打印顶层目录名称，帮助快速判断数据有没有被正确解包到预期结构里。
    top_level_dirs = sorted([p.name for p in root.iterdir() if p.is_dir()])
    if top_level_dirs:
        preview = ", ".join(top_level_dirs[:10])
        suffix = "" if len(top_level_dirs) <= 10 else " ..."
        print(f"{name}: top-level dirs -> {preview}{suffix}")

    return len(images)


def main():
    # 命令行入口：默认检查 data/raw 下的梵高图和照片图。
    parser = argparse.ArgumentParser(description="Check raw datasets for the style transfer project.")
    parser.add_argument("--vangogh", default="data/raw/vangogh", help="Path to the Van Gogh raw dataset root")
    parser.add_argument("--photo", default="data/raw/photo", help="Path to the photo raw dataset root")
    args = parser.parse_args()

    # 照片数据可能来自不同来源，所以先把可能的外层目录折叠到真正的图片根目录。
    vangogh_root = Path(args.vangogh)
    photo_root = resolve_photo_root(Path(args.photo))

    print("checking datasets")
    print("-" * 40)
    inspect_dataset("vangogh", vangogh_root)
    print("-" * 40)
    inspect_dataset("photo", photo_root)
    print("-" * 40)
    print("dataset check passed")


if __name__ == "__main__":
    main()
