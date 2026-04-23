"""把梵高风格迁移项目的数据阶段一次性推进到可训练状态。

这个脚本负责三件事：
1. 如果 clean 数据不存在，就从 raw 数据清洗生成 clean 数据；
2. 用 clean 数据生成 CycleGAN 训练所需的 prepared 目录结构；
3. 打印每一步的图片数量，方便快速确认数据阶段是否完成。

它不会训练模型，只负责把数据准备到“训练脚本能直接读取”的程度。
"""

import argparse
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.preprocess import preprocess_images
from datasets.split_dataset import split_dataset


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def count_images(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for p in path.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def ensure_clean_dataset(raw_dir: Path, clean_dir: Path, min_size: int, force_clean: bool) -> None:
    if force_clean and clean_dir.exists():
        shutil.rmtree(clean_dir)

    if count_images(clean_dir) > 0:
        print(f"skip clean -> {clean_dir} ({count_images(clean_dir)} images)")
        return

    print(f"build clean dataset: {raw_dir} -> {clean_dir}")
    preprocess_images(str(raw_dir), str(clean_dir), min_size)
    print(f"clean done: {clean_dir} ({count_images(clean_dir)} images)")


def rebuild_prepared_dataset(
    clean_photo_dir: Path,
    clean_style_dir: Path,
    prepared_dir: Path,
    test_ratio: float,
    seed: int,
    force_split: bool,
) -> None:
    if force_split and prepared_dir.exists():
        shutil.rmtree(prepared_dir)

    if count_images(prepared_dir) > 0:
        print(f"skip split -> {prepared_dir} ({count_images(prepared_dir)} images)")
        return

    print(f"build prepared dataset: {prepared_dir}")
    split_dataset(str(clean_photo_dir), str(clean_style_dir), str(prepared_dir), test_ratio, seed)

    for name in ["trainA", "trainB", "testA", "testB"]:
        folder = prepared_dir / name
        print(f"{name}: {count_images(folder)} images")


def main():
    parser = argparse.ArgumentParser(description="Prepare clean and prepared data for the Van Gogh training stage.")
    parser.add_argument("--raw-photo", default="data/raw/photo")
    parser.add_argument("--raw-style", default="data/raw/vangogh")
    parser.add_argument("--clean-photo", default="data/clean/photo")
    parser.add_argument("--clean-style", default="data/clean/vangogh")
    parser.add_argument("--prepared", default="data/prepared/vangogh")
    parser.add_argument("--min-size", type=int, default=256)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force-clean", action="store_true")
    parser.add_argument("--force-split", action="store_true")
    args = parser.parse_args()

    raw_photo = Path(args.raw_photo)
    raw_style = Path(args.raw_style)
    clean_photo = Path(args.clean_photo)
    clean_style = Path(args.clean_style)
    prepared = Path(args.prepared)

    ensure_clean_dataset(raw_photo, clean_photo, args.min_size, args.force_clean)
    ensure_clean_dataset(raw_style, clean_style, args.min_size, args.force_clean)
    rebuild_prepared_dataset(clean_photo, clean_style, prepared, args.test_ratio, args.seed, args.force_split)

    print("data stage ready")
    print(f"raw photo: {count_images(raw_photo)}")
    print(f"raw style: {count_images(raw_style)}")
    print(f"clean photo: {count_images(clean_photo)}")
    print(f"clean style: {count_images(clean_style)}")
    print(f"prepared total: {count_images(prepared)}")


if __name__ == "__main__":
    main()
