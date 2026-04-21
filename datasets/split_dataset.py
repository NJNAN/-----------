import argparse
import random
import shutil
from pathlib import Path


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def list_images(path: Path):
    return [p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]


def copy_split(files, train_dir: Path, test_dir: Path, test_ratio: float):
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    random.shuffle(files)
    test_count = max(1, int(len(files) * test_ratio))
    test_files = files[:test_count]
    train_files = files[test_count:]

    for idx, file in enumerate(train_files, start=1):
        shutil.copy2(file, train_dir / f"{idx:06d}{file.suffix.lower()}")

    for idx, file in enumerate(test_files, start=1):
        shutil.copy2(file, test_dir / f"{idx:06d}{file.suffix.lower()}")

    return len(train_files), len(test_files)


def split_dataset(photo_dir, style_dir, output_dir, test_ratio, seed):
    random.seed(seed)
    output = Path(output_dir)

    photo_files = list_images(Path(photo_dir))
    style_files = list_images(Path(style_dir))

    if not photo_files:
        raise RuntimeError(f"no photo images found: {photo_dir}")
    if not style_files:
        raise RuntimeError(f"no style images found: {style_dir}")

    train_a, test_a = copy_split(photo_files, output / "trainA", output / "testA", test_ratio)
    train_b, test_b = copy_split(style_files, output / "trainB", output / "testB", test_ratio)

    print(f"trainA: {train_a}, testA: {test_a}")
    print(f"trainB: {train_b}, testB: {test_b}")


def main():
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