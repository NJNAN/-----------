import argparse
from pathlib import Path

from PIL import Image


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def iter_images(root: Path):
    return [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS]


def first_image(root: Path):
    for path in iter_images(root):
        return path
    return None


def resolve_photo_root(raw_photo_dir: Path) -> Path:
    for candidate in [
        raw_photo_dir / "101_ObjectCategories",
        raw_photo_dir / "val2017",
    ]:
        if candidate.exists():
            return candidate
    return raw_photo_dir


def inspect_dataset(name: str, root: Path) -> int:
    if not root.exists():
        raise FileNotFoundError(f"{name} path not found: {root}")

    images = iter_images(root)
    if not images:
        raise RuntimeError(f"{name} has no images: {root}")

    sample = first_image(root)
    if sample is None:
        raise RuntimeError(f"{name} sample image not found: {root}")

    with Image.open(sample) as img:
        img.load()
        print(f"{name}: {len(images)} images")
        print(f"{name}: sample -> {sample}")
        print(f"{name}: sample size -> {img.size[0]}x{img.size[1]}, mode={img.mode}")

    top_level_dirs = sorted([p.name for p in root.iterdir() if p.is_dir()])
    if top_level_dirs:
        preview = ", ".join(top_level_dirs[:10])
        suffix = "" if len(top_level_dirs) <= 10 else " ..."
        print(f"{name}: top-level dirs -> {preview}{suffix}")

    return len(images)


def main():
    parser = argparse.ArgumentParser(description="Check raw datasets for the style transfer project.")
    parser.add_argument("--vangogh", default="data/raw/vangogh", help="Path to the Van Gogh raw dataset root")
    parser.add_argument("--photo", default="data/raw/photo", help="Path to the photo raw dataset root")
    args = parser.parse_args()

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
