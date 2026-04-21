import argparse
from pathlib import Path
from PIL import Image


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def preprocess_images(input_dir: str, output_dir: str, min_size: int) -> None:
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    files = [p for p in input_path.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    ok_count = 0
    skip_count = 0

    for file in files:
        try:
            with Image.open(file) as img:
                img = img.convert("RGB")
                w, h = img.size
                if w < min_size or h < min_size:
                    skip_count += 1
                    continue

                ok_count += 1
                save_path = output_path / f"{ok_count:06d}.jpg"
                img.save(save_path, quality=95)
        except Exception as exc:
            skip_count += 1
            print(f"skip {file}: {exc}")

    print(f"saved: {ok_count}")
    print(f"skipped: {skip_count}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-size", type=int, default=256)
    args = parser.parse_args()

    preprocess_images(args.input, args.output, args.min_size)


if __name__ == "__main__":
    main()