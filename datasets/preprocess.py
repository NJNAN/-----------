"""把原始图片清洗成适合训练使用的统一图片集合。

这个脚本做的事情很简单：
1. 递归扫描输入目录里的所有图片；
2. 统一转成 RGB，避免通道数不一致；
3. 过滤掉宽或高小于阈值的图片；
4. 重新命名成连续编号的 JPG 文件输出到目标目录。

它通常在训练前运行一次，用来把杂乱的原始素材整理干净。
"""

import argparse
from pathlib import Path
from PIL import Image


# 支持的图片后缀。这里只处理常见的栅格图片格式。
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def preprocess_images(input_dir: str, output_dir: str, min_size: int) -> None:
    """递归扫描原始图片，并按统一规则清洗后保存。

    参数：
        input_dir: 原始图片目录。支持子目录递归扫描。
        output_dir: 清洗后的输出目录，如果不存在会自动创建。
        min_size: 图片宽和高都必须达到的最小阈值，低于这个尺寸的图片会被跳过。
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Path.rglob 会递归扫描整个输入目录，所以原始数据可以按子文件夹组织。
    files = [p for p in input_path.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    ok_count = 0
    skip_count = 0

    for file in files:
        try:
            with Image.open(file) as img:
                # 统一成 RGB，避免灰度图、带透明通道的图片在后续训练里造成通道不一致。
                img = img.convert("RGB")
                w, h = img.size

                # 太小的图片保留意义不大，还可能拉低训练质量，所以直接跳过。
                if w < min_size or h < min_size:
                    skip_count += 1
                    continue

                ok_count += 1
                # 保存时重新编号，确保输出目录里的文件名连续、稳定、好管理。
                save_path = output_path / f"{ok_count:06d}.jpg"
                # quality=95 是清晰度和文件体积之间比较常见的折中。
                img.save(save_path, quality=95)
        except Exception as exc:
            # 某些损坏图片或奇怪编码图片会在打开时失败，这里只记录并跳过，不让整批处理被中断。
            skip_count += 1
            print(f"skip {file}: {exc}")

    print(f"saved: {ok_count}")
    print(f"skipped: {skip_count}")


def main():
    # 命令行入口：指定输入目录、输出目录和最小图片尺寸。
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-size", type=int, default=256)
    args = parser.parse_args()

    preprocess_images(args.input, args.output, args.min_size)


if __name__ == "__main__":
    main()