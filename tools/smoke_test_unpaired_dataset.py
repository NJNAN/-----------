"""对非配对数据集读取器做一次最小可运行验证。"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.unpaired_dataset import UnpairedImageDataset


def main():
    parser = argparse.ArgumentParser(description="Smoke test the unpaired dataset loader.")
    parser.add_argument("--trainA", default="data/prepared/vangogh/trainA")
    parser.add_argument("--trainB", default="data/prepared/vangogh/trainB")
    parser.add_argument("--resize-size", type=int, default=286)
    parser.add_argument("--image-size", type=int, default=256)
    args = parser.parse_args()

    dataset = UnpairedImageDataset(
        args.trainA,
        args.trainB,
        resize_size=args.resize_size,
        image_size=args.image_size,
        train=True,
    )

    sample = dataset[0]
    print(f"dataset length: {len(dataset)}")
    print(f"A shape: {tuple(sample['A'].shape)}")
    print(f"B shape: {tuple(sample['B'].shape)}")
    print(f"A path: {sample['A_path']}")
    print(f"B path: {sample['B_path']}")


if __name__ == "__main__":
    main()
