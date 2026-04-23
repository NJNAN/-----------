"""对生成器和判别器做一次最小前向测试。"""

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.blocks import init_weights
from models.discriminator import PatchDiscriminator
from models.generator import ResnetGenerator


def count_parameters(model) -> int:
    return sum(param.numel() for param in model.parameters())


def main():
    parser = argparse.ArgumentParser(description="Smoke test the generator and discriminator.")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--ngf", type=int, default=64)
    parser.add_argument("--ndf", type=int, default=64)
    parser.add_argument("--n-blocks", type=int, default=6)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    generator = ResnetGenerator(3, 3, ngf=args.ngf, n_blocks=args.n_blocks).to(device)
    discriminator = PatchDiscriminator(3, ndf=args.ndf).to(device)

    generator.apply(init_weights)
    discriminator.apply(init_weights)

    x = torch.randn(args.batch_size, 3, args.image_size, args.image_size, device=device)
    fake = generator(x)
    pred = discriminator(fake)

    print(f"input shape: {tuple(x.shape)}")
    print(f"generator output shape: {tuple(fake.shape)}")
    print(f"discriminator output shape: {tuple(pred.shape)}")
    print(f"generator params: {count_parameters(generator)}")
    print(f"discriminator params: {count_parameters(discriminator)}")


if __name__ == "__main__":
    main()
