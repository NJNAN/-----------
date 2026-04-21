import random
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


class UnpairedImageDataset(Dataset):
    def __init__(self, dir_a, dir_b, resize_size=286, image_size=256, train=True):
        self.files_a = sorted(
            [p for p in Path(dir_a).iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
        )
        self.files_b = sorted(
            [p for p in Path(dir_b).iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
        )

        if not self.files_a:
            raise RuntimeError(f"no images in {dir_a}")
        if not self.files_b:
            raise RuntimeError(f"no images in {dir_b}")

        if train:
            self.transform = transforms.Compose([
                transforms.Resize((resize_size, resize_size)),
                transforms.RandomCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])

    def __len__(self):
        return max(len(self.files_a), len(self.files_b))

    def __getitem__(self, index):
        path_a = self.files_a[index % len(self.files_a)]
        path_b = random.choice(self.files_b)

        image_a = Image.open(path_a).convert("RGB")
        image_b = Image.open(path_b).convert("RGB")

        return {
            "A": self.transform(image_a),
            "B": self.transform(image_b),
            "A_path": str(path_a),
            "B_path": str(path_b),
        }