"""把原始数据下载、解压并整理到项目预期的 raw 目录结构里。

这个脚本的目标不是训练，而是把外部来源的数据统一成项目可识别的形式：
1. 原始梵高图片压缩包解压到 data/raw/vangogh；
2. 照片数据从 COCO val2017 或 Caltech101 下载并解压到 data/raw/photo；
3. 去掉多余的外层目录、清理 __MACOSX 等杂项，让目录结构更规整。
"""

import argparse
import shutil
import tarfile
import urllib.request
import zipfile
from pathlib import Path


# 默认梵高原始压缩包文件名。用户如果手动下载了原始数据，可以直接放在这个名字下。
DEFAULT_VANGOGH_ZIP = Path("3sjjtjfhx7-2.zip")
# 原始梵高数据解压后的目标目录。
DEFAULT_VANGOGH_DIR = Path("data/raw/vangogh")
# 照片数据的目标目录。
DEFAULT_PHOTO_DIR = Path("data/raw/photo")
# 下载缓存目录。避免每次重复下载同一个压缩包。
DEFAULT_DOWNLOAD_DIR = Path("data/raw/_downloads")
# COCO val2017 的公开下载地址。
DEFAULT_COCO_VAL2017_URL = "http://images.cocodataset.org/zips/val2017.zip"
# Caltech101 的公开下载地址。
DEFAULT_CALTECH101_URL = "https://data.caltech.edu/records/mzrjq-6wc02/files/caltech-101.zip?download=1"
# 支持的图片后缀。这里只统计常见的图片格式。
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def ensure_dir(path: Path) -> None:
    """如果目录不存在，就连同父目录一起创建。"""
    path.mkdir(parents=True, exist_ok=True)


def count_files(path: Path) -> int:
    """递归统计目录下的所有文件数量。"""
    if not path.exists():
        return 0
    return sum(1 for p in path.rglob("*") if p.is_file())


def count_images(path: Path) -> int:
    """递归统计目录下的图片数量。"""
    if not path.exists():
        return 0
    return sum(1 for p in path.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def extract_zip(zip_path: Path, target_dir: Path) -> None:
    """把 zip 压缩包完整解压到目标目录。"""
    ensure_dir(target_dir)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(target_dir)


def download_file(url: str, dst: Path) -> None:
    """下载文件到本地；如果文件已经存在且大小大于 0，就直接复用缓存。"""
    ensure_dir(dst.parent)
    if dst.exists() and dst.stat().st_size > 0:
        return
    print(f"downloading: {url}")
    urllib.request.urlretrieve(url, dst)


def extract_archive(archive: Path, target_dir: Path) -> None:
    """解压 zip 或 tar.gz 压缩包。"""
    ensure_dir(target_dir)
    if archive.suffix.lower() == ".zip":
        with zipfile.ZipFile(archive, "r") as zf:
            zf.extractall(target_dir)
    else:
        with tarfile.open(archive, "r:gz") as tf:
            tf.extractall(target_dir)


def clean_dir(path: Path) -> None:
    """删除整个目录，常用于 --force 重新生成数据时。"""
    if path.exists():
        shutil.rmtree(path)


def flatten_root_folder(target_dir: Path, root_name: str) -> None:
    """去掉压缩包常见的最外层目录，只保留里面真正的内容。

    很多数据集压缩包解压后会多包一层目录，比如：
    target_dir/root_name/真正图片...
    这个函数会把 root_name 里面的内容搬到 target_dir 下面，
    让后续脚本看到的结构更扁平、更统一。
    """
    root_folder = target_dir / root_name
    if not root_folder.exists():
        return
    for item in root_folder.iterdir():
        shutil.move(str(item), str(target_dir / item.name))
    shutil.rmtree(root_folder)


def prepare_vangogh(zip_path: Path, target_dir: Path, force: bool = False) -> None:
    """准备梵高风格数据：检查压缩包、解压、整理目录。"""
    if not zip_path.exists():
        raise FileNotFoundError(f"vangogh archive not found: {zip_path}")
    if force:
        clean_dir(target_dir)
    if count_images(target_dir) > 0:
        print(f"skip vangogh extraction, already has files: {target_dir}")
        return

    # 这个压缩包解开后通常还会带一层根目录，所以后面要再扁平化一次。
    print(f"extracting vangogh archive -> {target_dir}")
    extract_zip(zip_path, target_dir)
    flatten_root_folder(target_dir, "3sjjtjfhx7-2")

    # macOS 打包时常见的隐藏目录，训练时没有任何意义，直接删掉。
    macosx = target_dir / "__MACOSX"
    if macosx.exists():
        shutil.rmtree(macosx)
    print(f"vangogh images: {count_images(target_dir)}")


def prepare_photo(photo_dir: Path, download_dir: Path, photo_source: str, photo_url: str, force: bool = False) -> None:
    """准备照片数据：下载、解压、再按数据源类型做二次整理。"""
    if force:
        clean_dir(photo_dir)
    if count_images(photo_dir) > 0:
        print(f"skip photo download, already has files: {photo_dir}")
        return

    ensure_dir(photo_dir)
    ensure_dir(download_dir)

    # 不同照片数据源的压缩包名字不同，所以这里根据来源选择缓存文件名。
    archive_name = "val2017.zip" if photo_source == "coco-val2017" else "caltech-101.zip"
    archive_path = download_dir / archive_name
    download_file(photo_url, archive_path)

    # 先把压缩包解到 photo 目录，再根据不同数据源做额外的目录整理。
    print(f"extracting {photo_source} archive -> {photo_dir}")
    extract_archive(archive_path, photo_dir)

    if photo_source == "coco-val2017":
        # COCO val2017 解压后一般会多一层 val2017 目录。
        flatten_root_folder(photo_dir, "val2017")
    else:
        # Caltech101 的 zip 里常常还嵌着一个 101_ObjectCategories.tar.gz，需要再解一次。
        nested_tar = photo_dir / "caltech-101" / "101_ObjectCategories.tar.gz"
        if nested_tar.exists():
            print(f"extracting nested caltech101 tar -> {photo_dir}")
            extract_archive(nested_tar, photo_dir)
            shutil.rmtree(photo_dir / "caltech-101")

    # 清理 macOS 压缩时残留的隐藏目录。
    macosx = photo_dir / "__MACOSX"
    if macosx.exists():
        shutil.rmtree(macosx)

    print(f"photo images: {count_images(photo_dir)}")


def main():
    # 命令行入口：先准备梵高数据，再准备照片数据。
    parser = argparse.ArgumentParser(description="Download and place datasets into the expected raw directories.")
    parser.add_argument("--vangogh-zip", default=str(DEFAULT_VANGOGH_ZIP))
    parser.add_argument("--vangogh-dir", default=str(DEFAULT_VANGOGH_DIR))
    parser.add_argument("--photo-dir", default=str(DEFAULT_PHOTO_DIR))
    parser.add_argument("--download-dir", default=str(DEFAULT_DOWNLOAD_DIR))
    parser.add_argument("--photo-source", choices=["coco-val2017", "caltech101"], default="coco-val2017")
    parser.add_argument("--photo-url", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    prepare_vangogh(Path(args.vangogh_zip), Path(args.vangogh_dir), force=args.force)

    # 如果没有显式传入 photo-url，就根据数据源自动选默认下载地址。
    photo_url = args.photo_url
    if photo_url is None:
        photo_url = DEFAULT_COCO_VAL2017_URL if args.photo_source == "coco-val2017" else DEFAULT_CALTECH101_URL
    prepare_photo(Path(args.photo_dir), Path(args.download_dir), args.photo_source, photo_url, force=args.force)

    print("done")
    print(f"vangogh dir: {args.vangogh_dir} ({count_images(Path(args.vangogh_dir))} images)")
    print(f"photo dir: {args.photo_dir} ({count_images(Path(args.photo_dir))} images)")


if __name__ == "__main__":
    main()
