import argparse
import shutil
import tarfile
import urllib.request
import zipfile
from pathlib import Path


DEFAULT_VANGOGH_ZIP = Path("3sjjtjfhx7-2.zip")
DEFAULT_VANGOGH_DIR = Path("data/raw/vangogh")
DEFAULT_PHOTO_DIR = Path("data/raw/photo")
DEFAULT_DOWNLOAD_DIR = Path("data/raw/_downloads")
DEFAULT_COCO_VAL2017_URL = "http://images.cocodataset.org/zips/val2017.zip"
DEFAULT_CALTECH101_URL = "https://data.caltech.edu/records/mzrjq-6wc02/files/caltech-101.zip?download=1"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def count_files(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for p in path.rglob("*") if p.is_file())


def count_images(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for p in path.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def extract_zip(zip_path: Path, target_dir: Path) -> None:
    ensure_dir(target_dir)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(target_dir)


def download_file(url: str, dst: Path) -> None:
    ensure_dir(dst.parent)
    if dst.exists() and dst.stat().st_size > 0:
        return
    print(f"downloading: {url}")
    urllib.request.urlretrieve(url, dst)


def extract_archive(archive: Path, target_dir: Path) -> None:
    ensure_dir(target_dir)
    if archive.suffix.lower() == ".zip":
        with zipfile.ZipFile(archive, "r") as zf:
            zf.extractall(target_dir)
    else:
        with tarfile.open(archive, "r:gz") as tf:
            tf.extractall(target_dir)


def clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def flatten_root_folder(target_dir: Path, root_name: str) -> None:
    root_folder = target_dir / root_name
    if not root_folder.exists():
        return
    for item in root_folder.iterdir():
        shutil.move(str(item), str(target_dir / item.name))
    shutil.rmtree(root_folder)


def prepare_vangogh(zip_path: Path, target_dir: Path, force: bool = False) -> None:
    if not zip_path.exists():
        raise FileNotFoundError(f"vangogh archive not found: {zip_path}")
    if force:
        clean_dir(target_dir)
    if count_images(target_dir) > 0:
        print(f"skip vangogh extraction, already has files: {target_dir}")
        return
    print(f"extracting vangogh archive -> {target_dir}")
    extract_zip(zip_path, target_dir)
    flatten_root_folder(target_dir, "3sjjtjfhx7-2")
    macosx = target_dir / "__MACOSX"
    if macosx.exists():
        shutil.rmtree(macosx)
    print(f"vangogh images: {count_images(target_dir)}")


def prepare_photo(photo_dir: Path, download_dir: Path, photo_source: str, photo_url: str, force: bool = False) -> None:
    if force:
        clean_dir(photo_dir)
    if count_images(photo_dir) > 0:
        print(f"skip photo download, already has files: {photo_dir}")
        return

    ensure_dir(photo_dir)
    ensure_dir(download_dir)
    archive_name = "val2017.zip" if photo_source == "coco-val2017" else "caltech-101.zip"
    archive_path = download_dir / archive_name
    download_file(photo_url, archive_path)
    print(f"extracting {photo_source} archive -> {photo_dir}")
    extract_archive(archive_path, photo_dir)

    if photo_source == "coco-val2017":
        flatten_root_folder(photo_dir, "val2017")
    else:
        nested_tar = photo_dir / "caltech-101" / "101_ObjectCategories.tar.gz"
        if nested_tar.exists():
            print(f"extracting nested caltech101 tar -> {photo_dir}")
            extract_archive(nested_tar, photo_dir)
            shutil.rmtree(photo_dir / "caltech-101")

    macosx = photo_dir / "__MACOSX"
    if macosx.exists():
        shutil.rmtree(macosx)

    print(f"photo images: {count_images(photo_dir)}")


def main():
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
    photo_url = args.photo_url
    if photo_url is None:
        photo_url = DEFAULT_COCO_VAL2017_URL if args.photo_source == "coco-val2017" else DEFAULT_CALTECH101_URL
    prepare_photo(Path(args.photo_dir), Path(args.download_dir), args.photo_source, photo_url, force=args.force)

    print("done")
    print(f"vangogh dir: {args.vangogh_dir} ({count_images(Path(args.vangogh_dir))} images)")
    print(f"photo dir: {args.photo_dir} ({count_images(Path(args.photo_dir))} images)")


if __name__ == "__main__":
    main()
