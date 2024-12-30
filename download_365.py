import os
from itertools import repeat
from multiprocessing.pool import ThreadPool
from pathlib import Path
from tarfile import is_tarfile
from zipfile import ZipFile, is_zipfile

import torch
from tqdm import tqdm

def unzip_file(file, path=None, exclude=(".DS_Store", "__MACOSX")):
    # Unzip a *.zip file to path/, excluding files containing strings in exclude list
    if path is None:
        path = Path(file).parent  # default path
    with ZipFile(file) as zipObj:
        for f in zipObj.namelist():  # list all archived filenames in the zip
            if all(x not in f for x in exclude):
                zipObj.extract(f, path=path)

# [NOTE] curl to `True` so you don't have to use torch
def download(url, dir=".", unzip=True, delete=True, curl=False, threads=1, retry=3):
    # Multithreaded file download and unzip function, used in data.yaml for autodownload
    def download_one(url, dir):
        out_path = dir / Path(url).name
        out_path = str(out_path).split('.tar.gz')[0]
        if os.path.exists(out_path):
            print(f"Path {out_path} exists, skipping download.")
            return
        # Download 1 file
        success = True
        if os.path.isfile(url):
            f = Path(url)  # filename
        else:  # does not exist
            f = dir / Path(url).name
            print(f"Downloading {url} to {f}...")
            for i in range(retry + 1):
                if curl:
                    s = "sS" if threads > 1 else ""  # silent
                    r = os.system(
                        f'curl -# -{s}L "{url}" -o "{f}" --retry 9 -C -'
                    )  # curl download with retry, continue
                    success = r == 0
                else:
                    torch.hub.download_url_to_file(
                        url, f, progress=threads == 1
                    )  # torch download
                    success = f.is_file()
                if success:
                    break
                elif i < retry:
                    print(f"⚠️ Download failure, retrying {i + 1}/{retry} {url}...")
                else:
                    print(f"❌ Failed to download {url}...")

        if unzip and success and (f.suffix == ".gz" or is_zipfile(f) or is_tarfile(f)):
            print(f"Unzipping {f}...")
            if is_zipfile(f):
                unzip_file(f, dir)  # unzip
            elif is_tarfile(f):
                os.system(f"tar xf {f} --directory {f.parent}")  # unzip
            elif f.suffix == ".gz":
                os.system(f"tar xfz {f} --directory {f.parent}")  # unzip
            if delete:
                f.unlink()  # remove zip

    dir = Path(dir)
    dir.mkdir(parents=True, exist_ok=True)  # make directory
    if threads > 1:
        pool = ThreadPool(threads)
        pool.imap(lambda x: download_one(*x), zip(url, repeat(dir)))  # multithreaded
        pool.close()
        pool.join()
    else:
        for u in [url] if isinstance(url, (str, Path)) else url:
            download_one(u, dir)

dir = Path("object365")  #
# Make Directories
for p in "images", "labels":
    (dir / p).mkdir(parents=True, exist_ok=True)
    for q in "train", "val":
        (dir / p / q).mkdir(parents=True, exist_ok=True)
# Train, Val Splits
for split, patches in [("train", 50 + 1), ("val", 43 + 1)]:
    print(f"Processing {split} in {patches} patches ...")
    images, labels = dir / "images" / split, dir / "labels" / split
    # Download
    url = f"https://dorc.ks3-cn-beijing.ksyun.com/data-set/2020Objects365%E6%95%B0%E6%8D%AE%E9%9B%86/{split}/"
    if split == "train":
        if not (dir / f"zhiyuan_objv2_{split}.json").is_file():
            download(
                [f"{url}zhiyuan_objv2_{split}.tar.gz"], dir=dir, delete=True
            )  # annotations json
        download(
            [f"{url}patch{i}.tar.gz" for i in range(patches)],
            dir=images,
            curl=False,
            delete=True,
            threads=8,
        )
    elif split == "val":
        download(
            [f"{url}zhiyuan_objv2_{split}.json"], dir=dir, delete=True
        )  # annotations json
        download(
            [f"{url}images/v1/patch{i}.tar.gz" for i in range(15 + 1)],
            dir=images,
            curl=False,
            delete=True,
            threads=8,
        )
        download(
            [f"{url}images/v2/patch{i}.tar.gz" for i in range(16, patches)],
            dir=images,
            curl=False,
            delete=True,
            threads=8,
        )

# merge all contents of the downloaded folders into the parent directory
# FROM:
# --- images
#     --- train
#         --- patch0
#             --- img0.jpg
#             --- img1.jpg
#             ...
#         --- patch1
#         ...
#     --- val
#         --- patch0
#         --- patch1
#         ...
# TO:
# --- images
#     --- train
#         --- img0.jpg
#         --- img1.jpg
#         ...
#     --- val
#         --- img0.jpg
#         --- img1.jpg
#         ...

count_files = 0
for p in dir.glob("images/*"):
    if p.is_dir():
        for q in tqdm(p.glob("patch*")):
            for f in q.glob("*"):
                f.rename(p / f.name)
                count_files += 1
            q.rmdir()

# check that the number of files in the parent directory is correct
# count the number of files in all subdirectory of images, excluding directories
count_files2 = sum(len(list(p.glob("*"))) for p in dir.glob("images/*") if p.is_dir())
assert count_files == count_files2, f"Number of files mismatch: {count_files} != {count_files2}"

print(f"Unzipped {count_files} files to {dir.resolve()}")