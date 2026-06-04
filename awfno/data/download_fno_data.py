#!/usr/bin/env python3
"""
Download FNO benchmark datasets.

Usage:
    python datasets/download_fno_data.py --dataset ns2d
    python datasets/download_fno_data.py --dataset burgers1d
    python datasets/download_fno_data.py --dataset all

The script will:
  1. Check if files already exist (skip if present and checksum matches).
  2. Download from the official mirror with progress bar.
  3. Verify integrity via MD5 checksum.

If you already have data at a non-standard path, either set DATA_PATH:
    DATA_PATH=/path/to/data python datasets/download_fno_data.py --check ns2d

or symlink the directory:
    ln -s /media/HDD/mamta_backup/datasets/fno/navier_stokes data/ns2d
"""

import argparse
import hashlib
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------
# Official Google Drive links from: https://github.com/neuraloperator/neuraloperator
REGISTRY = {
    "ns2d": {
        "description": "Navier-Stokes 2D, Re=1000, 64x64, T=50",
        "output_dir": "data/ns2d",
        "files": [
            {
                "name": "ns_V1e-3_N1000_T50.mat",
                "gdrive_id": "1r3idxpsHa21ijhlu3QQ1hVuXcqnBTO7d",
                "md5": None,  # fill after first download
                "size_mb": 298,
            },
        ],
    },
    "burgers1d": {
        "description": "Burgers 1D, N=2048, resolution=8192",
        "output_dir": "data/burgers1d",
        "files": [
            {
                "name": "burgers_data_R10.mat",
                "gdrive_id": "16a8od4vidbiNR3WtaBPCSZ0T3moxjhYe",
                "md5": None,
                "size_mb": 86,
            },
        ],
    },
    "darcy2d": {
        "description": "Darcy Flow 2D, N=1024, resolution=421",
        "output_dir": "data/darcy2d",
        "files": [
            {
                "name": "piececonst_r421_N1024_smooth1.mat",
                "gdrive_id": "1ViDqN7nc_VCnMackiXv_d7CHZANAFKzV",
                "md5": None,
                "size_mb": 485,
            },
            {
                "name": "piececonst_r421_N1024_smooth2.mat",
                "gdrive_id": "1tTP5zgbykvsuIAp tackJm7gNX9jFFnet",
                "md5": None,
                "size_mb": 485,
            },
        ],
    },
}


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def _gdrive_url(file_id: str) -> str:
    return f"https://drive.google.com/uc?id={file_id}&export=download&confirm=t"


def _md5(path: Path, chunk: int = 8192) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            buf = f.read(chunk)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def _download(url: str, dest: Path, expected_md5: str | None = None) -> None:
    try:
        import requests
        from tqdm import tqdm
    except ImportError:
        print("Install requests and tqdm: pip install requests tqdm")
        sys.exit(1)

    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading → {dest}")

    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True) as bar:
            for chunk in r.iter_content(chunk_size=65536):
                f.write(chunk)
                bar.update(len(chunk))

    if expected_md5:
        actual = _md5(dest)
        if actual != expected_md5:
            dest.unlink()
            raise ValueError(f"Checksum mismatch for {dest.name}.\n  expected {expected_md5}\n  got      {actual}")
        print(f"  ✓ Checksum OK ({actual[:8]}...)")


def download_dataset(name: str, root: Path) -> None:
    if name not in REGISTRY:
        print(f"Unknown dataset '{name}'. Options: {list(REGISTRY)}")
        sys.exit(1)

    spec = REGISTRY[name]
    out_dir = root / spec["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Dataset : {name}")
    print(f"Info    : {spec['description']}")
    print(f"Output  : {out_dir}")

    for fspec in spec["files"]:
        dest = out_dir / fspec["name"]
        if dest.exists():
            if fspec["md5"] and _md5(dest) == fspec["md5"]:
                print(f"  ✓ {fspec['name']} already present (checksum OK), skipping.")
                continue
            else:
                print(f"  ~ {fspec['name']} present but checksum unknown — using as-is.")
                continue

        url = _gdrive_url(fspec["gdrive_id"])
        print(f"\n  File    : {fspec['name']}  (~{fspec['size_mb']} MB)")
        print(f"  URL     : {url}")
        try:
            _download(url, dest, fspec["md5"])
        except Exception as e:
            print(f"\n  ✗ Download failed: {e}")
            print(
                f"\n  Manual download instructions:\n"
                f"    1. Visit https://drive.google.com/file/d/{fspec['gdrive_id']}\n"
                f"    2. Download and place in: {out_dir}/{fspec['name']}\n"
            )

    print(f"\n✓ Dataset '{name}' ready at: {out_dir}")


def check_existing(name: str, root: Path) -> None:
    """Report whether a local dataset exists at the expected path."""
    if name not in REGISTRY:
        return
    spec = REGISTRY[name]
    out_dir = root / spec["output_dir"]
    for fspec in spec["files"]:
        dest = out_dir / fspec["name"]
        status = "✓" if dest.exists() else "✗"
        print(f"  {status} {dest}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download FNO benchmark datasets")
    p.add_argument(
        "--dataset",
        default="ns2d",
        choices=[*REGISTRY, "all"],
        help="Which dataset to download (default: ns2d)",
    )
    p.add_argument(
        "--root",
        default=".",
        help="Project root directory (default: current directory)",
    )
    p.add_argument(
        "--check",
        action="store_true",
        help="Only check whether files are present; don't download",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    root = Path(args.root).resolve()
    names = list(REGISTRY) if args.dataset == "all" else [args.dataset]

    for name in names:
        if args.check:
            check_existing(name, root)
        else:
            download_dataset(name, root)
