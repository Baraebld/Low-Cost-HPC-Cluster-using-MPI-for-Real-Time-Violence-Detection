#!/usr/bin/env python3
"""
download_model.py
Helper utility to verify or download the trained model checkpoint (`hybrid_model3.pth`).
The weights are stored in the user's Hugging Face Space repository:
https://huggingface.co/spaces/Baraebld/Violence-Detection-model
"""

import sys
import os
from pathlib import Path
import urllib.request
import argparse

DEFAULT_MODEL_URL = "https://huggingface.co/spaces/Baraebld/Violence-Detection-model/resolve/main/models/hybrid_model3.pth"
TARGET_DIR = Path(__file__).resolve().parent.parent / "models"
TARGET_FILE = TARGET_DIR / "hybrid_model3.pth"

def main():
    parser = argparse.ArgumentParser(description="Download pre-trained weights for Violence Detection model")
    parser.add_argument("--url", default=DEFAULT_MODEL_URL, help="URL to download the model from")
    parser.add_argument("--force", action="store_true", help="Force redownload even if file exists")
    args = parser.parse_args()

    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    if TARGET_FILE.exists() and not args.force:
        size_mb = TARGET_FILE.stat().st_size / (1024 * 1024)
        print(f"[OK] Model already exists at: {TARGET_FILE} ({size_mb:.2f} MB)")
        return

    print(f"Downloading model checkpoint from: {args.url}")
    print(f"Destination: {TARGET_FILE}")

    def report_progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = downloaded / total_size * 100
            sys.stdout.write(f"\rDownloading... {percent:.1f}% ({downloaded / (1024*1024):.1f}/{total_size / (1024*1024):.1f} MB)")
        else:
            sys.stdout.write(f"\rDownloading... {downloaded / (1024*1024):.1f} MB")
        sys.stdout.flush()

    try:
        urllib.request.urlretrieve(args.url, str(TARGET_FILE), reporthook=report_progress)
        print("\n[OK] Model weights downloaded successfully!")
    except Exception as e:
        print(f"\n[ERROR] Download failed: {e}")
        print(f"Please manually download 'hybrid_model3.pth' (approx 162.7 MB) and place it in '{TARGET_DIR}'.")
        print(f"Available from: https://huggingface.co/spaces/Baraebld/Violence-Detection-model")
        sys.exit(1)

if __name__ == "__main__":
    main()
