from __future__ import annotations

import argparse
import os

from huggingface_hub import snapshot_download


def main() -> int:
    parser = argparse.ArgumentParser(description="Download or resume a Hugging Face model snapshot.")
    parser.add_argument("repo_id")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    path = snapshot_download(
        args.repo_id,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
        resume_download=True,
    )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
