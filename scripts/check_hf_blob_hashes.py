"""Verify Hugging Face snapshot symlink targets against blob SHA256 names."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--pattern", default="*.safetensors")
    args = parser.parse_args()

    snapshot = args.snapshot.expanduser().resolve()
    if not snapshot.exists():
        parser.error(f"snapshot does not exist: {snapshot}")

    checked = 0
    bad = 0
    for entry in sorted(snapshot.glob(args.pattern)):
        target = entry.resolve()
        expected = target.name
        actual = sha256_file(target)
        checked += 1
        if actual == expected:
            print(f"OK  {entry.name}")
        else:
            bad += 1
            print(f"BAD {entry.name} expected={expected} actual={actual}")

    print(f"checked={checked} bad={bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
