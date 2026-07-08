from __future__ import annotations

import argparse
from pathlib import Path

from safetensors import safe_open


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()

    bad = []
    for path in sorted(args.directory.glob("model-*.safetensors")):
        try:
            with safe_open(str(path), framework="pt") as handle:
                keys = list(handle.keys())[:3]
            print(f"OK {path.name} {keys}")
        except Exception as exc:  # pragma: no cover - diagnostic script
            bad.append(path)
            print(f"BAD {path.name} {type(exc).__name__}: {exc}")

    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
