#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from moladt.io.sdf import read_sdf_record


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit parser-inferred MolADT bonding systems for SDF files.")
    parser.add_argument("path", help="SDF file or directory of .sdf files")
    parser.add_argument("--limit", type=int, help="Maximum files to inspect")
    args = parser.parse_args()

    paths = tuple(_iter_sdf_paths(Path(args.path), args.limit))
    tag_counts: Counter[str] = Counter()
    files_with_inference = 0
    for path in paths:
        record = read_sdf_record(path)
        inferred = [
            system.tag
            for _, system in record.molecule.systems
            if system.tag is not None and (system.tag == "pi_ring" or system.tag.startswith("inferred_"))
        ]
        if inferred:
            files_with_inference += 1
        tag_counts.update(tag for tag in inferred if tag is not None)
        print(f"{path}\t{','.join(inferred) if inferred else '-'}")

    print("")
    print(f"files={len(paths)}")
    print(f"files_with_inference={files_with_inference}")
    for tag, count in sorted(tag_counts.items()):
        print(f"{tag}={count}")
    return 0


def _iter_sdf_paths(path: Path, limit: int | None) -> list[Path]:
    paths = [path] if path.is_file() else sorted(path.rglob("*.sdf"))
    if limit is not None:
        return paths[:limit]
    return paths


if __name__ == "__main__":
    raise SystemExit(main())
