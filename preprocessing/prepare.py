"""Validate the pinned DeepPCB data and write a deterministic processed manifest."""

import argparse
import hashlib
import json
from pathlib import Path

from preprocessing.dataset import CLASS_NAMES, DeepPCBDataset


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(raw_root):
    raw_root = Path(raw_root).resolve()
    datasets = {}
    for split in ("trainval", "test"):
        dataset = DeepPCBDataset(raw_root, split)
        boxes = 0
        for index in range(len(dataset)):
            _, target = dataset[index]
            boxes += len(target["labels"])
        split_file = raw_root / f"{split}.txt"
        datasets[split] = {
            "samples": len(dataset),
            "boxes": boxes,
            "split_sha256": sha256(split_file),
        }
    return {
        "dataset": "DeepPCB",
        "raw_root": str(raw_root),
        "class_names": {
            str(class_id): name
            for class_id, name in sorted(CLASS_NAMES.items())
        },
        "splits": datasets,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = build_manifest(args.raw_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Validated {manifest['dataset']} and wrote {args.output}")


if __name__ == "__main__":
    main()
