"""Validate the pinned DeepPCB data and write a deterministic processed manifest."""

import argparse
import hashlib
import json
from pathlib import Path


CLASS_NAMES = {
    1: "Open",
    2: "Short",
    3: "Mousebite",
    4: "Spur",
    5: "Copper",
    6: "Pin hole",
}


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
        split_file = raw_root / f"{split}.txt"
        if not split_file.is_file():
            raise FileNotFoundError(f"Missing split file: {split_file}")
        samples = 0
        boxes = 0
        for line_number, line in enumerate(split_file.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            fields = line.split()
            if len(fields) != 2:
                raise ValueError(f"{split_file}:{line_number}: expected two paths")
            base = Path(fields[0])
            image_path = raw_root / base.with_name(f"{base.stem}_test.jpg")
            annotation_path = raw_root / fields[1]
            if not image_path.is_file():
                raise FileNotFoundError(f"Missing dataset file: {image_path}")
            if not annotation_path.is_file():
                raise FileNotFoundError(f"Missing dataset file: {annotation_path}")
            if image_path.read_bytes()[:2] != b"\xff\xd8":
                raise ValueError(f"{image_path}: invalid JPEG signature")
            samples += 1
            for annotation_line, annotation in enumerate(
                annotation_path.read_text(encoding="utf-8").splitlines(), 1
            ):
                if not annotation.strip():
                    continue
                values = annotation.split()
                if len(values) != 5:
                    raise ValueError(
                        f"{annotation_path}:{annotation_line}: expected five integers"
                    )
                try:
                    x1, y1, x2, y2, class_id = map(int, values)
                except ValueError as error:
                    raise ValueError(
                        f"{annotation_path}:{annotation_line}: expected five integers"
                    ) from error
                if class_id not in CLASS_NAMES:
                    raise ValueError(
                        f"{annotation_path}:{annotation_line}: invalid class {class_id}"
                    )
                if not (0 <= x1 < x2 and 0 <= y1 < y2):
                    raise ValueError(
                        f"{annotation_path}:{annotation_line}: invalid bounding box"
                    )
                boxes += 1
        datasets[split] = {
            "samples": samples,
            "boxes": boxes,
            "split_sha256": sha256(split_file),
        }
    return {
        "dataset": "DeepPCB",
        "raw_root": "data/raw/DeepPCB/PCBData",
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
