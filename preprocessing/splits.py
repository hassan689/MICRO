"""Create a reproducible split that keeps DeepPCB groups together."""

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

try:
    from preprocessing.dataset import DEFAULT_DATA_ROOT, PROJECT_ROOT, DeepPCBDataset
except ImportError:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    DEFAULT_DATA_ROOT = PROJECT_ROOT / "data/raw/DeepPCB/PCBData"

    class DeepPCBDataset:
        def __init__(self, root=DEFAULT_DATA_ROOT, split="trainval"):
            self.root = Path(root)
            self.samples = []
            split_path = self.root / f"{split}.txt"
            for line_number, line in enumerate(split_path.read_text(encoding="utf-8").splitlines(), 1):
                if not line.strip():
                    continue
                fields = line.split()
                if len(fields) != 2:
                    raise ValueError(f"{split_path}:{line_number}: expected two paths")
                base = Path(fields[0])
                image_path = self.root / base.with_name(f"{base.stem}_test.jpg")
                label_path = self.root / fields[1]
                for path in (image_path, label_path):
                    if not path.is_file():
                        raise FileNotFoundError(f"Missing dataset file: {path}")
                self.samples.append((image_path, label_path))

DEFAULT_SPLIT_PATH = PROJECT_ROOT / "data/splits/deeppcb.json"


def grouped_indices(group_ids, val_fraction=0.2, seed=42):
    """Find a whole-group subset closest to the requested validation size."""
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between 0 and 1")
    groups = defaultdict(list)
    for index, group in enumerate(group_ids):
        groups[group].append(index)
    if len(groups) < 2:
        raise ValueError("At least two groups are needed for independent validation")
    names = sorted(groups)
    random.Random(seed).shuffle(names)
    # Subset-sum dynamic programming: one candidate per achievable sample count.
    choices = {0: ()}
    for name in names:
        for count, selected in list(choices.items()):
            choices.setdefault(count + len(groups[name]), selected + (name,))
    target = len(group_ids) * val_fraction
    count = min((n for n in choices if 0 < n < len(group_ids)),
                key=lambda n: (abs(n - target), n))
    val_groups = set(choices[count])
    train = [i for i, name in enumerate(group_ids) if name not in val_groups]
    val = [i for i, name in enumerate(group_ids) if name in val_groups]
    return train, val


def make_manifest(dataset, val_fraction=0.2, seed=42):
    paths = [image.relative_to(dataset.root).as_posix() for image, _ in dataset.samples]
    groups = [Path(path).parts[0] for path in paths]
    train, val = grouped_indices(groups, val_fraction, seed)
    return {
        "version": 1,
        "source": "trainval.txt",
        "source_sha256": hashlib.sha256((dataset.root / "trainval.txt").read_bytes()).hexdigest(),
        "seed": seed,
        "val_fraction": val_fraction,
        "grouping": "first directory in the official split path",
        "train_groups": sorted({groups[i] for i in train}),
        "val_groups": sorted({groups[i] for i in val}),
        "train_indices": train,
        "val_indices": val,
        "sample_paths": paths,
    }


def load_or_create_split(dataset, path=DEFAULT_SPLIT_PATH, val_fraction=0.2, seed=42):
    path = Path(path)
    expected = make_manifest(dataset, val_fraction, seed)
    if path.exists():
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest != expected:
            raise ValueError(f"Split settings or data changed: use a new --split-file instead of {path}")
        return manifest
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(expected, indent=2) + "\n", encoding="utf-8")
    return expected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_SPLIT_PATH)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    manifest = load_or_create_split(DeepPCBDataset(args.root), args.output,
                                    args.val_fraction, args.seed)
    print(f"Train: {len(manifest['train_indices'])}; validation: {len(manifest['val_indices'])}")
    print(f"Validation groups: {manifest['val_groups']}")
    print(f"Saved split: {args.output}")


if __name__ == "__main__":
    main()
