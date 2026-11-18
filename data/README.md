# Datasets

Keep downloaded data in `raw/` and generated training inputs in `processed/`.
Both directories are excluded from project Git history. This document records
the dataset source and structure without committing thousands of data files.

## DeepPCB

- Source: https://github.com/tangsanli5201/DeepPCB
- Imported revision: `08e98c4db5922613fb97176eb3d6497d48260cb1`
- Local location: `raw/DeepPCB/`
- Purpose: detection and classification of visible PCB defects.
- Preserved upstream files: `PCBData/`, `evaluation/`, `README.md`, and `LICENSE`.
- Removed upstream extras: Git metadata, presentation figures, and the annotation
  editor with its example files. These remain available from the source repository.

```text
data/
|-- README.md
`-- raw/
    `-- DeepPCB/
        |-- LICENSE
        |-- README.md
        |-- evaluation/          # Upstream benchmark scripts and test ground truth
        `-- PCBData/
            |-- trainval.txt
            |-- test.txt
            `-- group*/          # Image pairs and bounding-box annotation files
```

The upstream README describes 1,500 aligned image pairs: 1,000 training/validation
examples and 500 test examples. Preserve the provided split files.

Each example consists of:

- `<id>_test.jpg`: image to inspect for defects.
- `<id>_temp.jpg`: corresponding defect-free reference image.
- `<id>.txt`: bounding boxes and defect class IDs.

Split entries use a base image name such as `<id>.jpg`; loaders must resolve that
base name to the `_test.jpg` and `_temp.jpg` files. The second field is the
annotation path. All paths are relative to `PCBData/`.

Annotation fields are `x1 y1 x2 y2 class_id` in the local text files, where
coordinates are pixel positions. The upstream README describes the same five
fields using commas in its format description.

| ID | Defect |
| --- | --- |
| 1 | Open |
| 2 | Short |
| 3 | Mousebite |
| 4 | Spur |
| 5 | Spurious copper |
| 6 | Pin hole |

Class 0 is reserved for background and is not used in the annotations.
These are detection boxes, not segmentation masks or optimized-layout targets.

## Training preparation

The raw dataset is versioned with DVC. From the repository root, run
`dvc pull` to retrieve the pinned DeepPCB revision, then run `dvc repro` to
validate both official splits and create the tracked processed manifest at
`data/processed/deeppcb_manifest.json`. The imported dataset revision is
recorded in `data/raw/DeepPCB.dvc`; do not commit the downloaded raw files.

1. Read the original split files and resolve the image pairs and annotations.
2. Create a validation subset from `trainval.txt`; keep the official test set
   reserved for final evaluation. Keep related crops and augmented versions
   together when defining any new split.
3. Apply geometric augmentation consistently to both images and their boxes.
4. Write converted annotations or resized images to `processed/`, preserving raw
   files. Update box coordinates whenever image geometry changes.

The loader in `preprocessing/dataset.py` reads defective images and detection
targets. Run `python -m preprocessing.dataset --check-all` from the project root
to validate the training/validation split and save an annotated preview. It does
not itself apply random augmentation. `preprocessing/splits.py` creates the
separate training/validation assignment in `data/splits/deeppcb.json`; the default
is 800 training and 200 validation images, grouped by the first path directory.
Validation uses `group00041`. The source split's SHA256 and sample order are
recorded and checked before training. Original split files are never edited.
`training/train.py` uses this assignment and leaves the official test set unused.
DeepPCB supports the image defect-detection task; graph learning, routing, BOM
fusion, and yield prediction require additional data.

## Attribution and usage

Retain the upstream `LICENSE` and `README.md` when copying the dataset.
The repository includes an MIT license, while its README separately states that
the dataset is for research purposes only. Preserve both notices and do not
assume the software license removes the dataset usage restriction.
