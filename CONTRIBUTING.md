# Contributing

This project has an initial defect-detection training pipeline. Start with the current status and roadmap
in [README.md](README.md).

## Workflow

1. Use a dedicated branch for a focused change.
2. Create a virtual environment and install the dependencies needed for the work.
3. Put dataset loading and transforms in `preprocessing/`, model definitions in
   `models/`, training code in `training/`, and metrics in `evaluation/`.
4. Verify the affected behavior and describe the checks performed. Run dataset
   and training tests with `python -m unittest discover -s tests -v`. Use
   `python -m training.train --smoke-test` for a short end-to-end training check.
5. Update documentation when adding commands, dependencies, or dataset formats.

## Data and experiments

- Keep downloaded datasets in `data/raw/` and document their source, revision,
  expected format, and usage terms in `data/README.md`.
- Preserve raw images, annotations, and official split files. Write transformed
  data to `data/processed/`.
- Keep related examples and augmentations together when defining new splits.
  Never use the held-out test set for model selection.
- Keep datasets, virtual environments, checkpoints, logs, and generated outputs
  out of Git. Commit source code, configuration, and reproducibility notes.
- Retain third-party attribution and license notices.

## Code and review

Use clear names, small functions, and explicit configuration. Avoid import-time
side effects. Add tests for meaningful behavior as implementations are added,
and report known limitations rather than claiming unimplemented features.
