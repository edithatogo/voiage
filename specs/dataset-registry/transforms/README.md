# Dataset Registry Transform Scripts

This directory contains scripts for downloading and transforming open data
sources into the snapshot files used by tests and examples.

## Usage

Run scripts from the repository root:

```bash
# Transform all datasets
python specs/dataset-registry/transforms/refresh_all.py

# Transform a single dataset
python specs/dataset-registry/transforms/refresh_nhanes.py
```

## Requirements

Some transform scripts may require additional dependencies (e.g., `pandas`,
`requests`). These are documented in each script's header.

## Adding a New Dataset

1. Add a transform script following the pattern in `_template.py`.
2. Register the dataset in `../registry.json`.
3. Run the script to generate a snapshot.
4. Commit the snapshot to `../snapshots/`.

## Policy

- Snapshots are updated on an as-needed basis.
- Transform scripts should handle network errors gracefully.
- Generated snapshots must be small (<512 KB) for repository hygiene.
