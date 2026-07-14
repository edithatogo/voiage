.. _dataset-registry:

Dataset Registry
================

This document describes the dataset registry for voiage, including synthetic
datasets and open-data source mappings.

License
-------

All data sources referenced in this registry carry their own licenses and terms
of use. Users must comply with the original license terms for each dataset.

- **NHANES**: Public Domain (CDC/ATSDR)
- **MEPS**: Public Domain (AHRQ)
- **ClinicalTrials.gov**: Public Domain (U.S. National Library of Medicine)
- **World Bank**: CC BY-4.0
- **NOAA**: Public Domain (U.S. Government)
- **EPA**: Public Domain (U.S. Government)

Citation
--------

When using data from this registry in published work, please cite both the
original data source (as listed in ``specs/dataset-registry/registry.json``)
and the voiage library.

Transform
---------

Live data refresh and transformation scripts are located in
``specs/dataset-registry/transforms/``. These scripts download the latest
data from each source and produce the snapshot files used by tests and
examples.

Refresh Policy
--------------

Snapshots in ``specs/dataset-registry/snapshots/`` are updated on an as-needed
basis. Re-running the transform scripts produces fresh snapshots that can be
reviewed and committed when data sources change.
