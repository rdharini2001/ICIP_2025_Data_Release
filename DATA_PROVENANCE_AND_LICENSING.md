# Origin, attribution and licensing

## Original recordings

The underlying surveillance videos were collected and published by Fredrik Svanström, Fernando Alonso-Fernandez, and Cristofer Englund as the **Drone Detection Dataset**. The original public repository releases covered source media under **CC0 1.0 Universal** and requests scholarly attribution. This original CC0 dedication is not withdrawn by later curation or packaging.

Primary reference: Svanström, F., Alonso-Fernandez, F., and Englund, C. *A Dataset for Multi-Sensor Drone Detection*, Data in Brief 39 (2021) 107521. https://doi.org/10.1016/j.dib.2021.107521.

Source: https://github.com/DroneDetectionThesis/Drone-detection-dataset

## AVID preparation and paper

The bird/drone portion was curated for the IEEE ICIP 2025 Video and Image Processing Cup. This paper documents the 32,040 corresponding clean RGB-infrared pairs drawn from the challenge's Train and Validation splits, retaining existing annotations and video identifiers. The authors also introduce object-relative alignment measurements and the simulated AVID-C perturbation suite.

This release does **not** claim that its authors captured the original footage or newly annotated its bounding boxes. The original Test split is excluded because reliable cross-modal frame correspondence is unresolved.

## Distinct licensing layers

- **Original source imagery:** CC0 1.0 Universal, where covered by the source authors' dedication.
- **AVID dataset package:** the current Hugging Face listing is marked CC BY-NC 4.0 for the package; this does not replace or restrict the original CC0 source rights.
- **Analysis software and scripts:** MIT license (`LICENSE`).

Dataset: https://huggingface.co/datasets/Dharini24/AVID  
Code: https://github.com/rdharini2001/ICIP_2025_Data_Release

For research attribution, cite both the original data publication and the AVID paper. A published AVID DOI/arXiv identifier has not yet been inserted in this pre-submission package.
