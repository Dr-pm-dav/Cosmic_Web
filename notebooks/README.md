# Notebooks — the walkthrough

Run in order; each is self-contained and mirrors the distilled code in the `cosmicweb/` package.

**Foundations (synthetic data, no downloads)**
- `01_foundations.ipynb` — density fields and the T-web classification
- `02_foundations_zeldovich.ipynb` — the Zel'dovich approximation
- `03_foundations_ingestion.ipynb` — data ingestion routes
- `04_unet_2d.ipynb` — 2D U-Net: recover the web from sparse tracers
- `05_field_2lpt.ipynb` — second-order (2LPT) field generator
- `06_unet_2lpt_sweep.ipynb` — retrain + sweep tracer density
- `07_unet_3d.ipynb` — 3D U-Net on sub-cubes
- `08_capstone_toy.ipynb` — the toy pipeline end to end

**Real data (downloads CAMELS; GPU recommended)**
- `09_real_data.ipynb` — real N-body density -> T-web -> 3D U-Net
- `10_scaling_harness.ipynb` — train/val/test, augmentation, mixed precision, confusion matrix
- `11_multifield.ipynb` — field registry, void contents, multi-head reconstruction
