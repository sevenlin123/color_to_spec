---
name: spectra_reconstruction
description: Reconstructs TNO reflectance spectra from magnitude colors (like LSST or DES) using AutoGluon-trained PCA model.
---
# TNO Reflectance Spectra Reconstructor Skill

This skill allows the agent to reconstruct full-resolution TNO reflectance spectra ($0.3\ \mu\text{m}$ to $5.2\ \mu\text{m}$) with uncertainty bounds from input magnitude colors (e.g., $g-r, r-i, i-z, r-z$ in LSST or $g-r, r-z$ in DES).

## How it works
The reconstruction runs a Python script located at `scripts/reconstruct.py` within the skill directory. The script:
1. Parses input colors and propagates uncertainties (e.g., $r-z = (r-i) + (i-z)$).
2. Converts colors to normalized reflectances.
3. Automatically trains a new PCA + AutoGluon reconstructor model for the specific filter combination if it doesn't already exist in the `models/` cache folder (takes 1-2 minutes).
4. Runs Monte Carlo realizations to predict full spectra, generating a median spectrum and 1-sigma / 90% confidence bands.
5. Saves results as a CSV catalog and a PNG plot.

## Setup
Ensure the Python environment has the necessary packages installed (specifically `autogluon`, `scipy`, `pandas`, `scikit-learn`).

## How to Invoke

To reconstruct a spectrum, run the script from the command line:

```bash
python3 .agents/skills/spectra_reconstruction/scripts/reconstruct.py \
  --name "2025 NN80" \
  --system "LSST" \
  --colors "g-r=0.46991429,0.04782941 r-i=0.29678571,0.03845473 iz=0.120,0.0555" \
  --out-dir "./plots"
```

### Argument Reference
* `--name`: (Optional) The name of the asteroid target, used in titles and output filenames.
* `--system`: The photometry filter system (`LSST` or `DES`).
* `--colors`: Space-separated magnitude color definitions in the format `color=value,error`.
  - Supported colors for LSST: `g-r`, `r-i`, `i-z`, `r-z`, `iz` (treated as $i-z$).
  - Supported colors for DES: `g-r`, `r-z`.
* `--out-dir`: (Optional) The directory to save the output CSV and PNG plot. Defaults to the workspace `./plots/` directory.

## Outputs
The script outputs:
1. **CSV Catalog**: `[name]_reconstructed_spectrum.csv` containing:
   - `wavelength_um`
   - `median_reflectance`
   - `lower_1sigma`, `upper_1sigma`
   - `lower_90pct`, `upper_90pct`
2. **Comparison Plot**: `[name]_reconstructed_spectrum.png` showing the median reconstruction, uncertainty bands, and the input photometry data points.
