# Asteroid Reflectance Spectra Reconstructor

This is a lightweight distribution of the Asteroid Reflectance Spectra Reconstructor. It allows you to reconstruct full-resolution reflectance spectra ($0.3\ \mu\text{m}$ to $5.2\ \mu\text{m}$) and uncertainty intervals from broadband magnitude colors (LSST or DES).

## Package Directory Structure
* `.agents/skills/spectra_reconstruction/` - AI Agent Skill configuration and instructions (allows AI agents to automatically load and use this tool).
  - `.agents/skills/spectra_reconstruction/scripts/reconstruct.py` - Core command-line wrapper script.
* `src/` - Core library source code.
* `models/` - Pre-trained AutoGluon predictors and the base PCA/KDE prior.

## Installation
Ensure you have a Python 3 environment with the necessary dependencies installed:
```bash
pip install autogluon scipy pandas scikit-learn matplotlib
```

## How to Run Reconstructions
You can run reconstructions directly from the command line using the wrapper script:

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
For each reconstruction, the tool generates:
1. **CSV Catalog**: `[name]_reconstructed_spectrum.csv` containing the median spectrum and 1-sigma / 90% confidence boundaries.
2. **Comparison Plot**: `[name]_reconstructed_spectrum.png` showing the median spectrum, uncertainty bands, and the input photometry data points.
