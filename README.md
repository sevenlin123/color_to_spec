# Optical-to-NIR Spectra Reconstruction for Trans-Neptunian Objects (TNOs)

This repository contains a machine learning framework designed to reconstruct full-resolution reflectance spectra (from $0.4\ \mu\text{m}$ to $5.1\ \mu\text{m}$) of Trans-Neptunian Objects (TNOs) and asteroids from sparse magnitude colors (such as LSST `g-r`, `r-i`, `i-z`, `r-z` or DES `g-r`, `r-z`) with uncertainty quantification.

Rather than predicting a single spectrum, the model estimates a posterior probability distribution over the optical–NIR spectral manifold, enabling uncertainty-aware compositional inference and follow-up target prioritization.

---

## 🌌 Scientific Goal

The primary scientific objective of this project is to quantify how much information optical photometry (e.g., from the Vera C. Rubin Observatory / LSST) contains about the infrared spectral properties of TNOs. By mapping sparse colors to full-resolution spectra, we can:
1. Interpolate and reconstruct NIR spectral properties for large photometric surveys.
2. Quantify reconstruction certainty using information-theoretic metrics (e.g., Entropy, Information Gain, KL Divergence).
3. Detect rare/novel spectral types to prioritize targets for future space telescope follow-ups (e.g., JWST).

---

## 🛠️ Methodology

The reconstruction framework operates in a low-dimensional latent space:

1. **Dimensionality Reduction**: Principal Component Analysis (PCA) is applied to high-resolution reference spectra (10 components).
2. **Latent Space Modeling**: A Gaussian copula / Kernel Density Estimation (KDE) model learns the joint distribution of the PC coefficients.
3. **Color-to-Latent Mapping**: Regressors (AutoGluon Tabular / LightGBM) are trained to map observed photometry colors directly to PCA coefficients.
4. **Uncertainty Quantification**: Monte Carlo (MC) realizations propagate input photometric uncertainties to output spectra, producing:
   - Median reflectance spectra
   - $1\sigma$ (68%) confidence bands
   - $90\%$ confidence bands
   - Latent space posteriors and information metrics

---

## 📂 Repository Structure

- `src/`: Python source code containing training, reconstruction, and experiment scripts.
  - `reconstruct_spectra.py`: Core pipeline logic for loading models and executing reconstructions.
  - `spectra_generator.py`: PCA-based spectrum generation using KDE sampling.
  - `train_and_evaluate.py`: Training script for the AutoGluon regressors.
  - `metrics.py`: Script to compute information metrics (Entropy, KL Divergence, etc.).
- `notebook/`: Jupyter notebooks showing model training and verification workflows.
  - `reconstructor.ipynb`: Walkthrough of the training and reconstruction process.
  - `validation.ipynb`: Validation and diagnostic checks.
- `models/`: Pre-trained models and configuration metadata.
  - `base_pca_kde.pkl`: The core PCA projection and KDE latent model.
  - `*reconstructor/config.pkl`: Reconstructor configuration parameters.
  - `*reconstructor/correlation_matrix.npy`: PCA coefficient correlation matrices.
- `doc/`: Documentation files.
  - `experiment_plan.md`: Details of experiments A, B, and C.
  - `metrics.md`: Definition of information-theoretic metrics.
- `plots/`: Generated visualization products and csv catalogs from experiments.

---

## 🚀 Getting Started

### Prerequisites

Ensure you have the required dependencies installed:
```bash
pip install autogluon scipy pandas scikit-learn numpy matplotlib
```

### Running Reconstruction

You can run the reconstruction CLI tool to generate full-resolution spectra from colors. For example, using the custom `spectra_reconstruction` skill:

```bash
python3 .agents/skills/spectra_reconstruction/scripts/reconstruct.py \
  --name "2025 NN80" \
  --system "LSST" \
  --colors "g-r=0.4699,0.0478 r-i=0.2968,0.0384 i-z=0.1200,0.0555" \
  --out-dir "./plots"
```

#### Argument Reference:
* `--name`: (Optional) Name of the target asteroid, used in titles and output filenames.
* `--system`: The photometric system used (`LSST` or `DES`).
* `--colors`: Space-separated magnitude color definitions formatted as `color=value,error`.
* `--out-dir`: Directory to save the output CSV catalog and PNG plot.

---

## 📊 Outputs

For each reconstruction run, the tool outputs:
1. **CSV Catalog**: `[name]_reconstructed_spectrum.csv` containing median reflectance and $1\sigma$ / $90\%$ confidence bounds across wavelengths ($0.4\ \mu\text{m}$ to $5.1\ \mu\text{m}$).
2. **Comparison Plot**: `[name]_reconstructed_spectrum.png` showing reconstructed median spectrum, uncertainty intervals, and input photometry points.
3. **2D Latent Space Plot**: `[name]_latent_pc1_pc2.png` showing the target object's posterior Monte Carlo distribution and mean overlaid on the TNO population prior in PC1 vs PC2 space.
