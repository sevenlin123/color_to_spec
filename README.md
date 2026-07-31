# TNO Reflectance Spectra Reconstructor from Optical Colors

This repository provides a reusable software package and AI skill for **optical-color-to-spectral inference** of Trans-Neptunian Objects (TNOs) and asteroids. It includes:
- **Trained PCA Spectral Manifold**: Latent space representation and Kernel Density Estimation (KDE) prior (`base_pca_kde.pkl`).
- **Bayesian Inference Pipeline**: Regressors mapping broadband magnitude colors (LSST or DES) to full-resolution reflectance spectra ($0.4\ \mu\text{m}$ to $5.1\ \mu\text{m}$) with uncertainty bounds.
- **Spectral & Photometric Generation Utilities**: Modules for sampling posterior spectral distributions, reconstructing reflectance spectra, and generating synthetic optical color distributions via reverse projection.

> Rather than predicting a single spectrum, the model estimates a posterior probability distribution over the optical–NIR spectral manifold, enabling uncertainty-aware compositional inference and follow-up target prioritization.

---

## 🛠️ Package Directory Structure

- `.agents/skills/spectra_reconstruction/`: AI Agent Skill configuration and instructions.
  - `scripts/reconstruct.py`: Primary command-line interface for running reconstructions.
- `src/`: Core Python library modules:
  - `reconstruct_spectra.py`: Bayesian inference pipeline and spectrum reconstruction engine.
  - `spectra_generator.py`: PCA spectral manifold and synthetic spectrum / color generation (`PCASpectrumGenerator`).
  - `reverse_projection.py`: Utilities for projecting latent PCA posterior distributions back into synthetic optical color space.
  - `color_to_spec.py`: Photometric flux conversions and normalization utilities.
  - `metrics.py`: Information-theoretic metrics (Entropy, Information Gain, KL Divergence, Surprisal).
  - `sample_gmm.py`: Latent GMM/KDE sampling utilities.
  - `generate_gmm_average_spectra.py`: Class-average spectral distribution utilities.
  - `plot_latent_dist.py`: Visualization tools for latent posteriors and distributions.
- `models/`: Pre-trained reconstructors and the base PCA/KDE prior (`base_pca_kde.pkl`).

---

## ⚙️ Installation

Ensure you have a Python 3 environment with the necessary dependencies installed:
```bash
pip install autogluon scipy pandas scikit-learn matplotlib
```

---

## 🚀 Usage

### 1. Command-Line Spectral Inference
Reconstruct full-resolution spectra and uncertainty bands directly from magnitude colors:

```bash
python3 .agents/skills/spectra_reconstruction/scripts/reconstruct.py \
  --name "2025 NN80" \
  --system "LSST" \
  --colors "g-r=0.4699,0.0478 r-i=0.2968,0.0384 iz=0.120,0.0555" \
  --out-dir "./plots"
```

#### Argument Reference:
* `--name`: Target object identifier (used in filenames and plot titles).
* `--system`: Photometric filter system (`LSST` or `DES`).
* `--colors`: Space-separated magnitude color definitions formatted as `color=value,error`.
* `--out-dir`: Output directory for generated CSVs and plots.

### 2. Generating Synthetic Optical Color Distributions
Use `spectra_generator.py` or `reverse_projection.py` in Python to sample from the trained PCA manifold and compute synthetic colors:

```python
from src.spectra_generator import PCASpectrumGenerator

# Load pre-trained PCA spectral manifold
generator = PCASpectrumGenerator.load_model("models/base_pca_kde.pkl")

# Generate synthetic spectra and corresponding optical colors
synthetic_spectra = generator.generate(n_samples=1000)
synthetic_colors = generator.generate_colors(n_samples=1000, system="LSST")
```

---

## 📊 Outputs

For each reconstruction run, the tool outputs:
1. **CSV Catalog**: `[name]_reconstructed_spectrum.csv` containing median reflectance and $1\sigma$ / $90\%$ confidence bounds across wavelengths ($0.4\ \mu\text{m}$ to $5.1\ \mu\text{m}$).
2. **Comparison Plot**: `[name]_reconstructed_spectrum.png` showing reconstructed median spectrum, uncertainty intervals, and input photometry points.
3. **2D Latent Space Plot**: `[name]_latent_pc1_pc2.png` showing the target object's posterior Monte Carlo distribution and mean overlaid on the TNO population prior in PC1 vs PC2 space.
