import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from autogluon.tabular import TabularPredictor

# Resolve paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(script_dir)

from color_to_spec import color_to_spec_grz
from sample_gmm import sample_group
from reconstruct_spectra import load_and_preprocess_spectra

def main():
    model_dir = os.path.join(project_root, "models", "des_grz_reconstructor")
    plot_dir = os.path.join(project_root, "plots")
    artifact_dir = "/Users/hsingwel/.gemini/antigravity/brain/db176578-ab2c-45ff-89c8-754918c29b15"
    os.makedirs(plot_dir, exist_ok=True)
    
    # 1. Load the model configuration
    config_path = os.path.join(model_dir, "config.pkl")
    if not os.path.exists(config_path):
        print(f"Error: Model config not found at {config_path}.")
        sys.exit(1)
        
    with open(config_path, "rb") as f:
        config = pickle.load(f)
    pca = config['pca']
    norm_wl = config.get('norm_wl', 0.926)
    
    # Get the wavelength grid
    processed_dir = os.path.join(project_root, "data", "processed")
    _, _, wl_grid = load_and_preprocess_spectra(processed_dir, norm_wl=norm_wl)
    
    # 2. Load AutoGluon predictors
    print("Loading AutoGluon PC predictors...")
    predictors = {}
    for pc_idx in range(5):
        target = f"pc_{pc_idx}"
        predictors[target] = TabularPredictor.load(os.path.join(model_dir, f"predictor_{target}"))
        
    groups = ['NIRB+', 'NIRB-', 'NIRF+', 'NIRF-']
    group_colors = {
        'NIRB+': '#f1c40f',  # Yellow
        'NIRB-': '#3498db',  # Blue
        'NIRF+': '#9c27b0',  # Purple
        'NIRF-': '#e74c3c'   # Red
    }
    
    plt.figure(figsize=(9, 6))
    
    for g_name in groups:
        print(f"Generating average spectrum and 1-sigma bands for {g_name}...")
        # Sample 1000 times
        samples = sample_group(g_name, n_samples=1000)
        gr = samples[:, 0]
        rz = samples[:, 2]
        
        # Convert colors to z-normalized reflectances
        ref_list = []
        for i in range(1000):
            ref, _ = color_to_spec_grz(gr[i], rz[i], 0, 0, norm_band='z')
            ref_list.append(ref)
        ref_arr = np.array(ref_list)
        
        # Add 5% photometric noise and re-normalize to z-band
        ref_g = ref_arr[:, 0]
        ref_r = ref_arr[:, 1]
        ref_z = ref_arr[:, 2]
        
        noisy_g = ref_g + np.random.normal(0, 0.05 * ref_g)
        noisy_r = ref_r + np.random.normal(0, 0.05 * ref_r)
        noisy_z = ref_z + np.random.normal(0, 0.05 * ref_z)
        
        # Prevent division by zero or negative values
        noisy_z = np.clip(noisy_z, 1e-4, None)
        
        ref_arr_noisy = np.zeros_like(ref_arr)
        ref_arr_noisy[:, 0] = noisy_g / noisy_z
        ref_arr_noisy[:, 1] = noisy_r / noisy_z
        ref_arr_noisy[:, 2] = 1.0
        
        # Construct features DataFrame matching predictor expected columns
        features_df = pd.DataFrame({
            'mag_g': ref_arr_noisy[:, 0],
            'mag_r': ref_arr_noisy[:, 1],
            'mag_z': ref_arr_noisy[:, 2]
        })
        
        # Predict PC1 to PC5
        group_latent_first5 = np.zeros((1000, 5))
        for pc_idx in range(5):
            target = f"pc_{pc_idx}"
            pred_all = predictors[target].predict(features_df)
            median_col = min(pred_all.columns, key=lambda c: abs(float(c) - 0.5))
            group_latent_first5[:, pc_idx] = pred_all[median_col].values.ravel()
            
        # Pad latent vectors to full PCA components count
        n_components = pca.n_components_
        group_latent_full = np.zeros((1000, n_components))
        group_latent_full[:, :5] = group_latent_first5
        
        # Inverse transform to reconstruct the spectra for all 1000 points
        recon_spectra = pca.inverse_transform(group_latent_full) # Shape (1000, 900)
        
        # Calculate mean and standard deviation spectrum
        mean_spec = np.mean(recon_spectra, axis=0)
        std_spec = np.std(recon_spectra, axis=0)
        
        # Print coordinates for verification
        mean_latent_first5 = np.mean(group_latent_first5, axis=0)
        print(f"  Mean Latent: PC1={mean_latent_first5[0]:.3f}, PC2={mean_latent_first5[1]:.3f}, PC3={mean_latent_first5[2]:.3f}")
        
        # Plot the mean reconstructed spectrum as a line
        plt.plot(wl_grid, mean_spec, color=group_colors[g_name], linewidth=2.0, label=g_name)
        
        # Plot the 1-sigma interval as a shaded region
        plt.fill_between(wl_grid, mean_spec - std_spec, mean_spec + std_spec, 
                         color=group_colors[g_name], alpha=0.15, edgecolor='none')
        
    plt.xlabel('Wavelength ($\mu$m)', fontsize=12, fontweight='bold')
    plt.ylabel('Normalized Reflectance (at 0.926 $\mu$m)', fontsize=12, fontweight='bold')
    plt.title('Reconstructed Average Spectra of GMM Subgroups (DES grz Model)', fontsize=13, fontweight='bold')
    plt.legend(frameon=True, fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xlim(0.4, 5.1)
    
    save_path = os.path.join(plot_dir, "gmm_average_spectra.png")
    plt.savefig(save_path, dpi=250, bbox_inches='tight')
    
    # Save as PDF
    pdf_path = save_path.replace(".png", ".pdf")
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.close()
    
    # Copy to artifact directory
    os.system(f"cp {save_path} {artifact_dir}/")
    os.system(f"cp {pdf_path} {artifact_dir}/")
    print(f"Saved average spectra plot to: {save_path} and .pdf, and copied to artifact directory.")

if __name__ == '__main__':
    main()
