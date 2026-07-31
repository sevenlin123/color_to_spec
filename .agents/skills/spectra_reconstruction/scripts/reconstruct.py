import os
import sys
import argparse
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Resolve paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(script_dir))))
if not os.path.exists(os.path.join(project_root, "src")):
    project_root = "/Users/hsingwel/Documents/optical_to_NIR"
sys.path.append(os.path.join(project_root, "src"))

from reconstruct_spectra import load_and_preprocess_spectra, train_reconstructor_model
from color_to_spec import color_to_spec, color_to_spec_gri, color_to_spec_grz, color_to_spec_grizy
from autogluon.tabular import TabularPredictor

def parse_colors(color_string):
    """
    Parses color string like "g-r=0.4699,0.0478 r-i=0.2967,0.0384 i-z=0.120,0.0555"
    Returns a dict of colors: { 'g-r': (val, err), ... }
    """
    colors = {}
    for part in color_string.strip().split():
        if '=' not in part:
            continue
        name, val_err = part.split('=')
        val_str, err_str = val_err.split(',')
        colors[name.lower()] = (float(val_str), float(err_str))
    return colors

def main():
    parser = argparse.ArgumentParser(description="Reconstruct asteroid spectra from colors.")
    parser.add_argument("--name", type=str, default="asteroid", help="Name of the object.")
    parser.add_argument("--system", type=str, default="LSST", choices=["LSST", "DES"], help="Observational system.")
    parser.add_argument("--colors", type=str, required=True, help="Space-separated colors, e.g., 'g-r=0.4699,0.0478 r-i=0.2968,0.0385 i-z=0.120,0.0555'")
    parser.add_argument("--out-dir", type=str, default=None, help="Output directory for plots and CSV.")
    
    args = parser.parse_args()
    
    # 1. Parse colors
    color_dict = parse_colors(args.colors)
    print(f"Parsed colors: {color_dict}")
    
    # Determine bands subset
    if args.system == "LSST":
        if "g-r" in color_dict and "r-i" in color_dict:
            if "i-z" in color_dict or "r-z" in color_dict or "iz" in color_dict:
                if "z-y" in color_dict or "zy" in color_dict:
                    bands_subset = ["g", "r", "i", "z", "y"]
                    model_name = "lsst_reconstructor"
                else:
                    bands_subset = ["g", "r", "i", "z"]
                    model_name = "lsst_griz_reconstructor"
            else:
                bands_subset = ["g", "r", "i"]
                model_name = "lsst_gri_reconstructor"
        else:
            print("Error: For LSST system, 'g-r' and 'r-i' colors are required.")
            sys.exit(1)
        norm_wl = 0.75
        norm_band = 'i'
    elif args.system == "DES":
        if "g-r" in color_dict and "r-z" in color_dict:
            bands_subset = ["g", "r", "z"]
            model_name = "des_grz_reconstructor"
        else:
            print("Error: For DES system, 'g-r' and 'r-z' colors are required.")
            sys.exit(1)
        norm_wl = 0.926
        norm_band = 'z'
        
    model_dir = os.path.join(project_root, "models", model_name)
    print(f"Using model directory: {model_dir}")
    
    # 2. Check/Train model
    config_path = os.path.join(model_dir, "config.pkl")
    if not os.path.exists(config_path):
        print(f"Model config not found. Training new model '{model_name}' on the fly...")
        processed_dir = os.path.join(project_root, "data", "processed")
        spectra, file_names, wl_grid = load_and_preprocess_spectra(processed_dir, norm_wl=norm_wl)
        config, predictors, correlation_matrix = train_reconstructor_model(
            spectra=spectra,
            wl_grid=wl_grid,
            system_name=args.system,
            n_components=10,
            model_dir=model_dir,
            file_names=file_names,
            bands_subset=bands_subset,
            norm_wl=norm_wl
        )
    else:
        with open(config_path, "rb") as f:
            config = pickle.load(f)
            
    pca = config['pca']
    wl_grid = config['wl_grid']
    
    # Load predictors
    predictors = {}
    for pc_idx in range(10):
        target = f"pc_{pc_idx}"
        predictors[target] = TabularPredictor.load(os.path.join(model_dir, f"predictor_{target}"))
        
    # 3. Generate MC realizations
    n_samples = 10000
    np.random.seed(42)
    
    gr_val, gr_err = color_dict["g-r"]
    gr_samples = np.random.normal(gr_val, gr_err, size=n_samples)
    
    if len(bands_subset) == 5: # grizy
        ri_val, ri_err = color_dict["r-i"]
        ri_samples = np.random.normal(ri_val, ri_err, size=n_samples)
        
        if "iz" in color_dict:
            iz_val, iz_err = color_dict["iz"]
        else:
            iz_val, iz_err = color_dict["i-z"]
        iz_samples = np.random.normal(iz_val, iz_err, size=n_samples)
        
        if "zy" in color_dict:
            zy_val, zy_err = color_dict["zy"]
        else:
            zy_val, zy_err = color_dict["z-y"]
        zy_samples = np.random.normal(zy_val, zy_err, size=n_samples)
        
        ref_list = []
        for g_r, r_i, i_z, z_y in zip(gr_samples, ri_samples, iz_samples, zy_samples):
            ref_val, _ = color_to_spec_grizy(g_r, r_i, i_z, z_y, 0.0, 0.0, 0.0, 0.0, norm_band=norm_band)
            ref_list.append(ref_val)
        ref_arr = np.array(ref_list)
        
        features_df = pd.DataFrame({
            'mag_g': ref_arr[:, 0],
            'mag_r': ref_arr[:, 1],
            'mag_i': ref_arr[:, 2],
            'mag_z': ref_arr[:, 3],
            'mag_y': ref_arr[:, 4]
        })
        
        ref_norm, ref_err_norm = color_to_spec_grizy(gr_val, ri_val, iz_val, zy_val, gr_err, ri_err, iz_err, zy_err, norm_band=norm_band)
        filters_wl = np.array([0.481, 0.622, 0.756, 0.868, 0.975])
        
    elif len(bands_subset) == 4: # griz
        ri_val, ri_err = color_dict["r-i"]
        ri_samples = np.random.normal(ri_val, ri_err, size=n_samples)
        
        if "iz" in color_dict:
            iz_val, iz_err = color_dict["iz"]
            iz_samples = np.random.normal(iz_val, iz_err, size=n_samples)
            rz_samples = ri_samples + iz_samples
            rz_val = ri_val + iz_val
            rz_err = (ri_err**2 + iz_err**2)**0.5
        else:
            rz_val, rz_err = color_dict["r-z"]
            rz_samples = np.random.normal(rz_val, rz_err, size=n_samples)
            
        ref_list = []
        for g_r, r_i, r_z in zip(gr_samples, ri_samples, rz_samples):
            ref_val, _ = color_to_spec(g_r, r_i, r_z, 0.0, 0.0, 0.0, norm_band=norm_band)
            ref_list.append(ref_val)
        ref_arr = np.array(ref_list)
        
        features_df = pd.DataFrame({
            'mag_g': ref_arr[:, 0],
            'mag_r': ref_arr[:, 1],
            'mag_i': ref_arr[:, 2],
            'mag_z': ref_arr[:, 3]
        })
        
        ref_norm, ref_err_norm = color_to_spec(gr_val, ri_val, rz_val, gr_err, ri_err, rz_err, norm_band=norm_band)
        filters_wl = np.array([0.481, 0.622, 0.756, 0.868])
        
    elif bands_subset == ["g", "r", "i"]:
        ri_val, ri_err = color_dict["r-i"]
        ri_samples = np.random.normal(ri_val, ri_err, size=n_samples)
        
        ref_list = []
        for g_r, r_i in zip(gr_samples, ri_samples):
            ref_val, _ = color_to_spec_gri(g_r, r_i, 0.0, 0.0, norm_band=norm_band)
            ref_list.append(ref_val)
        ref_arr = np.array(ref_list)
        
        features_df = pd.DataFrame({
            'mag_g': ref_arr[:, 0],
            'mag_r': ref_arr[:, 1],
            'mag_i': ref_arr[:, 2]
        })
        
        ref_norm, ref_err_norm = color_to_spec_gri(gr_val, ri_val, gr_err, ri_err, norm_band=norm_band)
        filters_wl = np.array([0.481, 0.622, 0.756])
        
    elif bands_subset == ["g", "r", "z"]: # DES grz
        rz_val, rz_err = color_dict["r-z"]
        rz_samples = np.random.normal(rz_val, rz_err, size=n_samples)
        
        ref_list = []
        for g_r, r_z in zip(gr_samples, rz_samples):
            ref_val, _ = color_to_spec_grz(g_r, r_z, 0.0, 0.0, norm_band=norm_band)
            ref_list.append(ref_val)
        ref_arr = np.array(ref_list)
        
        features_df = pd.DataFrame({
            'mag_g': ref_arr[:, 0],
            'mag_r': ref_arr[:, 1],
            'mag_z': ref_arr[:, 2]
        })
        
        ref_norm, ref_err_norm = color_to_spec_grz(gr_val, rz_val, gr_err, rz_err, norm_band=norm_band)
        filters_wl = np.array([0.473, 0.642, 0.926])
        
    ref_norm = np.array(ref_norm)
    ref_err_norm = np.array(ref_err_norm)
    
    # Predict PC coordinates
    group_latent = np.zeros((n_samples, 10))
    for pc_idx in range(10):
        target = f"pc_{pc_idx}"
        pred_all = predictors[target].predict(features_df)
        median_col = min(pred_all.columns, key=lambda c: abs(float(c) - 0.5))
        group_latent[:, pc_idx] = pred_all[median_col].values.ravel()
        
    # Reconstruct spectra
    idx_norm = np.argmin(np.abs(wl_grid - norm_wl))
    recon_samples = pca.inverse_transform(group_latent)
    recon_samples = np.clip(recon_samples, 0.0, None)
    recon_samples = recon_samples / recon_samples[:, idx_norm].reshape(-1, 1)
    
    median_spec = np.median(recon_samples, axis=0)
    lower_spec_16 = np.percentile(recon_samples, 16, axis=0)
    upper_spec_84 = np.percentile(recon_samples, 84, axis=0)
    lower_spec_5 = np.percentile(recon_samples, 5, axis=0)
    upper_spec_95 = np.percentile(recon_samples, 95, axis=0)
    
    # Save results
    out_dir = args.out_dir
    if out_dir is None:
        out_dir = os.path.join(project_root, "plots")
    os.makedirs(out_dir, exist_ok=True)
    
    safe_name = args.name.replace(" ", "_")
    csv_path = os.path.join(out_dir, f"{safe_name}_reconstructed_spectrum.csv")
    df_out = pd.DataFrame({
        'wavelength_um': wl_grid,
        'median_reflectance': median_spec,
        'lower_1sigma': lower_spec_16,
        'upper_1sigma': upper_spec_84,
        'lower_90pct': lower_spec_5,
        'upper_90pct': upper_spec_95
    })
    df_out.to_csv(csv_path, index=False)
    print(f"Saved CSV to: {csv_path}")
    
    # Plotting
    plt.figure(figsize=(8, 5))
    plt.plot(wl_grid, median_spec, color='black', linewidth=2.5, label=f"{args.name} Reconstruction")
    plt.fill_between(wl_grid, lower_spec_16, upper_spec_84, color='black', alpha=0.15, label="1-sigma uncertainty")
    plt.fill_between(wl_grid, lower_spec_5, upper_spec_95, color='black', alpha=0.05, label="90% uncertainty")
    
    plt.errorbar(
        filters_wl,
        ref_norm,
        yerr=ref_err_norm,
        fmt='o',
        color='red',
        ecolor='red',
        markersize=8,
        capsize=5,
        linewidth=2,
        zorder=10,
        label="Input Photometry"
    )
    
    plt.xlabel('Wavelength ($\mu$m)', fontsize=14, fontweight='bold')
    plt.ylabel(f'Normalized Reflectance (at {norm_wl} $\mu$m)', fontsize=13, fontweight='bold')
    plt.title(f"Reconstructed Reflectance Spectrum: {args.name}", fontsize=14, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend(fontsize=10, frameon=True)
    plt.xlim(0.35, 5.15)
    plt.tight_layout()
    
    plot_path_png = os.path.join(out_dir, f"{safe_name}_reconstructed_spectrum.png")
    plt.savefig(plot_path_png, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to: {plot_path_png}")

    # --- 2D Latent Space Plot (PC1 vs PC2) ---
    try:
        plt.figure(figsize=(7, 6))
        
        # 1. Background prior distribution (TNO reference set)
        prior_pc1, prior_pc2 = None, None
        base_pca_path = os.path.join(os.path.dirname(model_dir), "base_pca_kde.pkl")
        if os.path.exists(base_pca_path):
            with open(base_pca_path, "rb") as f:
                base_data = pickle.load(f)
            if 'latent_data' in base_data and base_data['latent_data'] is not None:
                prior_pc1 = base_data['latent_data'][:, 0]
                prior_pc2 = base_data['latent_data'][:, 1]
                
        if prior_pc1 is not None:
            plt.scatter(
                prior_pc1, prior_pc2,
                color='gray', alpha=0.4, s=35,
                edgecolors='none', label="TNO Population Prior"
            )
            
            # Contour density for population prior
            try:
                from scipy.stats import gaussian_kde
                margin_x = (prior_pc1.max() - prior_pc1.min()) * 0.15
                margin_y = (prior_pc2.max() - prior_pc2.min()) * 0.15
                x_grid = np.linspace(prior_pc1.min() - margin_x, prior_pc1.max() + margin_x, 100)
                y_grid = np.linspace(prior_pc2.min() - margin_y, prior_pc2.max() + margin_y, 100)
                XX, YY = np.meshgrid(x_grid, y_grid)
                kernel = gaussian_kde(np.vstack([prior_pc1, prior_pc2]))
                ZZ = kernel(np.vstack([XX.ravel(), YY.ravel()])).reshape(XX.shape)
                plt.contour(XX, YY, ZZ, levels=4, colors='gray', alpha=0.35, linestyles='--')
            except Exception:
                pass

        # 2. Target Object Posterior Contours (PC1 vs PC2)
        target_pc1 = group_latent[:, 0]
        target_pc2 = group_latent[:, 1]
        
        try:
            from scipy.stats import gaussian_kde
            from matplotlib.lines import Line2D
            
            target_positions = np.vstack([target_pc1, target_pc2])
            target_kde = gaussian_kde(target_positions)
            
            # Grid for target posterior contours
            pad_x = (target_pc1.max() - target_pc1.min()) * 0.3
            pad_y = (target_pc2.max() - target_pc2.min()) * 0.3
            x_t = np.linspace(target_pc1.min() - pad_x, target_pc1.max() + pad_x, 100)
            y_t = np.linspace(target_pc2.min() - pad_y, target_pc2.max() + pad_y, 100)
            XX_t, YY_t = np.meshgrid(x_t, y_t)
            ZZ_t = target_kde(np.vstack([XX_t.ravel(), YY_t.ravel()])).reshape(XX_t.shape)
            
            # Compute multi-level credibility thresholds (95%, 80%, 68%, 50%, 30%)
            sample_densities = target_kde(target_positions)
            sorted_densities = np.sort(sample_densities)
            pcts = [0.05, 0.20, 0.32, 0.50, 0.70]  # 95%, 80%, 68%, 50%, 30% credibility
            levels = [sorted_densities[int(len(sorted_densities) * p)] for p in pcts] + [ZZ_t.max() * 1.05]
            
            # Continuous gradient fill using Reds colormap
            plt.contourf(
                XX_t, YY_t, ZZ_t,
                levels=levels,
                cmap='Reds',
                alpha=0.45,
                zorder=5
            )
            
            # Fine line contours
            plt.contour(
                XX_t, YY_t, ZZ_t,
                levels=levels[:-1],
                colors=['#c0392b', '#a93226', '#922b21', '#7b241c', '#641e16'],
                linewidths=[0.8, 1.0, 1.3, 1.6, 2.0],
                zorder=6
            )
            
            plt.scatter([], [], color='#c0392b', alpha=0.6, label=f"{args.name} Multi-level Posterior CI")
        except Exception:
            plt.scatter(
                target_pc1, target_pc2,
                color='tab:red', alpha=0.3, s=15,
                edgecolors='none', label=f"{args.name} Posterior Samples"
            )
        
        # Posterior Mean Marker
        mean_pc1 = np.mean(target_pc1)
        mean_pc2 = np.mean(target_pc2)
        plt.scatter(
            [mean_pc1], [mean_pc2],
            color='gold', marker='*', s=220,
            edgecolors='black', linewidths=1.2,
            zorder=10, label=f"{args.name} Posterior Mean"
        )

        # Labels & Explained Variance
        pc1_var = pca.explained_variance_ratio_[0] * 100 if hasattr(pca, 'explained_variance_ratio_') else 0
        pc2_var = pca.explained_variance_ratio_[1] * 100 if hasattr(pca, 'explained_variance_ratio_') else 0
        
        var1_str = f" ({pc1_var:.1f}% var)" if pc1_var > 0 else ""
        var2_str = f" ({pc2_var:.1f}% var)" if pc2_var > 0 else ""
        
        plt.xlabel(f'PC1 Coordinate{var1_str}', fontsize=12, fontweight='bold')
        plt.ylabel(f'PC2 Coordinate{var2_str}', fontsize=12, fontweight='bold')
        plt.title(f"Latent PC1-PC2 Posterior Distribution: {args.name}", fontsize=13, fontweight='bold')
        plt.grid(True, linestyle=':', alpha=0.5)
        plt.legend(fontsize=10, loc='best', frameon=True)
        plt.tight_layout()
        
        latent_plot_path_png = os.path.join(out_dir, f"{safe_name}_latent_pc1_pc2.png")
        plt.savefig(latent_plot_path_png, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved latent PC1-PC2 plot to: {latent_plot_path_png}")
    except Exception as e:
        print(f"Warning: Could not generate 2D latent space plot: {e}")

if __name__ == "__main__":
    main()

