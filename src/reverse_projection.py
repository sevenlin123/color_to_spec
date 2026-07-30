import os
import sys
import pickle
import yaml
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import gaussian_kde
from scipy.interpolate import interp1d

# Resolve paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(script_dir)

from reconstruct_spectra import SYSTEMS, calculate_photometry
from metrics import get_object_class, RegularizedKDE

def main():
    # 1. Setup paths
    models_dir = os.path.join(project_root, "models")
    plots_dir = os.path.join(project_root, "plots")
    artifact_dir = "/Users/hsingwel/.gemini/antigravity/brain/52536918-f1e0-4a4e-a7fb-446c1235ff36"
    
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(artifact_dir, exist_ok=True)
    
    base_path = os.path.join(models_dir, "base_pca_kde_0926.pkl")
    if not os.path.exists(base_path):
        base_path = os.path.join(models_dir, "base_pca_kde.pkl")
        
    # 2. Load the base model
    print(f"Loading DES dedicated PCA/KDE model from {base_path}...")
    with open(base_path, "rb") as f:
        base_data = pickle.load(f)
        
    pca = base_data['pca']
    wl_grid = base_data['wl_grid']
    kde_classes = base_data['kde_classes']
    latent_data = base_data['latent_data']
    n_components = pca.n_components_
    norm_wl = base_data.get('norm_wl', 0.926)
    
    print(f"PCA components: {n_components}")
    print(f"wl_grid range: {wl_grid.min():.2f} to {wl_grid.max():.2f} microns ({len(wl_grid)} points)")
    
    # 3. Load true spectra of the 48 training objects directly from data/processed
    from reconstruct_spectra import load_and_preprocess_spectra
    processed_dir = os.path.join(project_root, "data", "processed")
    training_true_spectra, file_names, wl_grid_proc = load_and_preprocess_spectra(processed_dir, norm_wl=norm_wl)
    
    training_names = [fn.replace("_merged_unified.csv", "").replace(".csv", "") for fn in file_names]
    training_classes = [get_object_class(fn) for fn in file_names]
    latent_data = pca.transform(training_true_spectra)
    
    print(f"Successfully loaded {len(training_true_spectra)} true spectra from {processed_dir}.")
    
    # 4. Generate decoded training spectra from latent_data as a consistency check
    training_decoded_spectra = pca.inverse_transform(latent_data)
    training_decoded_spectra = np.clip(training_decoded_spectra, 0.0, None)
    
    # Re-normalize at norm_wl to ensure exact anchoring
    idx_norm = np.argmin(np.abs(wl_grid - norm_wl))
    training_decoded_spectra = training_decoded_spectra / training_decoded_spectra[:, idx_norm].reshape(-1, 1)
    
    # 5. Draw 1000 samples per class from class-conditioned latent densities (5D)
    np.random.seed(42)
    n_samples_per_class = 1000
    classes = ['Water', 'CO2', 'Organic', 'Methanol']
    
    # Fit class-conditioned KDEs on actual training latent coordinates rather than base model fallbacks
    class_vectors = {cls: [] for cls in classes}
    for idx, cls in enumerate(training_classes):
        if cls in class_vectors:
            class_vectors[cls].append(latent_data[idx])
            
    fitted_kde_classes = {}
    for cls in classes:
        vectors = np.array(class_vectors[cls])
        if len(vectors) > 0:
            v_sub = vectors[:, :5]
            fitted_kde_classes[cls] = RegularizedKDE(v_sub.T, reg=0.05)
        else:
            print(f"Warning: No training objects found for class {cls}!")
            
    sampled_latent_list = []
    sampled_classes_list = []
    
    for cls in classes:
        kde = fitted_kde_classes[cls]
        # Draw 5D samples
        samples_5d = kde.resample(size=n_samples_per_class).T
        
        # Pad to 10D with zeros
        samples_10d = np.zeros((n_samples_per_class, n_components))
        samples_10d[:, :5] = samples_5d
        
        sampled_latent_list.append(samples_10d)
        sampled_classes_list.extend([cls] * n_samples_per_class)
        
    sampled_latent = np.vstack(sampled_latent_list)
    
    # Decode sampled latent vectors to spectra
    sampled_spectra = pca.inverse_transform(sampled_latent)
    sampled_spectra = np.clip(sampled_spectra, 0.0, None)
    # Re-normalize at norm_wl
    sampled_spectra = sampled_spectra / sampled_spectra[:, idx_norm].reshape(-1, 1)
    
    # 6. Compute synthetic colors for all categories
    # Solar colors
    gr_sun = 0.45
    rz_sun = 0.16
    ri_sun = 0.12
    iz_sun = 0.04
    
    def get_synthetic_colors(spectra_matrix, system_name):
        sys_info = SYSTEMS[system_name]
        bands = sys_info['bands']
        wavelengths = sys_info['wavelengths']
        widths = sys_info['widths']
        
        res_list = []
        for i in range(len(spectra_matrix)):
            spec = spectra_matrix[i]
            refs = calculate_photometry(spec, wavelengths, widths, wl_grid, add_noise=False)
            ref_dict = dict(zip(bands, refs))
            
            # Convert reflectances to colors
            c_gr = gr_sun - 2.5 * np.log10(ref_dict['g'] / ref_dict['r'])
            c_rz = rz_sun - 2.5 * np.log10(ref_dict['r'] / ref_dict['z'])
            c_ri = ri_sun - 2.5 * np.log10(ref_dict['r'] / ref_dict['i'])
            c_iz = iz_sun - 2.5 * np.log10(ref_dict['i'] / ref_dict['z'])
            
            res_list.append({
                f'{system_name}_g_ref': ref_dict['g'],
                f'{system_name}_r_ref': ref_dict['r'],
                f'{system_name}_i_ref': ref_dict['i'],
                f'{system_name}_z_ref': ref_dict['z'],
                f'{system_name}_g-r': c_gr,
                f'{system_name}_r-z': c_rz,
                f'{system_name}_r-i': c_ri,
                f'{system_name}_i-z': c_iz
            })
        return pd.DataFrame(res_list)
        
    print("Computing synthetic photometry and colors for DES and LSST...")
    # Training true spectra colors
    train_true_des = get_synthetic_colors(training_true_spectra, 'DES')
    train_true_lsst = get_synthetic_colors(training_true_spectra, 'LSST')
    
    # Training decoded spectra colors
    train_dec_des = get_synthetic_colors(training_decoded_spectra, 'DES')
    train_dec_lsst = get_synthetic_colors(training_decoded_spectra, 'LSST')
    
    # Sampled spectra colors
    sampled_des = get_synthetic_colors(sampled_spectra, 'DES')
    sampled_lsst = get_synthetic_colors(sampled_spectra, 'LSST')
    
    # 7. Validation sanity checks
    # Print out comparison of true vs decoded training colors
    print("\n--- Validation Sanity Checks ---")
    diff_gr = np.abs(train_true_des['DES_g-r'].values - train_dec_des['DES_g-r'].values)
    diff_rz = np.abs(train_true_des['DES_r-z'].values - train_dec_des['DES_r-z'].values)
    print(f"DES g-r difference (True vs Decoded PCA): Mean={diff_gr.mean():.4f}, Max={diff_gr.max():.4f}")
    print(f"DES r-z difference (True vs Decoded PCA): Mean={diff_rz.mean():.4f}, Max={diff_rz.max():.4f}")
    
    # Check for unphysical (extreme or NaN) values
    nans_sampled_gr = np.isnan(sampled_des['DES_g-r'].values).sum()
    nans_sampled_rz = np.isnan(sampled_des['DES_r-z'].values).sum()
    print(f"NaN values in sampled DES colors: g-r: {nans_sampled_gr}, r-z: {nans_sampled_rz}")
    
    # 8. Create and save outputs
    # A. CSV Catalog
    samples_df = pd.DataFrame()
    samples_df['Sample_ID'] = np.arange(len(sampled_latent))
    samples_df['Class_Label'] = sampled_classes_list
    
    for i in range(5):
        samples_df[f'PC{i+1}'] = sampled_latent[:, i]
        
    for sys in ['DES', 'LSST']:
        sys_df = get_synthetic_colors(sampled_spectra, sys)
        for col in sys_df.columns:
            samples_df[col] = sys_df[col].values
            
    csv_out_path = os.path.join(plots_dir, "reverse_projection_samples.csv")
    samples_df.to_csv(csv_out_path, index=False)
    print(f"Saved synthetic samples to: {csv_out_path}")
    
    # B. Config YAML
    config_data = {
        'number_of_samples_per_class': n_samples_per_class,
        'filter_system': 'DES and LSST',
        'solar_colors': {
            'g-r': gr_sun,
            'r-z': rz_sun,
            'r-i': ri_sun,
            'i-z': iz_sun
        },
        'filter_integration_method': 'Planck Solar Function Band Integration (reconstruct_spectra.calculate_photometry)',
        'KDE_bandwidth_settings': 'RegularizedKDE (scipy.stats.gaussian_kde subclass with diagonal regularization 0.05)',
        'PCA_components_used': 10,
        'random_seed': 42
    }
    
    yaml_out_path = os.path.join(plots_dir, "reverse_projection_config.yaml")
    with open(yaml_out_path, 'w') as fy:
        yaml.dump(config_data, fy, default_flow_style=False)
    print(f"Saved configuration metadata to: {yaml_out_path}")
    
    # C. Plotting
    # Class colors mapping
    class_styles = {
        'Water': {'color': '#3498db', 'label': 'H$_2$O', 'marker': 'o'},
        'CO2': {'color': '#f39c12', 'label': 'CO$_2$', 'marker': 's'},
        'Organic': {'color': '#e74c3c', 'label': 'Organic', 'marker': '^'},
        'Methanol': {'color': '#9c27b0', 'label': 'Methanol', 'marker': 'D'}
    }
    
    # 2D GMM marginal density for background contours
    from scipy.stats import multivariate_normal
    mu_B = np.array([0.609, 0.355])
    cov_B = np.array([
        [0.00434, 0.00547],
        [0.00547, 0.00727]
    ])
    mu_F = np.array([0.902, 0.524])
    cov_F = np.array([
        [0.00674, 0.00557],
        [0.00557, 0.00533]
    ])
    w_B = 0.446
    w_F = 0.554

    x_grid = np.linspace(0.2, 1.4, 100)
    y_grid = np.linspace(-0.15, 1.05, 100)
    X, Y = np.meshgrid(x_grid, y_grid)
    pos = np.dstack((X, Y))
    pdf_GMM = w_B * multivariate_normal(mu_B, cov_B).pdf(pos) + w_F * multivariate_normal(mu_F, cov_F).pdf(pos)
    
    # Define explicit levels starting from 1% of peak density to show the wide tails (down to 2-3sigma)
    gmm_levels = pdf_GMM.max() * np.array([0.01, 0.05, 0.1, 0.3, 0.6, 0.9])
    
    # Set up figure for two panels
    plt.figure(figsize=(14.5, 6.5))
    
    # ================= PANEL A: DES Color-Color Space =================
    plt.subplot(1, 2, 1)
    
    # Plot GMM background contours
    plt.contour(X, Y, pdf_GMM, levels=gmm_levels, colors='#555555', alpha=0.6, linewidths=1.0, linestyles='--')
    plt.text(mu_B[0] - 0.05, mu_B[1] - 0.05, 'NIRB', color='#333333', fontsize=15, fontweight='bold', alpha=0.9)
    plt.text(mu_F[0] - 0.05, mu_F[1] - 0.05, 'NIRF', color='#333333', fontsize=15, fontweight='bold', alpha=0.9)
    
    # 1. Plot KDE density maps (outline contours for each class, plotted in order with CO2 last)
    pos_flat = np.vstack([X.ravel(), Y.ravel()])
    plot_order = ['Water', 'Methanol', 'Organic', 'CO2']
    for cls in plot_order:
        cls_sampled_mask = np.array(sampled_classes_list) == cls
        x_cls = sampled_des.loc[cls_sampled_mask, 'DES_g-r'].values
        y_cls = sampled_des.loc[cls_sampled_mask, 'DES_r-z'].values
        
        # Fit 2D KDE
        values = np.vstack([x_cls, y_cls])
        kernel = gaussian_kde(values, bw_method=0.35)
        Z_cls = np.reshape(kernel(pos_flat).T, X.shape)
        
        # Define smooth gradient levels starting at 2% density floor to cover wide tails
        levels_cls = np.linspace(0.02 * Z_cls.max(), 1.00 * Z_cls.max(), 8)
        
        from matplotlib.colors import to_rgba, LinearSegmentedColormap
        g_color = class_styles[cls]['color']
        max_alpha = 0.65 if cls == 'CO2' else 0.45
        rgba_color = to_rgba(g_color, alpha=max_alpha)
        rgba_transparent = (rgba_color[0], rgba_color[1], rgba_color[2], 0.0)
        cmap_cls = LinearSegmentedColormap.from_list(f'cmap_{cls}', [rgba_transparent, rgba_color])
        
        plt.contourf(
            X, Y, Z_cls,
            levels=levels_cls,
            cmap=cmap_cls,
            zorder=2
        )
        
    # 2. Overlay training set objects, colored by class
    for cls in plot_order:
        cls_mask = np.array(training_classes) == cls
        if np.any(cls_mask):
            marker_size = 35 if cls != 'CO2' else 25
            plt.scatter(
                train_true_des.loc[cls_mask, 'DES_g-r'],
                train_true_des.loc[cls_mask, 'DES_r-z'],
                color=class_styles[cls]['color'],
                marker=class_styles[cls]['marker'],
                s=marker_size,
                edgecolor='black',
                linewidths=0.6,
                label=class_styles[cls]['label'],
                zorder=5
            )
            
    # Load and process all 7 special target spectra (SO277 family & RJ103 family)
    special_targets = [
        {'file': '2011SO277_merged_unified.csv', 'label': '2011SO277', 'marker': 'P', 'size': 160, 'color': '#111111'},
        {'file': 'spec_2016BP81_primary_extended.csv', 'label': '2016BP81 (Pri)', 'marker': '*', 'size': 210, 'color': '#111111', 'fixed_gr': 0.66, 'fixed_rz': 0.43},
        {'file': 'spec_2016BP81_secondary_extended.csv', 'label': '2016BP81 (Sec)', 'marker': 'v', 'size': 150, 'color': '#111111', 'fixed_gr': 0.66, 'fixed_rz': 0.43},
        {'file': 'spec_2016QV89_extended.csv', 'label': '2016QV89', 'marker': 'h', 'size': 160, 'color': '#111111'},
        {'file': '2006RJ103_merged_unified.csv', 'label': '2006RJ103', 'marker': 'X', 'size': 160, 'color': '#111111'},
        {'file': 'nt_spectrum/2004EW95.csv', 'label': '2004EW95', 'marker': '^', 'size': 150, 'color': '#111111'},
        {'file': '32.csv', 'label': '1998SN165', 'marker': 'D', 'size': 140, 'color': '#111111'},
    ]
    
    sys_info_des = SYSTEMS['DES']
    bands_des, wl_des, w_des = sys_info_des['bands'], sys_info_des['wavelengths'], sys_info_des['widths']
    idx_norm_sp = np.argmin(np.abs(wl_grid - norm_wl))
    
    special_coords = []
    for item in special_targets:
        f_path = os.path.join(processed_dir, item['file'])
        if not os.path.exists(f_path):
            f_path = os.path.join(project_root, 'data', 'raw', item['file'])
            
        if os.path.exists(f_path):
            df_sp = pd.read_csv(f_path)
            w_col = 'um' if 'um' in df_sp.columns else 'wavelength_um'
            w_sp = df_sp[w_col].values
            r_sp = df_sp['reflectance'].values
            
            if w_sp[0] > 0.45:
                m_s, c_s = np.polyfit(w_sp[:20], r_sp[:20], 1)
                w_opt = np.arange(0.4, w_sp[0], 0.005)
                r_opt = m_s * w_opt + c_s
                w_sp = np.concatenate([w_opt, w_sp])
                r_sp = np.concatenate([r_opt, r_sp])
                
            r_grid_sp = np.interp(wl_grid, w_sp, r_sp)
            r_norm_sp = r_grid_sp[idx_norm_sp]
            spec_norm_sp = r_grid_sp / r_norm_sp if r_norm_sp > 0 else r_grid_sp
            
            lat_sp = pca.transform(spec_norm_sp.reshape(1, -1))[0]
            refs_sp = calculate_photometry(spec_norm_sp, wl_des, w_des, wl_grid, add_noise=False)
            ref_dict_sp = dict(zip(bands_des, refs_sp))
            
            c_gr = item.get('fixed_gr', gr_sun - 2.5 * np.log10(ref_dict_sp['g'] / ref_dict_sp['r']))
            c_rz = item.get('fixed_rz', rz_sun - 2.5 * np.log10(ref_dict_sp['r'] / ref_dict_sp['z']))
            
            special_coords.append({
                'label': item['label'],
                'marker': item['marker'],
                'size': item['size'],
                'color': item['color'],
                'gr': c_gr,
                'rz': c_rz,
                'pc1': lat_sp[0],
                'pc2': lat_sp[1]
            })

    # 3. Highlight specific special targets (SO277, RJ103, BP81, QV89) in Panel A
    for item in special_coords:
        plt.scatter(
            item['gr'], item['rz'],
            s=item['size'], color=item['color'],
            edgecolor='white', linewidths=0.9,
            marker=item['marker'], zorder=7,
            label=item['label']
        )
            
    plt.xlabel('$g - r$', fontsize=20, fontweight='bold')
    plt.ylabel('$r - z$', fontsize=20, fontweight='bold')
    plt.xlim(0.25, 1.35)
    plt.ylim(-0.1, 1.0)
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Separate Legends for Panel A
    import matplotlib.lines as mlines
    class_handles = [
        mlines.Line2D([], [], color=class_styles[cls]['color'], marker=class_styles[cls]['marker'], linestyle='None', markersize=9, label=class_styles[cls]['label'])
        for cls in ['Water', 'CO2', 'Organic', 'Methanol']
    ]
    outlier_handles = [
        mlines.Line2D([], [], color=item['color'], marker=item['marker'], linestyle='None', markersize=11, label=item['label'])
        for item in special_coords
    ]
    
    ax1 = plt.gca()
    leg1 = ax1.legend(handles=class_handles, frameon=True, fontsize=13, loc='upper left')
    ax1.add_artist(leg1)
    ax1.legend(handles=outlier_handles, frameon=True, fontsize=11, loc='lower right')
    
    # ================= PANEL B: PCA Latent Space (PC1 vs PC2) =================
    plt.subplot(1, 2, 2)
    
    # Setup PC1 and PC2 grid limits
    sp_pc1 = [item['pc1'] for item in special_coords]
    sp_pc2 = [item['pc2'] for item in special_coords]
    all_pc1 = np.concatenate([sampled_latent[:, 0], latent_data[:, 0], sp_pc1])
    all_pc2 = np.concatenate([sampled_latent[:, 1], latent_data[:, 1], sp_pc2])
    pc1_min, pc1_max = all_pc1.min(), all_pc1.max()
    pc2_min, pc2_max = all_pc2.min(), all_pc2.max()
    pc1_pad = 0.08 * (pc1_max - pc1_min)
    pc2_pad = 0.08 * (pc2_max - pc2_min)
    pc1_lim = (pc1_min - pc1_pad, pc1_max + pc1_pad)
    pc2_lim = (pc2_min - pc2_pad, pc2_max + pc2_pad)
    
    pc1_grid = np.linspace(pc1_lim[0], pc1_lim[1], 100)
    pc2_grid = np.linspace(pc2_lim[0], pc2_lim[1], 100)
    X_pc, Y_pc = np.meshgrid(pc1_grid, pc2_grid)
    pos_pc_flat = np.vstack([X_pc.ravel(), Y_pc.ravel()])
    
    # 1. Plot KDE density maps for PC1 vs PC2
    for cls in plot_order:
        cls_sampled_mask = np.array(sampled_classes_list) == cls
        x_cls_pc = sampled_latent[cls_sampled_mask, 0]
        y_cls_pc = sampled_latent[cls_sampled_mask, 1]
        
        values_pc = np.vstack([x_cls_pc, y_cls_pc])
        kernel_pc = gaussian_kde(values_pc, bw_method=0.35)
        Z_cls_pc = np.reshape(kernel_pc(pos_pc_flat).T, X_pc.shape)
        
        levels_cls_pc = np.linspace(0.02 * Z_cls_pc.max(), 1.00 * Z_cls_pc.max(), 8)
        
        g_color = class_styles[cls]['color']
        max_alpha = 0.65 if cls == 'CO2' else 0.45
        rgba_color = to_rgba(g_color, alpha=max_alpha)
        rgba_transparent = (rgba_color[0], rgba_color[1], rgba_color[2], 0.0)
        cmap_cls_pc = LinearSegmentedColormap.from_list(f'cmap_pc_{cls}', [rgba_transparent, rgba_color])
        
        plt.contourf(
            X_pc, Y_pc, Z_cls_pc,
            levels=levels_cls_pc,
            cmap=cmap_cls_pc,
            zorder=2
        )
        
    # 2. Overlay training set objects in PC1 vs PC2 space (s=35, CO2 s=22)
    for cls in plot_order:
        cls_mask = np.array(training_classes) == cls
        if np.any(cls_mask):
            x_train_pc = latent_data[cls_mask, 0]
            y_train_pc = latent_data[cls_mask, 1]
            
            marker_size = 22 if cls == 'CO2' else 35
            plt.scatter(
                x_train_pc,
                y_train_pc,
                color=class_styles[cls]['color'],
                marker=class_styles[cls]['marker'],
                s=marker_size,
                edgecolor='black',
                linewidths=0.6,
                label=class_styles[cls]['label'],
                zorder=5
            )
            
    # 3. Highlight specific special targets (SO277, RJ103, BP81, QV89) in Panel B
    for item in special_coords:
        plt.scatter(
            item['pc1'], item['pc2'],
            s=item['size'], color=item['color'],
            edgecolor='white', linewidths=0.9,
            marker=item['marker'], zorder=7,
            label=item['label']
        )
            
    plt.xlabel('PC 1', fontsize=20, fontweight='bold')
    plt.ylabel('PC 2', fontsize=20, fontweight='bold')
    plt.xlim(pc1_lim)
    plt.ylim(pc2_lim)
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    plot_out_path_png = os.path.join(plots_dir, "reverse_projection_color_color.png")
    plot_out_path_pdf = os.path.join(plots_dir, "reverse_projection_color_color.pdf")
    
    plt.savefig(plot_out_path_png, dpi=250)
    plt.savefig(plot_out_path_pdf, bbox_inches='tight')
    plt.close()
    print(f"Saved plots to:\n  - {plot_out_path_png}\n  - {plot_out_path_pdf}")
    
    # Copy all files to artifact directory
    os.system(f"cp {csv_out_path} {artifact_dir}/")
    os.system(f"cp {yaml_out_path} {artifact_dir}/")
    os.system(f"cp {plot_out_path_png} {artifact_dir}/")
    os.system(f"cp {plot_out_path_pdf} {artifact_dir}/")
    print(f"Copied all outputs to artifact directory:\n  {artifact_dir}")

if __name__ == '__main__':
    main()
