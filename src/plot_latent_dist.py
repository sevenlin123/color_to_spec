import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from matplotlib.colors import to_rgb, LinearSegmentedColormap

# Resolve paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
plots_dir = os.path.join(project_root, "plots")
artifact_dir = "/Users/hsingwel/.gemini/antigravity/brain/52536918-f1e0-4a4e-a7fb-446c1235ff36"

sys.path.append(script_dir)
from metrics import get_object_class

def main():
    # 1. Load the generated samples
    csv_path = os.path.join(plots_dir, "reverse_projection_samples.csv")
    if not os.path.exists(csv_path):
        print(f"Error: Samples CSV not found at {csv_path}. Please run reverse_projection.py first.")
        sys.exit(1)
        
    print(f"Loading samples from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # 2. Compute latent distance
    df['r_latent'] = np.sqrt(df['PC1']**2 + df['PC2']**2)
    
    # Global normalization of latent distance for color shading
    r_min = df['r_latent'].min()
    r_max = df['r_latent'].max()
    df['norm_r'] = (df['r_latent'] - r_min) / (r_max - r_min)
    
    # Class colors mapping
    class_styles = {
        'Water': {'color': '#3498db', 'label': 'H$_2$O', 'marker': 'o'},
        'CO2': {'color': '#f39c12', 'label': 'CO$_2$', 'marker': 's'},
        'Organic': {'color': '#e74c3c', 'label': 'Organic', 'marker': '^'},
        'Methanol': {'color': '#9c27b0', 'label': 'Methanol', 'marker': 'D'}
    }
    
    # Function to create a custom colormap with high contrast alpha/shade gradient
    def make_alpha_cmap(color_hex):
        rgb = to_rgb(color_hex)
        # Start: 50% white blend (light/pale) with moderate alpha (0.40)
        c_start = (rgb[0] + (1 - rgb[0])*0.5, rgb[1] + (1 - rgb[1])*0.5, rgb[2] + (1 - rgb[2])*0.5, 0.40)
        # Middle: original bright class color with high alpha (0.85)
        c_mid = (rgb[0], rgb[1], rgb[2], 0.85)
        # End: 50% black blend (dark/deep) with solid alpha (0.95)
        c_end = (rgb[0]*0.45, rgb[1]*0.45, rgb[2]*0.45, 0.95)
        return LinearSegmentedColormap.from_list(f'cmap_{color_hex}', [c_start, c_mid, c_end])
    
    # Setup GMM background contours (for Panel A)
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
    gmm_levels = pdf_GMM.max() * np.array([0.01, 0.05, 0.1, 0.3, 0.6, 0.9])
    
    # Setup PC1 and PC2 limits (for Panel B)
    pc1_min, pc1_max = df['PC1'].min(), df['PC1'].max()
    pc2_min, pc2_max = df['PC2'].min(), df['PC2'].max()
    pc1_pad = 0.1 * (pc1_max - pc1_min)
    pc2_pad = 0.1 * (pc2_max - pc2_min)
    pc1_lim = (pc1_min - pc1_pad, pc1_max + pc1_pad)
    pc2_lim = (pc2_min - pc2_pad, pc2_max + pc2_pad)
    
    # Load training base data and special targets for outlier plotting
    base_path = os.path.join(project_root, "models", "base_pca_kde_0926.pkl")
    if not os.path.exists(base_path):
        base_path = os.path.join(project_root, "models", "base_pca_kde.pkl")
        
    has_base = False
    special_coords = []
    
    if os.path.exists(base_path):
        with open(base_path, "rb") as f:
            base_data = pickle.load(f)
        pca = base_data['pca']
        wl_grid = base_data['wl_grid']
        norm_wl = base_data.get('norm_wl', 0.926)
        
        from reconstruct_spectra import SYSTEMS, calculate_photometry
        processed_dir = os.path.join(project_root, "data", "processed")
        
        special_targets = [
            {'file': '2011SO277_merged_unified.csv', 'label': '2011SO277', 'marker': 'P', 'size': 160, 'color': '#111111'},
            {'file': 'spec_2016BP81_primary_extended.csv', 'label': '2016BP81 (Pri)', 'marker': '*', 'size': 210, 'color': '#111111', 'fixed_gr': 0.66, 'fixed_rz': 0.43},
            {'file': 'spec_2016BP81_secondary_extended.csv', 'label': '2016BP81 (Sec)', 'marker': 'v', 'size': 150, 'color': '#111111', 'fixed_gr': 0.66, 'fixed_rz': 0.43},
            {'file': 'spec_2016QV89_extended.csv', 'label': '2016QV89', 'marker': 'h', 'size': 160, 'color': '#111111'},
            {'file': '2006RJ103_merged_unified.csv', 'label': '2006RJ103', 'marker': 'X', 'size': 160, 'color': '#111111'},
            {'file': 'nt_spectrum/2004EW95.csv', 'label': '2004EW95', 'marker': '^', 'size': 150, 'color': '#111111'},
            {'file': '32.csv', 'label': '1998SN165', 'marker': 'D', 'size': 140, 'color': '#111111'},
        ]
        
        gr_sun, rz_sun = 0.45, 0.16
        sys_info = SYSTEMS['DES']
        bands, wavelengths, widths = sys_info['bands'], sys_info['wavelengths'], sys_info['widths']
        idx_norm_sp = np.argmin(np.abs(wl_grid - norm_wl))
        
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
                refs_sp = calculate_photometry(spec_norm_sp, wavelengths, widths, wl_grid, add_noise=False)
                ref_dict_sp = dict(zip(bands, refs_sp))
                
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
        has_base = True

    # Setup PC1 and PC2 limits
    sp_pc1 = [item['pc1'] for item in special_coords]
    sp_pc2 = [item['pc2'] for item in special_coords]
    all_pc1 = np.concatenate([df['PC1'].values, sp_pc1]) if len(sp_pc1) > 0 else df['PC1'].values
    all_pc2 = np.concatenate([df['PC2'].values, sp_pc2]) if len(sp_pc2) > 0 else df['PC2'].values
    
    pc1_min, pc1_max = all_pc1.min(), all_pc1.max()
    pc2_min, pc2_max = all_pc2.min(), all_pc2.max()
    pc1_pad = 0.08 * (pc1_max - pc1_min)
    pc2_pad = 0.08 * (pc2_max - pc2_min)
    pc1_lim = (pc1_min - pc1_pad, pc1_max + pc1_pad)
    pc2_lim = (pc2_min - pc2_pad, pc2_max + pc2_pad)

    # Plotting
    plt.figure(figsize=(14.5, 6.5))
    classes = ['Water', 'Methanol', 'Organic', 'CO2']  # CO2 plotted last on top
    
    # ================= PANEL A: DES Color-Color Space =================
    plt.subplot(1, 2, 1)

    # Plot sampled points colored by class and shaded by class-specific PC axis
    for cls in classes:
        cls_mask = df['Class_Label'] == cls
        df_cls = df[cls_mask]
        
        val = df_cls['PC2']
        v_min, v_max = val.min(), val.max()
        norm_val = (val - v_min) / (v_max - v_min) if v_max > v_min else np.zeros_like(val)
        
        cmap = make_alpha_cmap(class_styles[cls]['color'])
        
        plt.scatter(
            df_cls['DES_g-r'],
            df_cls['DES_r-z'],
            c=norm_val,
            cmap=cmap,
            marker=class_styles[cls]['marker'],
            s=12,
            edgecolors='none',
            zorder=3
        )
        
    # Overlay Outliers in Panel A
    if has_base:
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
    
    # Plot sampled points colored by class and shaded by class-specific PC axis in PC space
    for cls in classes:
        cls_mask = df['Class_Label'] == cls
        df_cls = df[cls_mask]
        
        val = df_cls['PC2']
        v_min, v_max = val.min(), val.max()
        norm_val = (val - v_min) / (v_max - v_min) if v_max > v_min else np.zeros_like(val)
        
        cmap = make_alpha_cmap(class_styles[cls]['color'])
        
        plt.scatter(
            df_cls['PC1'],
            df_cls['PC2'],
            c=norm_val,
            cmap=cmap,
            marker=class_styles[cls]['marker'],
            s=12,
            edgecolors='none',
            zorder=3
        )
        
    # Overlay Outliers in Panel B
    if has_base:
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
    
    # F. Save and copy
    plot_png = os.path.join(plots_dir, "reverse_projection_latent_dist.png")
    plot_pdf = os.path.join(plots_dir, "reverse_projection_latent_dist.pdf")
    
    plt.savefig(plot_png, dpi=250)
    plt.savefig(plot_pdf, bbox_inches='tight')
    plt.close()
    
    os.system(f"cp {plot_png} {artifact_dir}/")
    os.system(f"cp {plot_pdf} {artifact_dir}/")
    print(f"Saved new distance shaded plot to:\n  - {plot_png}\n  - {plot_pdf}")

if __name__ == '__main__':
    main()
