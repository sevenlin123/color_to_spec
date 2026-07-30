import os
import sys
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import KernelDensity
from sklearn.metrics import mean_squared_error
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import glob
import numpy as np

class PCASpectrumGenerator:
    def __init__(self, real_spectra, labels, n_components=5):
        """
        real_spectra: Array of shape (36, 100) - Your real TNO spectra
        n_components: Number of PCA components to keep
        """
        self.pca = PCA(n_components=n_components)
        
        # Compress the real spectra into the "Latent Space"
        self.latent_data = self.pca.fit_transform(real_spectra)
        print(f"PCA Variance Explained: {np.sum(self.pca.explained_variance_ratio_):.4f}")
        
        # KDE probability map
        self.kde = KernelDensity(kernel='gaussian', bandwidth=0.5)
        self.kde.fit(self.latent_data)
        
        self.labels = labels
        # Colors matching user's paper:
        # 0: Methanol (purple), 1: Organic (red), 2: CO2 (orange), 3: H2O (blue)
        self.colors = []
        for label in labels:
            if label == 0:
                self.colors.append('#9c27b0')  # methanol (purple)
            elif label == 1:
                self.colors.append('#e74c3c')  # organic (red)
            elif label == 2:
                self.colors.append('#f39c12')  # CO2 (orange)
            else:
                self.colors.append('#3498db')  # H2O (blue)
                
    def generate(self, n_samples=1000):
        new_latent_vectors = self.kde.sample(n_samples)
        mock_spectra = self.pca.inverse_transform(new_latent_vectors)
        return mock_spectra

    def visualize_latent_space(self, n_samples=2000, save_path=None, outlier_spectra=None, outlier_names=None):
        """
        Creates a corner plot comparing the Real Data vs KDE-Generated Data,
        optionally overlaying outlier projections.
        """
        mock_vectors = self.kde.sample(n_samples)
        real_vectors = self.latent_data
        n_dims = self.pca.n_components
        
        if outlier_spectra is not None:
            outlier_vectors = self.pca.transform(outlier_spectra)
            
        fig, axes = plt.subplots(n_dims, n_dims, figsize=(10, 9.5))
        
        for i in range(n_dims):
            for j in range(n_dims):
                ax = axes[i, j]
                
                # Diagonal: Density Histogram (1D)
                if i == j:
                    ax.hist(real_vectors[:, i], bins=10, density=True, 
                            color='#95a5a6', alpha=0.6, label='Real')
                    ax.hist(mock_vectors[:, i], bins=30, density=True, 
                            color='#f5b7b1', alpha=0.5, label='Prior')
                    
                    if outlier_spectra is not None:
                        for o_idx, o_name in enumerate(outlier_names):
                            o_val = outlier_vectors[o_idx, i]
                            # 2006RJ103: dashed, 2011SO277: dotted
                            linestyle = '--' if '2006RJ103' in o_name else ':'
                            ax.axvline(o_val, color='#2c3e50', linestyle=linestyle, alpha=0.8, linewidth=1.2)
                            
                    if i == n_dims - 1:
                        ax.set_xlabel(f"PC{i+1}", fontsize=11, fontweight='bold')
                    ax.tick_params(labelsize=10)
                        
                # Lower Triangle: Contours (2D)
                elif i > j:
                    ax.set_facecolor('#fff5f5')  # Light pinkish background for off-diagonals
                    
                    x = mock_vectors[:, j]
                    y = mock_vectors[:, i]
                    
                    # Define grid limits
                    xmin, xmax = x.min(), x.max()
                    ymin, ymax = y.min(), y.max()
                    dx = (xmax - xmin) * 0.1
                    dy = (ymax - ymin) * 0.1
                    xmin -= dx; xmax += dx
                    ymin -= dy; ymax += dy
                    
                    # Evaluate KDE on grid
                    X, Y = np.mgrid[xmin:xmax:60j, ymin:ymax:60j]
                    positions = np.vstack([X.ravel(), Y.ravel()])
                    values = np.vstack([x, y])
                    kernel = gaussian_kde(values)
                    Z = np.reshape(kernel(positions).T, X.shape)
                    
                    # Plot Mock Data as Contours
                    ax.contourf(X, Y, Z, cmap='Reds', alpha=0.3, levels=6)
                    ax.contour(X, Y, Z, colors='#c0392b', alpha=0.7, levels=6, linewidths=0.8)

                    # Real data points (colored by class)
                    ax.scatter(real_vectors[:, j], real_vectors[:, i], 
                               s=20, color=self.colors, alpha=0.8, marker='o', edgecolor='none')
                    
                    # Outliers
                    if outlier_spectra is not None:
                        for o_idx, o_name in enumerate(outlier_names):
                            o_vec = outlier_vectors[o_idx]
                            marker = 'x' if '2006RJ103' in o_name else '+'
                            ax.scatter(o_vec[j], o_vec[i],
                                       s=40, color='#2c3e50', alpha=0.9, marker=marker, linewidths=1.5)
                            
                    if i == n_dims - 1:
                        ax.set_xlabel(f"PC{j+1}", fontsize=11, fontweight='bold')
                    if j == 0:
                        ax.set_ylabel(f"PC{i+1}", fontsize=11, fontweight='bold')
                        
                    ax.tick_params(labelsize=10)
                    ax.set_xlim(xmin, xmax)
                    ax.set_ylim(ymin, ymax)
                    
                # Upper Triangle: Hide
                else:
                    ax.axis('off')
        
        # Place Legends exactly as in the user's paper image
        # Column 1: Real vs Mock Legend
        ax_leg1 = axes[0, 1]
        ax_leg1.axis('off')
        from matplotlib.patches import Patch
        legend_elements1 = [
            Patch(facecolor='#95a5a6', alpha=0.6, label='Real'),
            Patch(facecolor='#f5b7b1', alpha=0.5, label='Prior')
        ]
        ax_leg1.legend(handles=legend_elements1, loc='center left', frameon=False, fontsize=11)
        
        # Column 2: Classes Legend
        ax_leg2 = axes[0, 2]
        ax_leg2.axis('off')
        from matplotlib.lines import Line2D
        legend_elements2 = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#9c27b0', markersize=8, label='methanol'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#e74c3c', markersize=8, label='organic'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#f39c12', markersize=8, label='CO$_2$'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#3498db', markersize=8, label='H$_2$O')
        ]
        ax_leg2.legend(handles=legend_elements2, loc='center left', frameon=False, fontsize=11)
        
        # Column 3: Outliers Legend
        if outlier_spectra is not None:
            ax_leg3 = axes[0, 3]
            ax_leg3.axis('off')
            legend_elements3 = [
                Line2D([0], [0], marker='x', color='w', markeredgecolor='#2c3e50', markeredgewidth=1.5, markersize=8, label='2006RJ103'),
                Line2D([0], [0], marker='+', color='w', markeredgecolor='#2c3e50', markeredgewidth=1.5, markersize=8, label='2011SO277')
            ]
            ax_leg3.legend(handles=legend_elements3, loc='center left', frameon=False, fontsize=11)
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=250, bbox_inches='tight')
            print(f"Saved corner plot to: {save_path}")
        plt.close()

def generator():
    # Load spectra using load_and_preprocess_spectra
    import os
    import sys
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    sys.path.append(script_dir)
    from reconstruct_spectra import load_and_preprocess_spectra
    from metrics import get_object_class

    processed_dir = os.path.join(project_root, "data", "processed")
    spectra, file_names, wl_grid = load_and_preprocess_spectra(processed_dir, norm_wl=0.75)

    # Identify in-distribution objects and filter
    valid_indices = []
    valid_classes = []

    for idx, fn in enumerate(file_names):
        cls = get_object_class(fn)
        if cls in ['Water', 'CO2', 'Organic', 'Methanol']:
            valid_indices.append(idx)
            valid_classes.append(cls)

    filtered_spectra = spectra[valid_indices]

    # Map taxonomic classes to correct labels (0: Methanol, 1: Organic, 2: CO2, 3: Water)
    class_map = {'Methanol': 0, 'Organic': 1, 'CO2': 2, 'Water': 3}
    labels_filtered = np.array([class_map[cls] for cls in valid_classes])

    # Identify outliers
    outlier_spectra = []
    outlier_names = []
    for idx, fn in enumerate(file_names):
        cls = get_object_class(fn)
        if cls == 'Outlier':
            outlier_spectra.append(spectra[idx])
            clean_name = fn.split("_merged_unified")[0].split(".")[0]
            outlier_names.append(clean_name)
    outlier_spectra = np.array(outlier_spectra)

    gen = PCASpectrumGenerator(filtered_spectra, labels_filtered, n_components=5)
    return gen, filtered_spectra, wl_grid, outlier_spectra, outlier_names

if __name__ == "__main__":
    gen, real, wl, outlier_spectra, outlier_names = generator()
    save_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "plots")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "prior_corner_plot.png")
    gen.visualize_latent_space(n_samples=2000, save_path=save_path,
                               outlier_spectra=outlier_spectra, outlier_names=outlier_names)
    
    # Copy to artifact directory
    artifact_dir = "/Users/hsingwel/.gemini/antigravity/brain/db176578-ab2c-45ff-89c8-754918c29b15"
    os.makedirs(artifact_dir, exist_ok=True)
    os.system(f"cp {save_path} {artifact_dir}/prior_corner_plot.png")
    print("Successfully copied prior_corner_plot.png to artifact directory.")
