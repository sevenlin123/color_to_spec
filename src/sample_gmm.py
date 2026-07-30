import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 1. Define GMM parameters from literature table (Bernardinelli et al. 2025)
# Dimensions: [g-r, r-i, r-z]
GMM_PARAMS = {
    'NIRB': {
        'weight': 0.446,
        'mean': np.array([0.609, 0.223, 0.355]),
        'cov': np.array([
            [0.00434, 0.00337, 0.00547],
            [0.00337, 0.00268, 0.00437],
            [0.00547, 0.00437, 0.00727]
        ])
    },
    'NIRF': {
        'weight': 0.554,
        'mean': np.array([0.902, 0.347, 0.524]),
        'cov': np.array([
            [0.00674, 0.00362, 0.00557],
            [0.00362, 0.00205, 0.00321],
            [0.00557, 0.00321, 0.00533]
        ])
    }
}

def get_component_basis(comp_name):
    """
    Computes eigenvalues, eigenvectors, and aligns the major axis.
    """
    mean = GMM_PARAMS[comp_name]['mean']
    cov = GMM_PARAMS[comp_name]['cov']
    
    vals, vecs = np.linalg.eigh(cov)
    # Ascending sort: vals[2] is largest eigenvalue (l1), vecs[:, 2] is major axis (v1)
    l1, l2, l3 = vals[2], vals[1], vals[0]
    v1, v2, v3 = vecs[:, 2], vecs[:, 1], vecs[:, 0]
    
    # Align major axis v1 to point towards redder visible color (positive g-r)
    if v1[0] < 0:
        v1 = -v1
        
    return mean, v1, v2, v3, l1, l2, l3

def sample_group(group_name, n_samples=1000):
    """
    Directly samples from one of the four groups: 'NIRB+', 'NIRB-', 'NIRF+', 'NIRF-'.
    """
    comp_name = 'NIRB' if 'NIRB' in group_name else 'NIRF'
    is_plus = '+' in group_name
    
    mean, v1, v2, v3, l1, l2, l3 = get_component_basis(comp_name)
    
    # 1. Sample in the diagonalized eigenvector space
    # c1' is sampled from a Half-Normal distribution to split the Gaussian in half
    c1_abs = np.abs(np.random.normal(0, np.sqrt(l1), size=n_samples))
    c1_prime = c1_abs if is_plus else -c1_abs
    
    c2_prime = np.random.normal(0, np.sqrt(l2), size=n_samples)
    c3_prime = np.random.normal(0, np.sqrt(l3), size=n_samples)
    
    # 2. Transform back to the original color coordinates
    samples = np.zeros((n_samples, 3))
    for i in range(n_samples):
        samples[i] = mean + c1_prime[i]*v1 + c2_prime[i]*v2 + c3_prime[i]*v3
        
    return samples

def sample_gmm(n_samples=1000):
    """
    Samples from the full GMM and classifies each sample into one of the four groups:
    'NIRB+', 'NIRB-', 'NIRF+', 'NIRF-'.
    """
    weights = [GMM_PARAMS['NIRB']['weight'], GMM_PARAMS['NIRF']['weight']]
    chosen_idx = np.random.choice([0, 1], size=n_samples, p=weights)
    
    samples = np.zeros((n_samples, 3))
    groups = []
    
    # Precompute bases
    bases = {
        'NIRB': get_component_basis('NIRB'),
        'NIRF': get_component_basis('NIRF')
    }
    
    for i in range(n_samples):
        comp_name = 'NIRB' if chosen_idx[i] == 0 else 'NIRF'
        mean, v1, v2, v3, l1, l2, l3 = bases[comp_name]
        
        # Sample standard multivariate normal
        pt = np.random.multivariate_normal(mean, GMM_PARAMS[comp_name]['cov'])
        samples[i] = pt
        
        # Project onto the aligned major axis v1 to determine +/-
        proj = np.dot(pt - mean, v1)
        subgroup = '+' if proj >= 0 else '-'
        groups.append(comp_name + subgroup)
        
    return samples, groups

def visualize_samples(samples, groups, save_path=None):
    """
    Plots r-z vs g-r for samples classified into the four groups.
    """
    plt.figure(figsize=(6.5, 6))
    
    gr = samples[:, 0]
    rz = samples[:, 2]
    
    # Define color scheme for the four groups
    group_styles = {
        'NIRB+': {'color': '#e74c3c', 'label': 'NIRB+ (Redder)', 'marker': 'o'},
        'NIRB-': {'color': '#e67e22', 'label': 'NIRB- (Bluer)', 'marker': 'o'},
        'NIRF+': {'color': '#9c27b0', 'label': 'NIRF+ (Redder)', 'marker': '^'},
        'NIRF-': {'color': '#3498db', 'label': 'NIRF- (Bluer)', 'marker': '^'}
    }
    
    for group, style in group_styles.items():
        mask = np.array(groups) == group
        if np.any(mask):
            plt.scatter(gr[mask], rz[mask], s=8, color=style['color'], 
                        alpha=0.5, label=style['label'], marker=style['marker'])
            
    # Add means
    plt.scatter(GMM_PARAMS['NIRB']['mean'][0], GMM_PARAMS['NIRB']['mean'][2], 
                s=120, color='#111111', marker='X', edgecolor='white', label='NIRB Mean', zorder=5)
    plt.scatter(GMM_PARAMS['NIRF']['mean'][0], GMM_PARAMS['NIRF']['mean'][2], 
                s=120, color='#111111', marker='P', edgecolor='white', label='NIRF Mean', zorder=5)
    
    # Draw the boundary lines (cutting planes projected on g-r vs r-z)
    # The boundary is (x - mu)^T v1 = 0.
    # In 3D: v1[0]*(gr - mu[0]) + v1[1]*(ri - mu[1]) + v1[2]*(rz - mu[2]) = 0.
    # To plot on 2D (gr vs rz), we can assume ri is at the mean value (ri = mu[1]).
    # Then the line is: v1[0]*(gr - mu[0]) + v1[2]*(rz - mu[2]) = 0
    # => rz = mu[2] - (v1[0]/v1[2]) * (gr - mu[0])
    for comp in ['NIRB', 'NIRF']:
        mean, v1, v2, v3, _, _, _ = get_component_basis(comp)
        gr_grid = np.linspace(mean[0] - 0.25, mean[0] + 0.25, 100)
        rz_line = mean[2] - (v1[0]/v1[2]) * (gr_grid - mean[0])
        plt.plot(gr_grid, rz_line, color='#2c3e50', linestyle='--', alpha=0.7, 
                 linewidth=1.2, label=f'{comp} Boundary Plane' if comp == 'NIRB' else None)
        
    plt.xlabel('$g - r$', fontsize=12, fontweight='bold')
    plt.ylabel('$r - z$', fontsize=12, fontweight='bold')
    plt.xlim(0.2, 1.3)
    plt.ylim(-0.1, 1.0)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(frameon=True, fontsize=9, loc='upper left')
    plt.title('Sampled & Subdivided GMM Color-Color Distribution', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=250)
        print(f"Saved GMM verification plot to: {save_path}")
    plt.close()

if __name__ == '__main__':
    # 1. Test full GMM sampler with group classification
    n_samples = 4000
    samples, groups = sample_gmm(n_samples)
    print(f"Successfully generated {n_samples} samples from the GMM and classified into 4 groups.")
    
    for g in ['NIRB+', 'NIRB-', 'NIRF+', 'NIRF-']:
        count = sum(1 for grp in groups if grp == g)
        print(f"  {g} sample count: {count} ({count / n_samples * 100:.1f}%)")
        
    # Save verification plot
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    save_dir = os.path.join(project_root, 'plots')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'gmm_samples.png')
    visualize_samples(samples, groups, save_path=save_path)
    
    # Copy to artifact directory
    artifact_dir = "/Users/hsingwel/.gemini/antigravity/brain/e2b6098b-8b81-4f5c-8dac-d536836da393"
    os.system(f"cp {save_path} {artifact_dir}/gmm_samples.png")
    print("Successfully copied gmm_samples.png to artifact directory.")
    
    # 2. Test direct group sampler
    test_group = 'NIRB+'
    group_samples = sample_group(test_group, n_samples=5)
    print(f"\nDirectly sampled 5 points from {test_group}:")
    print(group_samples)
