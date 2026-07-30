import os
import glob
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, gaussian_kde
from scipy.interpolate import interp1d
from scipy.constants import h, c, k
from sklearn.decomposition import PCA
from sklearn.neighbors import KernelDensity
from autogluon.tabular import TabularPredictor

# Band center wavelengths and widths (in microns)
SYSTEMS = {
    'DES': {
        'bands': ['g', 'r', 'i', 'z', 'y'],
        'wavelengths': np.array([0.473, 0.642, 0.784, 0.926, 1.009]),
        'widths': np.array([0.15, 0.15, 0.15, 0.15, 0.11])
    },
    'LSST': {
        'bands': ['g', 'r', 'i', 'z', 'y'],
        'wavelengths': np.array([0.481, 0.622, 0.756, 0.868, 0.975]),
        'widths': np.array([0.15, 0.14, 0.13, 0.10, 0.09])
    }
}

def planck(wav, wav_target, T=5778):
    """
    Planck function calculation matching the reference project's scaling and implementation.
    """
    a = 2.0 * h * c**2
    b = h * c / (wav * k * T)
    b_target = h * c / (wav_target * k * T)
    intensity = a / ( (wav**5) * (np.exp(b) - 1.0) )
    intensity_target = a / ( (wav_target**5) * (np.exp(b_target) - 1.0) )
    return intensity_target/intensity.max()

def calculate_photometry(spectrum, wavelengths, widths, wl_grid, uncertainty=0.2, add_noise=True):
    """
    Calculates synthetic photometry with gaussian noise scaled by solar planck function.
    """
    spec_phot = interp1d(wl_grid, spectrum, bounds_error=False, fill_value="extrapolate")
    phots = []
    for wl_center, width in zip(wavelengths, widths):
        w_min = wl_center - width / 2
        w_max = wl_center + width / 2
        wls = np.linspace(w_min, w_max, num=100)
        
        # Calculate Planck profile across the band fine grid
        p_wls = planck(wls, w_max, w_min)
        r_wls = spec_phot(wls)
        
        # Compute observed flux F(lambda) = R(lambda) * P(lambda)
        f_wls = r_wls * p_wls
        f_mean = f_wls.mean()
        p_mean = p_wls.mean()
        
        if add_noise:
            sigma = uncertainty * f_mean
            sigma = max(sigma, 1e-9)  # Avoid standard deviation <= 0
            f_sampled = np.random.normal(f_mean, sigma)
        else:
            f_sampled = f_mean
            
        phot = f_sampled / p_mean
        phots.append(phot)
    return np.array(phots)

def load_and_preprocess_spectra(processed_dir, num_points=900, norm_wl=0.75):
    """
    Loads all 48 processed CSV files, interpolates them onto a uniform log-wavelength grid,
    and normalizes them so that the reflectance at norm_wl is 1.0.
    """
    csv_files = glob.glob(os.path.join(processed_dir, "*.csv"))
    # Filter out any files that are not the TNO or Trojan CSVs (e.g. metadata)
    csv_files = [f for f in csv_files if os.path.basename(f) != '.gitkeep' and not os.path.basename(f).startswith('spec_')]
    
    spectra_list = []
    file_names = []
    
    # Define uniform log-wavelength grid
    log_wl_grid = np.linspace(np.log(0.4), np.log(5.1), num_points)
    wl_grid = np.exp(log_wl_grid)
    
    for f in csv_files:
        df = pd.read_csv(f)
        w_col = 'um' if 'um' in df.columns else 'wavelength_um'
        w = df[w_col].values
        r = df['reflectance'].values
        
        # Resample on log-wavelength grid
        r_grid = np.interp(wl_grid, w, r)
        
        # Normalize at norm_wl
        r_norm = np.interp(norm_wl, wl_grid, r_grid)
        if r_norm > 0:
            r_grid_norm = r_grid / r_norm
            spectra_list.append(r_grid_norm)
            file_names.append(os.path.basename(f))
            
    spectra = np.vstack(spectra_list)
    print(f"Preprocessed {len(spectra)} spectra onto a log-spaced grid of {num_points} points.")
    return spectra, file_names, wl_grid

class PCASpectrumGenerator:
    def __init__(self, real_spectra=None, n_components=10, bandwidth=0.5):
        """
        Fits PCA and KDE on normalized spectra (or acts as an empty container if spectra=None).
        """
        if real_spectra is not None:
            self.pca = PCA(n_components=n_components)
            self.latent_data = self.pca.fit_transform(real_spectra)
            self.explained_variance = np.sum(self.pca.explained_variance_ratio_)
            print(f"PCA Variance Explained ({n_components} PCs): {self.explained_variance:.4f}")
            
            self.kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth)
            self.kde.fit(self.latent_data)
        else:
            self.pca = None
            self.kde = None
            self.latent_data = None
            self.explained_variance = 0.0
        
    def generate(self, n_samples=1000):
        """
        Generates synthetic spectra.
        """
        new_latent = self.kde.sample(n_samples)
        mock_spectra = self.pca.inverse_transform(new_latent)
        # Ensure non-negative reflectances
        mock_spectra = np.clip(mock_spectra, 0.0, None)
        # Re-normalize at 0.75 um to ensure consistent scaling
        # (Since 0.75 um is at some index on the grid, we find the closest index or interpolate)
        return mock_spectra

def extract_features(spectra, wl_grid, system_name, uncertainty=0.05, add_noise=True, bands_subset=None):
    """
    Extracts normalized photometry features for the chosen system (DES or LSST) using Planck-based Gaussian sampling.
    Optional bands_subset slices specific filters (e.g. ['g', 'r', 'i']).
    """
    sys_info = SYSTEMS[system_name]
    bands = sys_info['bands']
    wavelengths = sys_info['wavelengths']
    widths = sys_info['widths']
    
    if bands_subset is not None:
        indices = [bands.index(b) for b in bands_subset]
        bands = [bands[i] for i in indices]
        wavelengths = wavelengths[indices]
        widths = widths[indices]
        
    features_list = []
    for spec in spectra:
        feats = calculate_photometry(spec, wavelengths, widths, wl_grid, uncertainty=uncertainty, add_noise=add_noise)
        features_list.append(feats)
        
    features_df = pd.DataFrame(features_list, columns=[f'mag_{b}' for b in bands])
    return features_df

def train_reconstructor_model(spectra, wl_grid, system_name, n_components=10, model_dir="models/", file_names=None, bands_subset=None, norm_wl=0.75, base_pca_path=None):
    """
    Trains AutoGluon regressors to predict full PCA coefficients from 5-band photometry.
    """
    # Check if a fully-trained model cache exists in model_dir
    config_path = os.path.join(model_dir, "config.pkl")
    corr_path = os.path.join(model_dir, "correlation_matrix.npy")
    
    cache_exists = False
    if os.path.exists(config_path) and os.path.exists(corr_path):
        try:
            with open(config_path, "rb") as f:
                cached_config = pickle.load(f)
            # Ensure the cache matches n_components, system_name, bands_subset, and norm_wl
            if (cached_config.get('n_components') == n_components and 
                cached_config.get('system_name') == system_name and 
                cached_config.get('bands_subset') == bands_subset and
                cached_config.get('norm_wl', 0.75) == norm_wl):
                # Check if all AutoGluon predictor folders and their predictor.pkl files exist
                all_predictors_exist = True
                for i in range(n_components):
                    pred_file = os.path.join(model_dir, f"predictor_pc_{i}", "predictor.pkl")
                    if not os.path.exists(pred_file):
                        all_predictors_exist = False
                        break
                if all_predictors_exist:
                    cache_exists = True
        except Exception as e:
            print(f"Failed to check cache in {model_dir}: {e}. Training from scratch.")
            
    if cache_exists:
        print(f"Loading pre-trained reconstructor model from cache in {model_dir}...")
        # Load predictors
        predictors = {}
        for i in range(n_components):
            target = f"pc_{i}"
            predictors[target] = TabularPredictor.load(os.path.join(model_dir, f"predictor_{target}"))
        correlation_matrix = np.load(corr_path)
        return cached_config, predictors, correlation_matrix

    # 1. Fit or Load PCA & KDE
    if base_pca_path is None:
        base_filename = "base_pca_kde_0926.pkl" if abs(norm_wl - 0.926) < 1e-3 else "base_pca_kde_075.pkl"
        base_pca_path = os.path.join(os.path.dirname(model_dir), base_filename)
        if not os.path.exists(base_pca_path):
            base_pca_path = os.path.join(os.path.dirname(model_dir), "base_pca_kde.pkl")
    
    if os.path.exists(base_pca_path):
        print(f"Loading dedicated PCA and KDE model from {base_pca_path}...")
        with open(base_pca_path, "rb") as f:
            base_data = pickle.load(f)
        pca = base_data['pca']
        kde = base_data['kde']
        wl_grid = base_data['wl_grid']
        generator = base_data.get('generator')
        if generator is None:
            generator = PCASpectrumGenerator(real_spectra=None, n_components=n_components)
            generator.pca = pca
            generator.kde = kde
            generator.latent_data = base_data.get('latent_data')
    else:
        if spectra is None:
            raise ValueError(f"Base PCA/KDE model not found at {base_pca_path} and no spectra provided for training.")
        print(f"Fitting dedicated PCA and training KDE generator (norm_wl={norm_wl})...")
        generator = PCASpectrumGenerator(spectra, n_components=n_components)
        
    # 2. Generate augmented training data (10,000 synthetic spectra)
    print("Generating synthetic training data...")
    synthetic_spectra = generator.generate(n_samples=10000)
    
    # 3. Extract features and targets
    X_train = extract_features(synthetic_spectra, wl_grid, system_name, bands_subset=bands_subset)
    Y_train_pca = generator.pca.transform(synthetic_spectra)
    
    # Determine train and validation splits for AutoGluon and residuals
    if spectra is None:
        X_train_fit = X_train.iloc[:9000]
        Y_train_fit = Y_train_pca[:9000]
        X_val = X_train.iloc[9000:]
        Y_val_pca = Y_train_pca[9000:]
    else:
        X_train_fit = X_train
        Y_train_fit = Y_train_pca
        X_val = extract_features(spectra, wl_grid, system_name, bands_subset=bands_subset)
        Y_val_pca = generator.latent_data
        
    # Prepare training dataframe
    train_df = X_train_fit.copy()
    target_cols = [f'pc_{i}' for i in range(n_components)]
    for i, col in enumerate(target_cols):
        train_df[col] = Y_train_fit[:, i]
        
    # 4. Train AutoGluon predictors
    predictors = {}
    dense_quantiles = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
                       0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]
    
    os.makedirs(model_dir, exist_ok=True)
    
    print(f"Training AutoGluon predictors for {n_components} PCs...")
    for i, target in enumerate(target_cols):
        print(f"Training model for Component {target}...")
        # Drop other target PCs to avoid cheating
        train_data_specific = train_df.drop(columns=[t for t in target_cols if t != target])
        
        predictor = TabularPredictor(
            label=target,
            problem_type='quantile',
            quantile_levels=dense_quantiles,
            path=os.path.join(model_dir, f"predictor_{target}"),
            verbosity=0
        )
        
        predictor.fit(
            train_data_specific,
            hyperparameters={'GBM': {}},
            time_limit=60
        )
        predictors[target] = predictor
        
    # 5. Compute validation residuals and correlation matrix
    if spectra is not None:
        print("Computing residual correlation matrix on real spectra...")
    else:
        print("Computing residual correlation matrix on synthetic validation split...")
        
    predictions = []
    for i, target in enumerate(target_cols):
        # Predict all quantiles, then extract the median (0.50 quantile)
        pred_all = predictors[target].predict(X_val)
        median_col = min(pred_all.columns, key=lambda c: abs(float(c) - 0.5))
        predictions.append(pred_all[median_col].values.ravel())
        
    predictions = np.column_stack(predictions)
    residuals = Y_val_pca - predictions
    correlation_matrix = np.corrcoef(residuals.T)
    np.save(os.path.join(model_dir, "correlation_matrix.npy"), correlation_matrix)
    print("Residual correlation matrix computed and saved.")
    
    # 6. Get or compute prior and class distributions
    from metrics import get_object_class, fit_gaussian_distribution, RegularizedKDE
    
    if spectra is not None:
        prior_mean, prior_cov = fit_gaussian_distribution(generator.latent_data)
        kde_prior = RegularizedKDE(generator.latent_data[:, :5].T, reg=0.05)
        
        class_vectors = {
            'Water': [],
            'Organic': [],
            'Methanol': [],
            'CO2': []
        }
        
        if file_names is not None:
            for idx, f_name in enumerate(file_names):
                cls = get_object_class(f_name)
                if cls in class_vectors:
                    class_vectors[cls].append(generator.latent_data[idx])
                    
        class_distributions = {}
        kde_classes = {}
        for cls in class_vectors:
            vectors = np.array(class_vectors[cls])
            if len(vectors) > 0:
                c_mean, c_cov = fit_gaussian_distribution(vectors)
                v_sub = vectors[:, :5]
                kde_classes[cls] = RegularizedKDE(v_sub.T, reg=0.05)
            else:
                c_mean = np.zeros(n_components)
                c_cov = np.eye(n_components)
                kde_classes[cls] = RegularizedKDE(np.random.normal(0, 1, size=(5, 50)), reg=0.05)
            class_distributions[cls] = (c_mean, c_cov)
            
        # Save base model including all metrics (first-time fitting)
        if not os.path.exists(base_pca_path):
            base_data = {
                'pca': generator.pca,
                'kde': generator.kde,
                'wl_grid': wl_grid,
                'generator': generator,
                'latent_data': generator.latent_data,
                'prior_mean': prior_mean,
                'prior_cov': prior_cov,
                'class_distributions': class_distributions,
                'kde_prior': kde_prior,
                'kde_classes': kde_classes
            }
            os.makedirs(os.path.dirname(base_pca_path), exist_ok=True)
            with open(base_pca_path, "wb") as f:
                pickle.dump(base_data, f)
            print(f"Saved complete base PCA and KDE model to {base_pca_path}")
    else:
        # Load from base_data
        prior_mean = base_data['prior_mean']
        prior_cov = base_data['prior_cov']
        class_distributions = base_data['class_distributions']
        kde_prior = base_data['kde_prior']
        kde_classes = base_data['kde_classes']
        
    config = {
        'pca': generator.pca,
        'wl_grid': wl_grid,
        'system_name': system_name,
        'n_components': n_components,
        'bands_subset': bands_subset,
        'norm_wl': norm_wl,
        'prior_mean': prior_mean,
        'prior_cov': prior_cov,
        'class_distributions': class_distributions,
        'kde_prior': kde_prior,
        'kde_classes': kde_classes
    }
    with open(os.path.join(model_dir, "config.pkl"), "wb") as f:
        pickle.dump(config, f)
    print("Prior and class distribution parameters saved to config.pkl.")
    
    return config, predictors, correlation_matrix

def reconstruct_spectrum(features_df, model_dir="models/", num_samples=1000, return_samples=False):
    """
    Reconstructs the normalized full spectrum with confidence intervals using the trained models.
    """
    with open(os.path.join(model_dir, "config.pkl"), "rb") as f:
        config = pickle.load(f)
    pca = config['pca']
    wl_grid = config['wl_grid']
    n_components = config['n_components']
    
    correlation_matrix = np.load(os.path.join(model_dir, "correlation_matrix.npy"))
    
    # Load predictors
    predictors = {}
    for i in range(n_components):
        target = f"pc_{i}"
        predictors[target] = TabularPredictor.load(os.path.join(model_dir, f"predictor_{target}"))
        
    # Generate multivariate normal samples correlated by validation residual correlation matrix
    mv_normal_samples = np.random.multivariate_normal(
        mean=np.zeros(n_components),
        cov=correlation_matrix,
        size=num_samples
    )
    
    # Map to uniform probabilities [0, 1]
    correlated_uniforms = norm.cdf(mv_normal_samples)
    
    # Map back to PC space using predicted quantiles
    final_samples = np.zeros((num_samples, n_components))
    for dim in range(n_components):
        target = f"pc_{dim}"
        pred_quantiles = predictors[target].predict(features_df)
        q_levels = [float(c) for c in pred_quantiles.columns]
        q_values = pred_quantiles.values[0]
        
        inverse_cdf = interp1d(
            x=q_levels,
            y=q_values,
            kind='linear',
            bounds_error=False,
            fill_value=(q_values[0], q_values[-1])
        )
        final_samples[:, dim] = inverse_cdf(correlated_uniforms[:, dim])
        
    # Reconstruct back to full spectrum
    recon_samples = pca.inverse_transform(final_samples)
    recon_samples = np.clip(recon_samples, 0.0, None)
    
    # Force exact normalization to 1.0 at norm_wl
    norm_wl = config.get('norm_wl', 0.75)
    idx_norm = np.argmin(np.abs(wl_grid - norm_wl))
    recon_samples = recon_samples / recon_samples[:, idx_norm].reshape(-1, 1)
    
    # Calculate median and percentiles
    median_spec = np.median(recon_samples, axis=0)
    lower_spec = np.percentile(recon_samples, 5, axis=0)
    upper_spec = np.percentile(recon_samples, 95, axis=0)
    
    if return_samples:
        return wl_grid, median_spec, lower_spec, upper_spec, final_samples
    return wl_grid, median_spec, lower_spec, upper_spec
