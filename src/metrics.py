import os
import numpy as np
from scipy.stats import gaussian_kde
from scipy import linalg

class RegularizedKDE(gaussian_kde):
    """
    Subclass of scipy.stats.gaussian_kde that adds diagonal regularization
    to the covariance matrix to prevent singularity with small sample sizes.
    """
    def __init__(self, dataset, bw_method=None, weights=None, reg=0.05):
        self.reg = reg
        super().__init__(dataset, bw_method=bw_method, weights=weights)
        
    def _compute_covariance(self):
        self.factor = self.covariance_factor()
        if not hasattr(self, '_data_cho_cov'):
            if self.dataset.shape[1] <= 1:
                raw_cov = np.eye(self.dataset.shape[0])
            else:
                raw_cov = np.atleast_2d(np.cov(self.dataset, rowvar=1, bias=False, aweights=self.weights))
                if np.any(np.isnan(raw_cov)):
                    raw_cov = np.eye(self.dataset.shape[0])
            # Apply diagonal regularization
            self._data_covariance = raw_cov + self.reg * np.eye(raw_cov.shape[0])
            self._data_cho_cov = linalg.cholesky(self._data_covariance, lower=True)
            
        self.covariance = self._data_covariance * self.factor**2
        self.cho_cov = (self._data_cho_cov * self.factor).astype(np.float64)
        self.log_det = 2 * np.log(np.diag(self.cho_cov * np.sqrt(2 * np.pi))).sum()


# Class mappings for TNOs (by ID number)
TNO_CLASSES = {
    # Methanol-rich (label 0)
    20: 'Methanol', 19: 'Methanol', 13: 'Methanol', 39: 'Methanol', 15: 'Methanol', 29: 'Methanol',
    
    # Organic-rich (label 1)
    40: 'Organic', 58: 'Organic', 59: 'Organic', 104: 'Organic', 45: 'Organic', 43: 'Organic', 6: 'Organic', 57: 'Organic', 56: 'Organic',
    
    # CO2-rich (label 2)
    11: 'CO2', 133: 'CO2', 9: 'CO2', 114: 'CO2', 12: 'CO2', 153: 'CO2', 116: 'CO2', 7: 'CO2', 47: 'CO2', 51: 'CO2', 21: 'CO2', 10: 'CO2', 3: 'CO2', 5: 'CO2', 30: 'CO2', 42: 'CO2', 31: 'CO2', 44: 'CO2', 46: 'CO2', 36: 'CO2', 2: 'CO2', 25: 'CO2',
    
    # Water-rich (label 3)
    18: 'Water', 55: 'Water', 32: 'Water', 124: 'Water', 27: 'Water', 22: 'Water', 34: 'Water', 41: 'Water', 52: 'Water', 1: 'Water', 54: 'Water', 50: 'Water', 48: 'Water', 24: 'Water'
}

# Class mappings for Neptune Trojans (by string name)
NT_CLASSES = {
    '2013VX30': 'Methanol',
    '2007VL305': 'Water',
    '2011HM102': 'Water',
    '2010TS191': 'Water',
    '2011WG157': 'Water',
    '2008LC18': 'Water',
    '2006RJ103': 'Outlier',
    '2011SO277': 'Outlier'
}

def get_object_class(filename):
    """
    Returns the spectral class ('Water', 'Organic', 'Methanol', 'CO2', or 'Outlier') 
    given the spectrum file path or object ID.
    """
    base = os.path.basename(filename)
    name, ext = os.path.splitext(base)
    # Strip suffixes and convert string representing float/int to clean integer
    name_clean = name.split("_merged_unified")[0].split(".")[0]
    
    if name_clean in NT_CLASSES:
        return NT_CLASSES[name_clean]
    try:
        obj_id = int(name_clean)
        return TNO_CLASSES.get(obj_id, "Unknown")
    except ValueError:
        return "Unknown"


def compute_entropy(cov_matrix):
    """
    Computes Shannon entropy for a multivariate Gaussian distribution:
    H = 0.5 * ln((2 * pi * e)^k * |Sigma|)
    """
    k = cov_matrix.shape[0]
    sign, logdet = np.linalg.slogdet(cov_matrix)
    # Using slogdet for numerical stability with small determinants
    return 0.5 * (k * np.log(2.0 * np.pi * np.e) + logdet)

def compute_kl_divergence(mu1, cov1, mu2, cov2, reg=0.05):
    """
    Computes Kullback-Leibler divergence between two multivariate Gaussian distributions:
    D_KL( N(mu1, cov1) || N(mu2, cov2) )
    Closed-form solution:
    0.5 * [ ln(|cov2|/|cov1|) - k + tr(cov2^-1 * cov1) + (mu1-mu2)^T * cov2^-1 * (mu1-mu2) ]
    """
    k = mu1.shape[0]
    
    # Regularize covariances to ensure positive-definiteness and invertibility
    cov1_reg = cov1 + reg * np.eye(k)
    cov2_reg = cov2 + reg * np.eye(k)
    
    sign1, logdet1 = np.linalg.slogdet(cov1_reg)
    sign2, logdet2 = np.linalg.slogdet(cov2_reg)
    
    cov2_inv = np.linalg.inv(cov2_reg)
    
    # Trace term: tr(Sigma_2^-1 * Sigma_1)
    trace_term = np.trace(cov2_inv @ cov1_reg)
    
    # Mean difference term: (mu_1 - mu_2)^T * Sigma_2^-1 * (mu_1 - mu_2)
    diff = mu1 - mu2
    diff_term = diff.T @ cov2_inv @ diff
    
    kl = 0.5 * ((logdet2 - logdet1) - k + trace_term + diff_term)
    return max(kl, 0.0)  # KL divergence is mathematically non-negative

def compute_class_probabilities(kl_divergences):
    """
    Converts KL divergences to standard classes into soft probabilities:
    Score_i = exp(-KL_i), P_i = Score_i / sum(Score_j)
    """
    scores = np.exp(-np.array(kl_divergences))
    total = np.sum(scores)
    if total == 0:
        return np.ones_like(scores) / len(scores)
    return scores / total

def fit_gaussian_distribution(latent_vectors, reg=0.05):
    """
    Estimates the mean vector and covariance matrix from a set of latent vectors,
    applying diagonal shrinkage/regularization for numerical stability.
    """
    mu = np.mean(latent_vectors, axis=0)
    if len(latent_vectors) <= 1:
        # Fallback to identity if we have insufficient samples
        cov = np.eye(latent_vectors.shape[1])
    else:
        cov = np.cov(latent_vectors, rowvar=False)
    
    # Apply regularization to the diagonal of the covariance matrix
    cov_reg = cov + reg * np.eye(cov.shape[0])
    return mu, cov_reg

def calculate_all_metrics(mu_post, cov_post, config, filename, final_samples=None):
    """
    Computes all information metrics for a given posterior distribution 
    and returns a structured dict of outcomes matching the recommended schema.
    All entropy and KL divergence calculations are restricted to the first 5 PCA dimensions.
    """
    k_limit = 5
    
    mu_post_sub = mu_post[:k_limit]
    cov_post_sub = cov_post[:k_limit, :k_limit]
    
    prior_mean = config['prior_mean'][:k_limit]
    prior_cov = config['prior_cov'][:k_limit, :k_limit]
    class_dist = config['class_distributions']
    
    # 1. Closed-form Gaussian metrics
    h_prior = compute_entropy(prior_cov)
    h_post = compute_entropy(cov_post_sub)
    ig = h_prior - h_post
    kl_prior = compute_kl_divergence(mu_post_sub, cov_post_sub, prior_mean, prior_cov)
    
    # Metric 4: Class KL Divergence (legacy Gaussian-based for classification)
    classes = ['Water', 'Organic', 'Methanol', 'CO2']
    kl_classes = {}
    for cls in classes:
        cls_mean, cls_cov = class_dist[cls]
        cls_mean_sub = cls_mean[:k_limit]
        cls_cov_sub = cls_cov[:k_limit, :k_limit]
        kl_classes[cls] = compute_kl_divergence(mu_post_sub, cov_post_sub, cls_mean_sub, cls_cov_sub)
        
    # Metric 5: Class Probability
    kl_values = [kl_classes[c] for c in classes]
    p_values = compute_class_probabilities(kl_values)
    p_classes = {cls: p for cls, p in zip(classes, p_values)}
    
    # 2. Non-parametric KDE metrics
    kde_prior = config.get('kde_prior', None)
    kde_classes = config.get('kde_classes', None)
    
    if kde_prior is not None and kde_classes is not None:
        if final_samples is None:
            # Fallback: sample from the Gaussian posterior
            final_samples = np.random.multivariate_normal(mu_post, cov_post, size=1000)
            
        final_samples_sub = final_samples[:, :k_limit]
        
        # Surprisal_KDE = -1/N * sum(log p_prior^KDE(z_j))
        log_probs_prior = kde_prior.logpdf(final_samples_sub.T)
        surprisal_kde = -np.mean(log_probs_prior)
        
        # Fit KDE on the posterior samples
        kde_post = gaussian_kde(final_samples_sub.T)
        log_probs_post = kde_post.logpdf(final_samples_sub.T)
        
        # Entropy_KDE = -1/N * sum(log p_post^KDE(z_j))
        entropy_kde = -np.mean(log_probs_post)
        
        # KL_KDE = 1/N * sum(log p_post^KDE(z_j) - log p_prior^KDE(z_j))
        kl_kde = np.mean(log_probs_post - log_probs_prior)
        
        # Non-parametric KDE Class metrics
        kl_classes_kde = {}
        for cls in classes:
            kde_cls = kde_classes[cls]
            log_probs_cls = kde_cls.logpdf(final_samples_sub.T)
            kl_classes_kde[cls] = np.mean(log_probs_post - log_probs_cls)
            
        kl_values_kde = [kl_classes_kde[c] for c in classes]
        p_values_kde = compute_class_probabilities(kl_values_kde)
        p_classes_kde = {cls: p for cls, p in zip(classes, p_values_kde)}
    else:
        import warnings
        warnings.warn("kde_prior or kde_classes is missing from config. Falling back to Gaussian metrics. Please retrain/recreate the model.")
        surprisal_kde = kl_prior
        entropy_kde = h_post
        kl_kde = kl_prior
        kl_classes_kde = {cls: kl_classes[cls] for cls in classes}
        p_classes_kde = {cls: p_classes[cls] for cls in classes}
    
    # Extract Object_ID from file path
    base = os.path.basename(filename)
    obj_id = base.split('_merged_unified')[0].split('.csv')[0]
    
    return {
        'Object_ID': obj_id,
        'Information_Gain': ig,
        'KL_Water': kl_classes['Water'],
        'KL_Organic': kl_classes['Organic'],
        'KL_Methanol': kl_classes['Methanol'],
        'KL_CO2': kl_classes['CO2'],
        'P_Water': p_classes['Water'],
        'P_Organic': p_classes['Organic'],
        'P_Methanol': p_classes['Methanol'],
        'P_CO2': p_classes['CO2'],
        
        # Explicit fields
        'Entropy_Gaussian': h_post,
        'KL_Gaussian': kl_prior,
        'Surprisal_KDE': surprisal_kde,
        'Entropy_KDE': entropy_kde,
        'KL_KDE': kl_kde,
        
        # KDE-based class metrics
        'KL_Water_KDE': kl_classes_kde['Water'],
        'KL_Organic_KDE': kl_classes_kde['Organic'],
        'KL_Methanol_KDE': kl_classes_kde['Methanol'],
        'KL_CO2_KDE': kl_classes_kde['CO2'],
        'P_Water_KDE': p_classes_kde['Water'],
        'P_Organic_KDE': p_classes_kde['Organic'],
        'P_Methanol_KDE': p_classes_kde['Methanol'],
        'P_CO2_KDE': p_classes_kde['CO2']
    }

if __name__ == "__main__":
    print("Running metrics.py self-test...")
    k = 10
    mu1 = np.zeros(k)
    cov1 = np.eye(k)
    
    mu2 = np.ones(k) * 0.5
    cov2 = np.eye(k) * 1.5
    
    h_prior = compute_entropy(cov2)
    h_post = compute_entropy(cov1)
    ig = h_prior - h_post
    
    kl = compute_kl_divergence(mu1, cov1, mu2, cov2)
    
    print("Self-test success!")
    print(f"k = {k}")
    print(f"Entropy 1 (Post): {h_post:.4f}")
    print(f"Entropy 2 (Prior): {h_prior:.4f}")
    print(f"Information Gain: {ig:.4f}")
    print(f"KL Divergence: {kl:.4f}")
