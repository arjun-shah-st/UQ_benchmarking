from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from scipy.stats import zscore
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import norm

def estimate_predictive_log_likelihood(y_true, y_pred, y_std):
    """
    Estimate the average log-likelihood of observed y under predicted Gaussian intervals.
    """
    eps = 1e-8  # prevent divide-by-zero
    y_std = np.maximum(y_std, eps)  # stabilize very small stds

    log_likelihoods = norm.logpdf(y_true, loc=y_pred, scale=y_std)
    return np.mean(log_likelihoods)

def generate_composite_function(X, complexity_level=3):
    n_samples, intrinsic_dim = X.shape
    y = np.zeros(n_samples)

    for i in range(min(intrinsic_dim, complexity_level + 1)):
        xi = X[:, i]
        xj = X[:, (i + 1) % intrinsic_dim]
        xk = X[:, (i + 2) % intrinsic_dim]

        if complexity_level == 1:
            y += 0.5 * xi
        elif complexity_level == 2:
            y += 0.3 * np.sin(2 * np.pi * xi) + 0.2 * xi**2
        elif complexity_level == 3:
            y += 0.3 * np.sin(3 * xi) + 0.4 * xi**2 + 0.2 * xi * xj
        elif complexity_level == 4:
            y += 0.2 * np.sin(5 * xi) * xj**2 + 0.1 * np.exp(-5 * (xi - 0.5)**2)
        else:
            y += 0.1 * np.sin(10 * xi) * np.cos(5 * xk) + \
                 0.2 * xi**3 + 0.1 * np.abs(xj - 0.5)
    return y


def generate_noise(x, noise_type="gaussian", noise_scale=0.1, heteroscedasticity_scale = 0.0, outlier_pct = 0.0):
    n = x.shape[0]
    local_scale = np.mean(x**2, axis=1)  # smooth function of inputs
    local_scale = (local_scale - local_scale.min()) / (local_scale.max() - local_scale.min() + 1e-8)  # normalize to [0, 1]
    if noise_type == "gaussian": 
        base_noise = np.random.normal(0, noise_scale, size=n)
        hetero_noise = np.random.normal(0, noise_scale * local_scale, size=n)
    elif noise_type == "laplace":
        base_noise = np.random.laplace(0, noise_scale, size=n)
        hetero_noise = np.random.laplace(0, noise_scale * local_scale, size=n)
    elif noise_type == "uniform":
        base_noise = np.random.uniform(-noise_scale, noise_scale, size=(x.shape[0],))
        hetero_noise = np.random.uniform(-noise_scale * local_scale, noise_scale * local_scale, size=n)
    else:
        raise ValueError(f"Unknown noise_type: {noise_type}")

    noise = (1 - heteroscedasticity_scale) * base_noise + heteroscedasticity_scale * hetero_noise
    
    return noise


def estimate_outliers(residuals, threshold=3.0):
    z_scores = np.abs(zscore(residuals))
    return int(np.sum(z_scores > threshold))

def compute_poly_r2(X_train, X_test, y_train, y_test, degree):
    poly_model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    poly_model.fit(X_train, y_train)
    return r2_score(y_test, poly_model.predict(X_test))


def extract_dataset_meta_features(X, y, test_size=0.2):
    # Fit GPR to obtain residuals
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) \
         + WhiteKernel(noise_level=1e-1, noise_level_bounds=(1e-5, 1e1))
    model = GaussianProcessRegressor(kernel=kernel, normalize_y=True, optimizer='fmin_l_bfgs_b', n_restarts_optimizer=5)
    model.fit(X_train, y_train)
    y_pred, y_std = model.predict(X_test, return_std=True)
    # Log Likelihood of GPR fit: 
    log_likelihood = estimate_predictive_log_likelihood(y_test, y_pred, y_std)
    residuals = y_test- y_pred

    # Signal and noise estimation
    signal_var = np.var(y_pred)
    noise_var = np.var(residuals)
    snr = signal_var / noise_var if noise_var > 0 else np.inf

    # Heteroscedasticity estimate using residuals vs features
    abs_res = np.abs(residuals)
    model_het = LinearRegression().fit(X_test, abs_res)
    het_score = model_het.score(X_test, abs_res)

    eps = 1e-3
    grad_norms = []

    for i in range(X.shape[1]):
        X_shifted = X_test.copy()
        X_shifted[:, i] += eps
        y_shifted = model.predict(X_shifted)
        gradient_i = (y_shifted - y_pred) / eps
        grad_norms.append(gradient_i)

    grad_norms = np.vstack(grad_norms)  # shape: (input_dim, n_samples)
    grad_magnitude = np.sqrt(np.sum(grad_norms**2, axis=0))  # L2 norm per sample
    # Local smoothness from GPR prediction surface
    local_smoothness = np.mean(grad_magnitude)


    # Polynomial approximation of GPR fit
    linear_r2 = compute_poly_r2(X_train, X_test, y_train, y_test, degree=1)
    poly2_r2 = compute_poly_r2(X_train, X_test, y_train, y_test, degree=2)
    poly5_r2 = compute_poly_r2(X_train, X_test, y_train, y_test, degree=5)
    gpr_r2 = r2_score(y_test, y_pred)


    return {
        "n_samples": X.shape[0],
        "input_dim": X.shape[1],
        "output_variance": np.var(y),
        "signal_to_noise": snr,
        "n_outliers": int(np.sum(zscore(residuals) > 3)),
        "heteroscedasticity": het_score,
        "complexity_linear_r2": linear_r2,
        "complexity_poly2_r2": poly2_r2,
        "complexity_poly5_r2": poly5_r2,
        'complexity_gpr_r2': gpr_r2,
        'complexity_log_likelihood': log_likelihood,
        "local_smoothness": local_smoothness
    }

def generate_synthetic_dataset(
    n_samples=500,
    input_dim=10,
    intrinsic_dim=3,
    complexity_level=3,
    noise_type="gaussian",
    noise_scale=0.1,
    heteroscedasticity=0,
    random_state=None,
    test_size=0.2,
):
    np.random.seed(random_state)

    X = np.random.uniform(0, 1, size=(n_samples, input_dim))
    indices = np.random.choice(input_dim, size=intrinsic_dim, replace=False)
    X_intrinsic = X[:, indices]

    y_true = generate_composite_function(X_intrinsic, complexity_level)
    noise = generate_noise(X_intrinsic, noise_type, noise_scale, heteroscedasticity)
    y = y_true + noise

    X = StandardScaler().fit_transform(X.reshape(-1, input_dim))
    y = StandardScaler().fit_transform(y.reshape(-1, 1))

    X_train, X_test, y_train, y_test, y_true_train, y_true_test = train_test_split(
        X, y, y_true, test_size=test_size, random_state=random_state
    )

    meta = extract_dataset_meta_features(X_train, y_train)
    meta.update({
        "intrinsic_dim": intrinsic_dim,
        "intrinsic_dims_used": indices.tolist(),
        "noise_type": noise_type,
        "complexity_level": complexity_level,
    })

    return X_train, X_test, y_train, y_test, meta


def plot_synthetic_dataset(X_train, y_train):
    df = pd.DataFrame(X_train, columns=[f"x{i}" for i in range(X_train.shape[1])])
    df["y"] = y_train
    pd.plotting.scatter_matrix(df, alpha=0.5, figsize=(12, 12))
    plt.tight_layout()
    plt.show()


# Example usage
if __name__ == "__main__":
    X_train, X_test, y_train, y_test, meta = generate_synthetic_dataset(
        input_dim=4,
        intrinsic_dim=4,
        complexity_level=2,
        noise_scale=0.1,
        heteroscedasticity=1,
        random_state=42
    )

    print("Meta-features:")
    for k, v in meta.items():
        print(f"  {k}: {v}")

    plot_synthetic_dataset(X_train, y_train)
