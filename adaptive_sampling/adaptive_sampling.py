import torch
from torchkde import KernelDensity
import scipy.stats as st 

def uniform_input_sampler(M, d=1, bounds=(-1, 1), random_state=None):
    if random_state is not None: 
        torch.manual_seed(random_state)
    return (bounds[1] - bounds[0]) * torch.rand(M, d) + bounds[0]

def fit_kde(trained_net, input_sampler, M=100000, kde_kernel='gaussian', random_state=None): 
    # Step 1: Sample from input prior p(x)
    X_samples = input_sampler(M, random_state=random_state)  # shape (M, d)

    # Step 2: Push samples through model to get µ(x)
    mu_samples, lower, upper = trained_net.predict(X_samples)
    
    mu_samples = mu_samples.reshape(-1, 1)
    bandwidth = 1.06 * torch.std(mu_samples) * len(mu_samples) ** (-1/5)

    # Step 3: Fit KDE to {mu(x)} to get p_mu(y)
    kde = KernelDensity(kernel=kde_kernel, bandwidth=bandwidth)
    kde.fit(mu_samples)  # Fit KDE to predicted means
    return kde

def compute_weighting(trained_net, x_query, kde, d, bounds): 
    # run x_query through model: 
    mean, lower, upper = trained_net.predict(x_query)
    mean = mean.reshape(-1, 1)

    log_density = kde.score_samples(mean)

    log_area = d * torch.log(bounds[1]- bounds[0])
    alpha = trained_net.alpha 

    z = st.norm.ppf(1 - alpha / 2)
    std = (upper - lower) / (2 * z)
    std = std.reshape(-1, 1)

    log_weights = 2 * torch.log(std) - log_area - log_density

    weights = torch.exp(log_weights)

    return weights

def compute_US_LW_acquisition_function(trained_net, X_train, y_train, n_samples):
    pass 