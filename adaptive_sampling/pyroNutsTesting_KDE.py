import matplotlib.pyplot as plt
import numpy as np
from .quantile_regression_test import analyze_quantiles, analyze_quantiles_dim, analyze_conformal_quantiles
import pyro
import pyro.distributions as dist
import torch
from pyro.infer import MCMC, NUTS
import seaborn as sns
import pandas as pd
from scipy.stats import norm
from UBAS.generators.input_generator import InputGenerator
from torchkde import KernelDensity

def main(ndim, npoints=1000):
    nn, init_inputs, init_targets = analyze_conformal_quantiles(ndim, npoints, False)
    bounds = np.ones((2, ndim)) * np.array([[-1, 1]]).T 
    input_sampler = InputGenerator(bounds, ndim, seed=42)
    mean, lower, upper = nn.predict(init_inputs)
    residuals = torch.absolute(mean - init_targets)
    largest_residual = 2 * torch.max(residuals).to(torch.float32)
    sample_inputs = input_sampler.uniformly_sample(batch_samples=100000)
    sample_mean, sample_lower, sample_upper = nn.predict(sample_inputs)
    sample_mean = sample_mean.reshape(-1, 1) 
    bandwidth = 1.06 * torch.std(sample_mean) * len(sample_mean) ** (-1/5)
    kde = KernelDensity(kernel='cauchy', bandwidth=bandwidth)
    kde.fit(sample_mean)

    samples = torch.linspace(0, 1, 100)
    scores = kde.score_samples(samples.reshape(-1, 1)).detach()
    #scores = torch.exp(-scores).detach().reshape(-1, 1) * (torch.abs(upper - lower) / largest_residual).reshape(-1, 1) ** 2
    plt.plot(samples.detach(), scores.detach())
    plt.show()

    print(largest_residual)
    def log_pdf(x):
        mean, lower, upper = nn.predict(x.unsqueeze(0))
        log_sigma = torch.log(torch.abs(upper - lower) / largest_residual)
        #log_pdf = dist.MultivariateNormal(torch.tensor(np.zeros(ndim)), torch.tensor(np.identity(ndim) * 0.25)).log_prob(x)
        log_w = kde.score_samples(mean.reshape(-1, 1))    

        return 2 * log_sigma - log_w

    def model():
        x = pyro.sample("x", dist.Uniform(torch.ones(ndim) * -1, torch.ones(ndim)))
        log_likelihood = log_pdf(x)

        pyro.factor("log_likelihood", log_likelihood)


    def run_nuts(model, initial_params=None):
        nuts_kernel = NUTS(model)
        mcmc = MCMC(nuts_kernel, num_samples=200, warmup_steps=300, initial_params=initial_params)
        mcmc.run()

        samples = mcmc.get_samples()

        x_samples = samples["x"].cpu().numpy()  # shape: (num_samples, num_variables)
        predictions = nn.predict(x_samples)
        lb = predictions[1]
        ub = predictions[2]
        widths = ub - lb
        num_samples, num_chains = x_samples.shape

        plt.figure(figsize=(12, 4))
        for chain in range(num_chains):
            plt.plot(x_samples[:, chain], label=f"Variable {chain+1}")
        plt.title("Trace Plot for x")
        plt.xlabel("Sample")
        plt.ylabel("x")
        plt.legend()
        plt.tight_layout()
        plt.show()



        #for chain in range(num_chains):
        #    plt.hist(x_samples[:, chain], density=True, bins=40, edgecolor='black')
        #    plt.xlabel(f"Variable {chain+1}")
        #    plt.ylabel("Normalized Ocurrences")
        #    x = np.linspace(-1, 1, 100)
        #    pdf = []
        #    for x_val in x:
        #        pdf.append(np.exp(log_pdf(torch.tensor([x_val]))))
        #    plt.plot(x, pdf)
        #    plt.show()

        # Convert x_samples (PyTorch tensor) to a Pandas DataFrame
        # Assume x_samples shape is [num_samples, num_dims]
        x_samples_np = x_samples.detach().numpy() if isinstance(x_samples, torch.Tensor) else x_samples
        num_dims = x_samples_np.shape[1]
        df = pd.DataFrame(x_samples_np, columns=[f"x{i+1}" for i in range(num_dims)])
        df["widths"] = widths.detach()
        df["values"] = lb.detach() + widths.detach() / 2
        # Custom function to overlay the standard normal curve
        def add_normal_curve(ax, *args, **kwargs):
            x_vals = np.linspace(-1, 1, 50)
            y_vals = norm.pdf(x_vals, loc=0, scale=0.2 ** 0.5)
            y_vals_scaled = y_vals * (ax.get_ylim()[1]-ax.get_ylim()[0]) / max(y_vals) * 0.84 - 1 # scale to match histogram height
            if i == 0:
                y_vals_scaled = y_vals * (ax.get_ylim()[1]-ax.get_ylim()[0]) / max(y_vals) * 0.84 + 0
            ax.plot(x_vals, y_vals_scaled, color='red', lw=2)

        # Create the pairplot with histograms and overlay normal
        if ndim >= 10:
            g = sns.pairplot(df, diag_kind="hist", corner=True, vars = ['x1', 'x3', 'x5', 'x8', 'x10'])
        else:
            g = sns.pairplot(df, diag_kind="hist", corner=True)

        # Add the normal curve to the diagonal histograms
        for i in range(num_dims if num_dims < 6 else 5):
            ax = g.axes[i][i]
            if ax is not None:
                add_normal_curve(ax)

        plt.suptitle("Pairplot of Samples with Normal Overlay", y=1.02)
        plt.show()

        flat_samples = x_samples.reshape(-1)  # Flatten all chains
        plt.figure(figsize=(6, 4))
        sns.histplot(flat_samples, bins=40, kde=True, color="skyblue")
        plt.title("Posterior Distribution of x")
        plt.xlabel("x")
        plt.ylabel("Density")
        plt.tight_layout()
        plt.show()


        ranks = np.argsort(np.argsort(flat_samples)) / len(flat_samples)

        plt.figure(figsize=(6, 4))
        sns.histplot(ranks, bins=20, color='coral', kde=False)
        plt.title("Rank Plot for x")
        plt.xlabel("Normalized Rank")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.show()

    initial_params = torch.zeros(ndim)
    for i in range(5):
        run_nuts(model)

if __name__ == "__main__":
    main(5)

