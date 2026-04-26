import numpy as np 
from scipy.stats import qmc 
import torch 
import torch.optim as optim 
import seaborn as sns 
import pandas as pd 
import matplotlib.pyplot as plt 
import pickle
import tqdm 

def fisher_information_linear(X, sigma2):
    return (1 / sigma2) * (X.T @ X)

def d_optimal_value(F): 
    return np.linalg.slogdet(F)[1]

def a_optimal_value(F): 
    return -np.trace(np.linalg.inv(F))

def e_optimal_value(F): 
    return np.linalg.eigvalsh(F)[0]

def evaluate_optimality(F, criterion):
    if criterion == "D":
        return d_optimal_value(F)
    elif criterion == "A":
        return a_optimal_value(F)
    elif criterion == "E":
        return e_optimal_value(F)
    else:
        raise ValueError("criterion must be one of 'D', 'A', or 'E'")
    
def generate_sobol_candidates(d, num_points, bounds, random_seed):
    sampler = qmc.Sobol(d=d, scramble=True, seed=random_seed)
    u = sampler.random(num_points)
    low, high = bounds
    return low + (high - low) * u

def refine_candidate(x0, criterion, X_current, sigma2, bounds, steps=10, lr=0.05, random_seed=42): 
    torch.manual_seed(random_seed)
    x = torch.tensor(x0, dtype=torch.float32, requires_grad=True) 
    low, high = bounds 

    optimizer = optim.Adam([x], lr=lr)

    for _ in range(steps): 
        optimizer.zero_grad() 

        X_aug = torch.tensor(X_current, dtype=torch.float32)
        X_aug = torch.cat([X_aug, x.unsqueeze(0)], dim=0)

        F = (1.0 / sigma2) * X_aug.T @ X_aug 

        #print (F)
        if criterion == "D": 
            sign, logdet = torch.linalg.slogdet(F) 
            score = logdet

        elif criterion == "A": 
            score = -torch.trace(torch.linalg.inv(F))

        elif criterion == "E": 
            eigvals = torch.linalg.eigvalsh(F)
            score = eigvals[0]

        else: 
            raise ValueError("Unknown criterion")
        
        loss = -score + 100 * torch.relu(torch.abs(x) - 1).sum()
        loss.backward() 
        optimizer.step() 

        with torch.no_grad():
            x[:] = torch.clamp(x, low, high)

    return x.detach().numpy()

def select_optimal_design(
        X_init, 
        X_test, 
        y_test, 
        theta, 
        bounds=[-1, 1], 
        criterion="D", 
        sigma_threshold=0.1, 
        d=2, 
        b=10, 
        sobol_factor=5, 
        n_candidates=20,
        refine_steps=10, 
        random_seed=42
): 
    X_current = X_init.copy() 
    gen = np.random.default_rng(seed=random_seed)
    sigma2 = (sigma_threshold * np.linalg.norm(theta)) ** 2 

    num_candidates = d ** sobol_factor 

    for step in tqdm.tqdm(range(b)): 
        X_sobol = generate_sobol_candidates(d, num_candidates, bounds, random_seed) 

        best_score = -np.inf 
        best_x = None 

        for x0 in X_sobol[:n_candidates]: 
            x_ref = refine_candidate(
                x0, criterion, X_current, sigma2, bounds, steps=refine_steps
            )

            X_aug = np.vstack([X_current, x_ref.reshape([-1, d])])
            F = fisher_information_linear(X_aug, sigma2)
            score = evaluate_optimality(F, criterion)

            if score > best_score: 
                best_score = score 
                best_x = x_ref 

        X_current = np.vstack([X_current, best_x])

    epsilon = gen.normal(0, np.sqrt(sigma2), size=(len(X_current), 1))

    y = X_current @ theta + epsilon

    theta_hat = np.linalg.lstsq(X_current, y, rcond=None)[0]

    y_pred = X_test @ theta_hat 
    mse = np.mean((y_pred - y_test) ** 2) 

    return {
        "X_final": X_current,
        "theta_hat": theta_hat,
        "mse_test": mse
    }


def run_all_experiments(
    n_init_factor=3,
    n_test=5000,
    b_factor=9,
    bounds=[-1,1],
    sobol_factor=5,
    refine_steps=10,
    dim_scaling_factor=1, 
    random_seed=42
):
    """
    Runs all 18 experiments (dimension × sigma × criterion) with 5 trials each.
    
    Returns:
        results_df: pandas DataFrame with columns:
            ["dimension", "sigma_threshold", "criterion", "trial", "mse"]
        Also produces a seaborn strip plot.
    """
    rng = np.random.default_rng(random_seed)
    #dims = [2, 6]
    #sigmas = [0.05, 0.1, 0.2]
    #criteria = ["D", "A", "E"]
    dims = [2, 6]
    sigmas = [0.05]
    criteria = ["D", "A", "E"]
    n_trials = 1

    records = []

    for d in dims:
        # Randomly initialize theta and dataset
        theta_true = rng.standard_normal(size = (d, 1))
        for sigma in sigmas:
            for trial in range(n_trials):
                n_init = n_init_factor * d ** dim_scaling_factor
                b = b_factor * d ** dim_scaling_factor

                sigma2 = (sigma * np.linalg.norm(theta_true)) ** 2
                # Initial design points
                X_init = rng.uniform(-1, 1, size=(n_init, d))

                # Test data
                X_test = rng.uniform(-1, 1, size=(n_test, d))
                epsilon_test = rng.normal(0, np.sqrt(sigma2), size=n_test)
                y_test = X_test @ theta_true + epsilon_test
                
                # Random Sampling 
                X_random_add = rng.uniform(-1, 1, size=(b, d))
                X_random = np.vstack([X_init, X_random_add]) 

                epsilon = rng.normal(0, np.sqrt(sigma2), size=(n_init+b, 1))
                y_random = X_random @ theta_true + epsilon 
                
                theta_hat_random = np.linalg.lstsq(X_random, y_random, rcond=None)[0]
                y_pred_random = X_test @ theta_hat_random 
                mse_random = np.mean((y_pred_random - y_test) ** 2)

                records.append({
                    "dimension": d,
                    "sigma_threshold": sigma,
                    "criterion": "Random",
                    "trial": trial,
                    "mse": mse_random})

                for i, crit in enumerate(criteria):
                    # Run your design function
                    result = select_optimal_design(
                        X_init=X_init,
                        X_test=X_test,
                        y_test=y_test,
                        theta=theta_true, 
                        bounds=bounds,
                        criterion=crit,
                        sigma_threshold=sigma,
                        d=d,
                        b=b,
                        sobol_factor=sobol_factor,
                        refine_steps=refine_steps,
                        random_seed=random_seed + trial*5 + d*10
                    )

                    records.append({
                        "dimension": d,
                        "sigma_threshold": sigma,
                        "criterion": crit,
                        "trial": trial,
                        "mse": result["mse_test"]
                    })

                    X = pd.DataFrame(result["X_final"])
                    pd.plotting.scatter_matrix(X)
                    plt.show()

                    print(f"[DONE] d={d}, sigma={sigma}, crit={crit}, trial={trial}, mse={result['mse_test']:.4f}")

    results_df = pd.DataFrame(records)

    return results_df

def plot_results(df_results):
    # Get facet structure
    dims = sorted(df_results["dimension"].unique())
    sigmas = sorted(df_results["sigma_threshold"].unique())
    criteria_order = ["A", "D", "E", "Random"]

    # Create figure with independent axes (sharey=False ensures independence)
    fig, axes = plt.subplots(
        nrows=len(dims),
        ncols=len(sigmas),
        figsize=(4 * len(sigmas), 4 * len(dims)),
        sharey=False
    )

    # If axes becomes 1D array, convert to 2D
    if len(dims) == 1 and len(sigmas) == 1:
        axes = np.array([[axes]])
    elif len(dims) == 1:
        axes = axes[np.newaxis, :]
    elif len(sigmas) == 1:
        axes = axes[:, np.newaxis]

    # Iterate over facets
    for i, d in enumerate(dims):
        for j, sigma in enumerate(sigmas):
            ax = axes[i, j]

            # filter panel data
            df_sub = df_results[
                (df_results["dimension"] == d) &
                (df_results["sigma_threshold"] == sigma)
            ]

            # draw boxplot
            sns.boxplot(
                data=df_sub,
                x="criterion",
                y="mse",
                order=criteria_order,
                palette="Set2",
                ax=ax
            )

            # y-axis logarithmic
            ax.set_yscale("log")

            # axis labels and title
            if i == len(dims) - 1:
                ax.set_xlabel("Optimality Criterion")
            else:
                ax.set_xlabel("")

            if j == 0:
                ax.set_ylabel("Test MSE")
            else:
                ax.set_ylabel("")

            ax.set_title(f"σ = {sigma}, d = {d}")

            # improve spacing
            ax.tick_params(axis="x", rotation=30)

    plt.tight_layout()
    plt.show()

df = run_all_experiments()

#with open("linear_results.pkl", "rb") as f: 
     #df = pickle.load(f)
#with open("linear_results.pkl", 'wb') as f: 
    #pickle.dump(df, f)

#plot_results(df)

