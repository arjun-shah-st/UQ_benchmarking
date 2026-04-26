"""
OED implementation adapted to run experiments on the 8-D Borehole function.
This script runs adaptive selection using A-, D-, and E-optimality (Jacobian-based
Fisher computed via autograd) and compares with a random baseline.

Run the script locally (python oed_nn_borehole.py). It prints MSEs and shows
simple diagnostic plots. The code is written for clarity rather than speed.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import grad
import matplotlib.pyplot as plt
from scipy.stats import qmc
from sklearn.preprocessing import StandardScaler
import tqdm
import seaborn as sns
import pandas as pd

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class MLPRegressor(nn.Module):
    def __init__(self, d_in, hidden=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

# ---------------------------------------------------------------------------
# Fisher computations
# ---------------------------------------------------------------------------

def _flatten_grads(grads):
    return torch.cat([g.reshape(-1) for g in grads])


def fisher_matrix_full(model, X, device='cpu', reg=1e-8):
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    F = torch.zeros((n_params, n_params), dtype=torch.float32, device=device)
    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    for x in X_t:
        model.zero_grad()
        y = model(x.unsqueeze(0))
        grads = grad(y.squeeze(), model.parameters(), retain_graph=False, create_graph=False)
        grads = [g if g is not None else torch.zeros_like(p) for g, p in zip(grads, model.parameters())]
        g_flat = _flatten_grads(grads).detach()
        F += torch.ger(g_flat, g_flat)
    F = F + reg * torch.eye(n_params, device=device)
    return F.cpu().numpy()

# ---------------------------------------------------------------------------
# Optimality criteria
# ---------------------------------------------------------------------------

def d_opt_value(F):
    sign, logdet = np.linalg.slogdet(F)
    return logdet

def a_opt_value(F, rcond=1e-4):
    # return -trace(pinv(F))
    Finv = np.linalg.pinv(F, rcond=rcond)
    return -np.trace(Finv)

def e_opt_value(F, eps=1e-6):
    vals = np.linalg.eigvalsh(F)
    return np.min(vals)

# helper to choose
_CRIT_FN = {
    'D': d_opt_value,
    'A': a_opt_value,
    'E': e_opt_value
}

# ---------------------------------------------------------------------------
# Candidate generation and (optional) local refinement
# ---------------------------------------------------------------------------

def generate_candidates(d, n_candidates, bounds, seed=None):
    lb, ub = bounds
    if seed is None:
        rng = np.random.default_rng()
        return rng.uniform(lb, ub, size=(n_candidates, d))
    sampler = qmc.Sobol(d=d, scramble=True, seed=seed)
    u = sampler.random(n_candidates)
    return lb + (ub - lb) * u


def refine_via_local_opt(x0, model, F_base, bounds,
                         criterion='D', steps=12, lr=1e-1,
                         device='cpu', reg=1e-3,
                         grad_weight=True):
    """
    Local refinement of a candidate point via gradient ascent on the OED criterion.
    - grad_weight: if True, multiply acquisition score by ||grad_x f(x)|| to avoid redundant points
    """
    x = torch.tensor(x0.reshape(1, -1), dtype=torch.float32,
                     device=device, requires_grad=True)
    optimizer = optim.Adam([x], lr=lr)
    F_base_t = torch.tensor(F_base, dtype=torch.float32, device=device)

    for _ in range(steps):
        optimizer.zero_grad()

        # Compute gradient wrt parameters
        y = model(x)
        grads = grad(y.squeeze(), model.parameters(), create_graph=True)
        g_flat = torch.cat([g.reshape(-1) for g in grads]).unsqueeze(1)

        # Fisher update
        F = F_base_t + g_flat @ g_flat.T + reg * torch.eye(F_base_t.shape[0], device=device)

        # Compute criterion
        if criterion == 'D':
            sign, logdet = torch.slogdet(F)
            score = logdet
        elif criterion == 'A':
            score = -torch.trace(torch.linalg.pinv(F))
        elif criterion == 'E':
            eigvals = torch.linalg.eigvalsh(F)
            score = eigvals.min()
        else:
            raise ValueError("unknown criterion")

        # Gradient weighting (input sensitivity)
        if grad_weight:
            x.requires_grad_(True)
            y_pred = model(x)
            grad_x = torch.autograd.grad(y_pred.squeeze(), x, retain_graph=True)[0]
            grad_norm = grad_x.norm()
            score = score * grad_norm

        # Loss = negative of score (we do gradient ascent)
        loss = -score

        # Box penalty
        lb_t = torch.tensor(bounds[0], dtype=torch.float32, device=device)
        ub_t = torch.tensor(bounds[1], dtype=torch.float32, device=device)
        penalty = torch.sum(torch.relu(lb_t - x)) + torch.sum(torch.relu(x - ub_t))
        loss = loss + 100.0 * penalty

        # Backprop and step
        loss.backward()
        optimizer.step()

        # Soft projection back into bounds
        with torch.no_grad():
            x.clamp_(lb_t, ub_t)

    return x.detach().cpu().numpy().reshape(-1)


# ---------------------------------------------------------------------------
# Core OED loop (now supports A/D/E)
# ---------------------------------------------------------------------------

def select_design_oed(
    model,
    X_init, y_init,
    X_pool_bounds,
    n_steps=40,
    n_candidates=30,
    criterion='D',
    lambda_reg=1e-8,
    candidate_refine=True,
    refine_steps=8,
    device='cpu',
    retrain_steps_per_add=20,
    lr=1e-3,
    verbose=True,
    grad_weight=False
):
    """
    Online adaptive OED selection with optional gradient weighting.

    Parameters:
    - grad_weight: if True, multiply acquisition score by ||grad_x f(x)|| to avoid redundant points
    """
    d = X_init.shape[1]
    X_cur = X_init.copy()
    y_cur = y_init.copy()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    #   Fit scaler on initial y_init only
    y_scaler = StandardScaler().fit(y_init.reshape(-1, 1))

    # Transform current targets
    y_cur_scaled = y_scaler.transform(y_cur.reshape(-1, 1)).ravel()

    X_t = torch.tensor(X_cur, dtype=torch.float32, device=device)
    y_t = torch.tensor(y_cur_scaled, dtype=torch.float32, device=device)
    model.train()
    for _ in range(1250):
        optimizer.zero_grad()
        pred = model(X_t)
        loss = loss_fn(pred, y_t)
        loss.backward()
        optimizer.step()

    selected = []
    lb, ub = X_pool_bounds
    crit_fn = _CRIT_FN[criterion]

    for it in tqdm.tqdm(range(n_steps)):
        # Compute current Fisher matrix
        F_base = fisher_matrix_full(model, X_cur, device=device, reg=lambda_reg)

        # Generate candidates
        X_cand = generate_candidates(d, n_candidates, (lb, ub), seed=None)
        best_score = -np.inf
        best_x = None
        best_Fadd = None

        for x0 in X_cand:
            # Optional local refinement
            if candidate_refine:
                x_try = refine_via_local_opt(x0, model, F_base, (lb, ub),
                                             criterion=criterion,
                                             steps=refine_steps,
                                             device=device,
                                             reg=lambda_reg)
            else:
                x_try = x0

            # Compute Fisher for candidate
            F_add = fisher_matrix_full(model, x_try.reshape(1, -1), device=device, reg=0.0)
            F_total = (F_base + F_add) / float(len(X_cur) + 1)

            score = crit_fn(F_total)

            # Compute input gradient magnitude weighting
            if grad_weight:
                x_t = torch.tensor(x_try.reshape(1, -1), dtype=torch.float32, device=device, requires_grad=True)
                y_pred = model(x_t)
                grads = torch.autograd.grad(y_pred.squeeze(), x_t, retain_graph=False)[0]
                grad_norm = grads.norm().item()
                score *= grad_norm

            if score > best_score:
                best_score = score
                best_x = x_try
                best_Fadd = F_add

        if best_x is None:
            if verbose:
                print('No candidate found; stopping')
            break

        # Add selected point
        y_new = borehole_function(best_x.reshape(1, -1)).ravel()
        X_cur = np.vstack([X_cur, best_x.reshape(1, -1)])
        y_cur = np.concatenate([y_cur, y_new])

        # Continue training
        model.train()
        # Always transform using the original scaler
        y_cur_scaled = y_scaler.transform(y_cur.reshape(-1, 1)).ravel()

        X_t = torch.tensor(X_cur, dtype=torch.float32, device=device)
        y_t = torch.tensor(y_cur_scaled, dtype=torch.float32, device=device)

        for _ in range(retrain_steps_per_add):
            optimizer.zero_grad()
            pred = model(X_t)
            loss = loss_fn(pred, y_t)
            loss.backward()
            optimizer.step()

        selected.append(best_x.reshape(-1))
        if verbose:
            print(f"Iter {it+1}/{n_steps} ({criterion}): selected point={best_x}, score={best_score:.3e}")

    return np.array(selected), X_cur, y_cur, y_scaler

# ---------------------------------------------------------------------------
# Borehole function
# ---------------------------------------------------------------------------

def borehole_function(X):
    X = np.asarray(X)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    N, d = X.shape
    if d == 8:
        BORE_MIN = np.array([0.05, 100.0, 63070.0, 990.0, 63.1, 700.0, 1120.0, 9855.0])
        BORE_MAX = np.array([0.15, 50000.0, 115600.0, 1110.0, 116.0, 820.0, 1680.0, 12045.0])
        xr = 0.5 * (X + 1.0) * (BORE_MAX - BORE_MIN) + BORE_MIN
        r_w = xr[:, 0]
        r = xr[:, 1]
        T_u = xr[:, 2]
        H_u = xr[:, 3]
        T_l = xr[:, 4]
        H_l = xr[:, 5]
        L = xr[:, 6]
        K_w = xr[:, 7]
        num = 2.0 * np.pi * T_u * (H_u - H_l)
        log_term = np.log(r / r_w)
        den = log_term * (1.0 + (2.0 * L * T_u) / (log_term * r_w ** 2 * K_w) + (T_u / T_l))
        return (num / den).reshape(-1, 1)
    else:
        return (np.sum(np.sin(3 * X) + (X ** 3), axis=1)).reshape(-1, 1)

# ---------------------------------------------------------------------------
# Run experiments on 8-D borehole: A, D, E and Random baseline
# ---------------------------------------------------------------------------

def run_experiments(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)

    d = 8
    lb = -1 * np.ones(d)
    ub = 1 * np.ones(d)

    n_init = 450
    n_steps = 150
    n_candidates = 1000

    # initial and test sets
    X_init = np.random.uniform(lb, ub, size=(n_init, d))
    y_init = borehole_function(X_init).ravel()
    X_test = np.random.uniform(lb, ub, size=(300, d))
    y_test = borehole_function(X_test).ravel()

    crits = ['A', 'D', 'E']
    results = {}
    selected_points = {}

    for crit in crits:
        print(f"Running {crit}-optimal OED...")
        model = MLPRegressor(d_in=d, hidden=16)
        selected, X_final, y_final, y_scaler = select_design_oed(
            model,
            X_init, y_init,
            X_pool_bounds=(lb, ub),
            n_steps=n_steps,
            n_candidates=n_candidates,
            criterion=crit,
            lambda_reg=1e-4,
            candidate_refine=False,
            refine_steps=10,
            device='cpu',
            retrain_steps_per_add=100,
            lr=1e-1,
            verbose=True
        )
        model = MLPRegressor(d_in=d, hidden=16)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-1)
        # Use the same scaler fit on y_init
        y_final_scaled = y_scaler.transform(y_final.reshape(-1, 1)).ravel()
        X_final_t = torch.tensor(X_final, dtype=torch.float32)
        y_final_t = torch.tensor(y_final_scaled, dtype=torch.float32)

        loss_fn = nn.MSELoss()
        for _ in range(2000): 
            optimizer.zero_grad() 
            pred = model(X_final_t)
            loss = loss_fn(pred, y_final_t)
            loss.backward() 
            optimizer.step()    
        # evaluate
        model.eval()
        pred = model(torch.tensor(X_test, dtype=torch.float32)).detach().numpy()
        pred = y_scaler.inverse_transform(pred.reshape(-1, 1)).ravel()
        mse = np.mean((pred - y_test)**2)
        results[crit] = mse
        selected_points[crit] = selected
        print(f"{crit} MSE: {mse:.6f}\n")

    # Random baseline
    print("Running random baseline...")
    model_r = MLPRegressor(d_in=d, hidden=16)
    X_rand_add = np.random.uniform(lb, ub, size=(n_steps, d))
    y_rand_add = borehole_function(X_rand_add).ravel()
    X_r = np.vstack([X_init, X_rand_add])
    y_r = np.concatenate([y_init, y_rand_add])
    y_r_scaled = y_scaler.transform(y_r.reshape(-1,1)).ravel()

    optimizer = torch.optim.Adam(model_r.parameters(), lr=1e-1)
    loss_fn = nn.MSELoss()
    Xr_t = torch.tensor(X_r, dtype=torch.float32)
    yr_t = torch.tensor(y_r_scaled, dtype=torch.float32)
    model_r.train()
    for _ in range(2000):
        optimizer.zero_grad()
        pred = model_r(Xr_t)
        loss = loss_fn(pred, yr_t)
        loss.backward()
        optimizer.step()

    model_r.eval()
    pred = model_r(torch.tensor(X_test, dtype=torch.float32)).detach().numpy()
    pred = y_scaler.inverse_transform(pred.reshape(-1, 1)).ravel()
    mse_rand = np.mean((pred - y_test)**2)
    results['Random'] = mse_rand

    print('\n=== Summary MSEs ===')
    for k, v in results.items():
        print(f"{k}: {v:.6f}")

    # simple plot: bar chart of MSEs
    keys = list(results.keys())
    vals = [results[k] for k in keys]
    #plt.figure(figsize=(6,4))
    #plt.bar(keys, vals)
    #plt.ylabel('Test MSE')
    #plt.title('OED (A/D/E) vs Random on Borehole')
    #plt.show()

    return results, selected_points


def plot_scatter_matrix(selected_points, criterion_name='OED'):
    """
    selected_points: np.array of shape (n_points, d)
    criterion_name: string for labeling
    """
    d = selected_points.shape[1]
    col_names = [f'x{i+1}' for i in range(d)]
    df = pd.DataFrame(selected_points, columns=col_names)
    
    sns.set(style="whitegrid")
    g = sns.pairplot(df, diag_kind='hist', corner=False, palette="tab10")
    plt.show()

import pickle
import matplotlib.pyplot as plt
import numpy as np

def main(n_trials=5, seed_base=0):
    all_results = {c: [] for c in ['A', 'D', 'E', 'Random']}
    all_selected = {c: [] for c in ['A', 'D', 'E', 'Random']}

    for trial in range(n_trials):
        seed = seed_base + 10 * trial
        print(f"\n=== Trial {trial+1}/{n_trials} (seed={seed}) ===")
        results, selected_points = run_experiments(seed=seed)

        # store results
        for crit in ['A', 'D', 'E']:
            all_results[crit].append(results[crit])
            all_selected[crit].append(selected_points[crit])
        all_results['Random'].append(results['Random'])
        # For Random, just store the added random points
        all_selected['Random'].append(selected_points.get('Random', None))

    # Save to pickle
    with open('oed_borehole_results_2.pkl', 'wb') as f:
        pickle.dump({'results': all_results, 'selected_points': all_selected}, f)
    print("\nResults saved to 'oed_borehole_results_2.pkl'")

    # Prepare data for boxplot
    fig, ax = plt.subplots(figsize=(8,5))
    data_to_plot = [all_results[crit] for crit in ['A', 'D', 'E', 'Random']]
    ax.boxplot(data_to_plot, labels=['A', 'D', 'E', 'Random'])
    ax.set_ylabel('Test MSE')
    ax.set_title(f'OED vs Random over {n_trials} trials')
    plt.show()

def plot_results_nonlinear(df_results, n_trials):
    """
    df_results is wide-form:
         A      D      E     Random
    with each row = 1 trial.
    """

    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    sns.set(style="whitegrid")
    plt.rcParams["figure.constrained_layout.use"] = True

    # Convert wide → long for seaborn
    df_long = df_results.melt(
        value_vars=["A", "D", "E", "Random"],
        var_name="criterion",
        value_name="mse"
    )

    # Consistent style with linear plots
    g = sns.catplot(
        data=df_long,
        x="criterion",
        y="mse",
        kind="box",
        height=4,
        aspect=1.2,
        palette="Set2",
        order=["A", "D", "E", "Random"]
    )

    g.set_axis_labels("Optimality Criterion", "Test MSE")
    g.set(yscale="log")
    g.set(ylim=[1, 10])
    for ax in g.axes.flatten():
        ax.spines["right"].set_visible(True)
        ax.spines["top"].set_visible(True) 
    plt.show()


if __name__ == '__main__':
    #main(n_trials=5, seed_base=75)
    with open("oed_borehole_results.pkl", 'rb') as f: 
        results = pickle.load(f)
        df = pd.DataFrame(results["results"])
        print(df.head())
        samples = results["selected_points"]

    plot_scatter_matrix(samples["A"][0])
    plot_scatter_matrix(samples["D"][0])
    plot_scatter_matrix(samples["E"][0])
    plot_results_nonlinear(df, 5)
