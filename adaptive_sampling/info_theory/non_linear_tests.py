import numpy as np 
from scipy.stats import qmc 
import torch
import torch.nn as nn 
import torch.optim as optim 
from sklearn.preprocessing import StandardScaler
import seaborn as sns 
import pandas as pd 
import matplotlib.pyplot as plt 
import pickle 
import tqdm 
from collections import defaultdict
import math

N_EPOCHS = 1500
LAMBDA_REG=1e-3

# Canonical Borehole parameter ranges
BORE_MIN = torch.tensor([
    0.05,      # r_w
    100.0,     # r
    63070.0,   # T_u
    990.0,     # H_u
    63.1,      # T_l
    700.0,     # H_l
    1120.0,    # L
    9855.0     # K_w
], dtype=torch.float32)

BORE_MAX = torch.tensor([
    0.15,      # r_w
    50000.0,   # r
    115600.0,  # T_u
    1110.0,    # H_u
    116.0,     # T_l
    820.0,     # H_l
    1680.0,    # L
    12045.0    # K_w
], dtype=torch.float32)

def _map_to_range(x):
    """Map x from [-1,1] to the canonical Borehole ranges."""
    return 0.5 * (x + 1.0) * (BORE_MAX - BORE_MIN) + BORE_MIN


def f_true_borehole_scaled(x):
    """
    Borehole function assuming input x ∈ [-1, 1]^8.
    x: tensor shape (8,) or (N,8)
    """
    if x.ndim == 1:
        x = x.unsqueeze(0)

    # Map to physical parameter ranges
    xr = _map_to_range(x)

    r_w = xr[:, 0]
    r   = xr[:, 1]
    T_u = xr[:, 2]
    H_u = xr[:, 3]
    T_l = xr[:, 4]
    H_l = xr[:, 5]
    L   = xr[:, 6]
    K_w = xr[:, 7]

    numerator = 2.0 * math.pi * T_u * (H_u - H_l)
    log_term = torch.log(r / r_w)

    denominator = log_term * (
        1.0 +
        (2.0 * L * T_u) / (log_term * r_w**2 * K_w) +
        (T_u / T_l)
    )

    return numerator / denominator

def f_true(X):
    xt = torch.tensor(X, dtype=torch.float32)
    return f_true_borehole_scaled(xt).detach().numpy()

def f_true(x): 
    return np.max(x, axis=1)

class MLPRegressor(nn.Module): 
    def __init__(self, d, hidden=64): 
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, hidden), 
            nn.ReLU(), 
            nn.Linear(hidden, hidden), 
            nn.ReLU(), 
            nn.Linear(hidden, 1)
        )

    def forward(self, x): 
        return self.net(x)
    
def get_penultimate_activations(model, X_scaled_torch):
    """
    X_scaled_torch: torch.tensor shape (n, d), dtype=torch.float32
    returns: np.array shape (n, hidden) of penultimate activations (as floats)
    """
    model.eval()
    with torch.no_grad():
        # forward through all layers except last linear
        # assumes model.net is nn.Sequential([... , nn.Linear(hidden,1)])
        penult = model.net[:-1](X_scaled_torch)   # shape (n, hidden)
        return penult.cpu().numpy()

def fisher_last_layer_from_activations(activations_np, lambda_reg=LAMBDA_REG):
    """
    activations_np: (n, h) numpy array
    returns: F_np shape ((h+1),(h+1))
    """
    n, h = activations_np.shape
    # build g matrix shape (n, h+1)
    ones = np.ones((n, 1), dtype=activations_np.dtype)
    G = np.hstack([activations_np, ones])   # (n, h+1)

    # F = (1/sigma2) * G^T G
    F = (1.0) * (G.T @ G)

    # damping to avoid singularity (use when computing inverses)
    if lambda_reg is not None and lambda_reg > 0:
        F = F + lambda_reg * np.eye(h + 1, dtype=F.dtype)

    return F

# Combined convenience wrapper:
def fisher_last_layer(model, X_scaled, lambda_reg=LAMBDA_REG):
    """
    X_scaled: numpy array shape (n, d) OR torch tensor (n,d)
    returns numpy F (h+1, h+1)
    """
    if isinstance(X_scaled, torch.Tensor):
        X_t = X_scaled
    else:
        X_t = torch.tensor(X_scaled, dtype=torch.float32)

    A = get_penultimate_activations(model, X_t)   # numpy (n, h)
    return fisher_last_layer_from_activations(A, lambda_reg=lambda_reg)

def d_opt_value(F):
    return np.linalg.slogdet(F)[1]

def a_opt_value(F):
    return -np.trace(np.linalg.inv(F))

def e_opt_value(F):
    return np.linalg.eigvalsh(F)[0]

def evaluate_opt(F, crit):
    if crit == "D": return d_opt_value(F)
    if crit == "A": return a_opt_value(F)
    if crit == "E": return e_opt_value(F)
    raise ValueError

def generate_sobol(d, num_points, bounds, seed):
    sampler = qmc.Sobol(d=d, scramble=True, seed=seed)
    u = sampler.random(num_points)
    low, high = bounds
    return low + (high - low) * u

def refine_candidate(x0, model, X_current_scaled, x_scaler, criterion, bounds, steps=10, lr=0.05, lambda_reg=LAMBDA_REG):
    """
    x0: numpy array (d,)
    X_current_scaled: numpy array (n, d) already scaled
    x_scaler: your scaler (so you can transform candidate to scaled coordinates)
    """
    low, high = bounds
    # keep optimization in torch for x, but evaluate F via activations
    x = torch.tensor(x0, dtype=torch.float32, requires_grad=True)
    optimizer = optim.Adam([x], lr=lr)

    # Precompute base activations and F_base
    X_cur_t = torch.tensor(X_current_scaled, dtype=torch.float32)
    F_base = fisher_last_layer(model, X_cur_t, lambda_reg=lambda_reg)

    for _ in range(steps):
        optimizer.zero_grad()

        # candidate in original input coords (unscaled) -> scale -> convert to numpy
        x_np_unscaled = x.detach().cpu().numpy().reshape(1, -1)  # shape (1, d)
        x_scaled = x_scaler.transform(x_np_unscaled)            # (1, d)

        # compute F_add from activations of candidate
        F_add = fisher_last_layer(model, torch.tensor(x_scaled, dtype=torch.float32), lambda_reg=0.0)    # no extra reg here

        F = F_base + F_add
        F_t = torch.tensor(F, dtype=torch.float32)

        # compute score
        if criterion == "D":
            sign, logdet = torch.linalg.slogdet(F_t)
            score = logdet
        elif criterion == "A":
            # for stability, use pinv if necessary
            # convert to numpy for trace of inverse:
            Fi = np.linalg.pinv(F)   # (h+1,h+1)
            score = -np.trace(Fi)
            score = torch.tensor(score, dtype=torch.float32)
        elif criterion == "E":
            eigs = np.linalg.eigvalsh(F)
            score = torch.tensor(eigs[0], dtype=torch.float32)
        else:
            raise ValueError("Unknown criterion")

        low_slack = torch.tensor(x_scaler.transform(np.ones((1, len(x0))) * low), dtype=torch.float32) - x 
        upper_slack = x - torch.tensor(x_scaler.transform(np.ones((1, len(x0))) * high), dtype=torch.float32)

        norm_penalty = torch.relu(low_slack) + torch.relu(upper_slack)
        penalty = 100.0 * norm_penalty.sum()

        loss = -score + penalty
        loss.backward()
        optimizer.step()

        unscaled_x = x_scaler.inverse_transform(x.detach().cpu().numpy().reshape(1, -1))

        with torch.no_grad():
            unscaled_x[:] = np.clip(unscaled_x, low, high)

    return unscaled_x

def select_design_nonlinear(
    X_init,
    X_test,
    y_test,
    bounds,
    criterion,
    d,
    b,
    sobol_factor,
    refine_steps,
    random_seed=42,
    lambda_reg=LAMBDA_REG
): 
    gen = np.random.default_rng(random_seed)

    # scaling
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    x_scaler.fit(X_init)
    y_init_unscaled = f_true(X_init)
    y_scaler.fit(y_init_unscaled.reshape(-1, 1))

    # scaled initial data
    X_current_scaled = x_scaler.transform(X_init)
    y_current_scaled = y_scaler.transform(y_init_unscaled.reshape(-1, 1))

    # build model + train initial mode
    def train_model(Xs, ys, steps=N_EPOCHS):
        model = MLPRegressor(d)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.MSELoss()
        X_t = torch.tensor(Xs, dtype=torch.float32)
        y_t = torch.tensor(ys, dtype=torch.float32)
        for _ in range(steps):
            optimizer.zero_grad()
            pred = model(X_t)
            loss = loss_fn(pred, y_t)
            loss.backward()
            optimizer.step()

        return model

    model = train_model(X_current_scaled, y_current_scaled, steps=N_EPOCHS)
    num_candidates = d ** sobol_factor

    F_base = fisher_last_layer(model, torch.tensor(X_current_scaled, dtype=torch.float32), lambda_reg=lambda_reg)

    # Adaptive acquisition loop
    for step in tqdm.tqdm(range(b)):
        X_sobol = generate_sobol(d, num_candidates, bounds, random_seed)

        best_score = -np.inf
        best_x = None

        for x0 in X_sobol[:20]:  # small subset
            x_ref = refine_candidate(
                x0, model,
                X_current_scaled,
                x_scaler, criterion, bounds,
                steps=refine_steps
            )

            # Evaluate new Fisher
            F_add = fisher_last_layer(model, x_scaler.transform(x_ref.reshape(1, -1)))
            score = evaluate_opt(F_base + F_add, criterion)

            if score > best_score:
                best_F_add = F_add
                best_score = score
                best_x = x_ref

        # Add best point
        y_new_unscaled = f_true(best_x.reshape(1, -1))
        y_new_scaled = y_scaler.transform(y_new_unscaled.reshape(-1, 1))

        X_current_scaled = np.vstack([X_current_scaled, x_scaler.transform(best_x.reshape(1, -1))])
        y_current_scaled = np.vstack([y_current_scaled, y_new_scaled])

        F_base = F_base + best_F_add

        model = train_model(X_current_scaled, y_current_scaled, steps=N_EPOCHS)

    # Final evaluation
    X_test_scaled = x_scaler.transform(X_test)
    X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32)

    y_pred_scaled = model(X_test_t).detach().numpy()
    y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1))

    mse = np.mean((y_pred.ravel() - y_test.ravel())**2)

    print(y_pred, y_test)

    return {
        "mse_test": mse,
        "X":  X_current_scaled, 
        "y": y_current_scaled 
    }


def run_all_experiments_nonlinear(
    n_init_factor=100,
    n_test=5000,
    b_factor=200,
    bounds=[-1, 1],
    sobol_factor=5,
    refine_steps=8,
    random_seed=42
):
    rng = np.random.default_rng(random_seed)
    criteria = ["A", "D", "E"]
    n_trials = 5
    d = 8

    lb, ub = bounds

    records = []
    samples = defaultdict(list)

    for trial in range(n_trials): 
        n_init = n_init_factor 
        b = b_factor 

        X_test = rng.uniform(lb, ub, size=(n_test, d))
        y_test = f_true(X_test)

        X_init = rng.uniform(lb, ub, size=(n_init, d))
        y_init = f_true(X_init)


        X_rand  = rng.uniform(lb, ub, size=(b, d))
        X_comb = np.vstack([X_init, X_rand])
        y_comb = f_true(X_comb)

        x_scal = StandardScaler().fit(X_init)
        y_scal = StandardScaler().fit(y_init.reshape(-1, 1))

        Xs = x_scal.transform(X_comb)
        ys = y_scal.transform(y_comb.reshape(-1, 1))

        model = MLPRegressor(d)
        opt = optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.MSELoss() 

        X_t = torch.tensor(Xs, dtype=torch.float32)
        y_t = torch.tensor(ys, dtype=torch.float32)

        for _ in range(N_EPOCHS):
            opt.zero_grad()
            pred = model(X_t)
            loss = loss_fn(pred, y_t)
            loss.backward()
            opt.step()


        # Evaluate random baseline
        X_test_s = x_scal.transform(X_test)
        X_test_t = torch.tensor(X_test_s, dtype=torch.float32)
        y_pred_s = model(X_test_t).detach().numpy().reshape(-1, 1)
        y_pred = y_scal.inverse_transform(y_pred_s).ravel()

        mse_rand = np.mean((y_pred - y_test)**2)

        records.append({
            "dimension": d,
            "criterion": "Random",
            "trial": trial,
            "mse": mse_rand, 
        })
        samples["Random"].append((X_comb, y_comb))

        for crit in criteria: 
            result = select_design_nonlinear(
                X_init=X_init,
                        X_test=X_test,
                        y_test=y_test,
                        bounds=bounds,
                        criterion=crit,
                        d=d,
                        b=b,
                        sobol_factor=sobol_factor,
                        refine_steps=refine_steps,
                        random_seed=random_seed + trial
            )
            records.append({
            "dimension": d,
            "criterion": crit,
            "trial": trial,
            "mse": result["mse_test"], 
            })
            samples[crit].append((x_scal.inverse_transform(result["X"]), y_scal.inverse_transform(result["y"]).reshape(-1, 1)))
            print(f"[DONE] d={d}, crit={crit}, trial={trial}, mse={result['mse_test']}")

    return pd.DataFrame(records), samples

def plot_results(df_results):
    g = sns.catplot(
        data=df_results,
        x="criterion",
        y="mse",
        kind="box",
        height=4,
        aspect=1.1,
        order=["Random", "A", "D", "E"]
    )

    g.set_axis_labels("Criterion", "Test MSE")
    g.set(yscale="log")
    plt.tight_layout()
    plt.show()

df, samples = run_all_experiments_nonlinear()

#print(samples)

with open("non_linear_results.pkl", 'wb') as f: 
    pickle.dump(df, f)

with open("non_linear_samples.pkl", 'wb') as f: 
    pickle.dump(samples, f)

#plot_results(df)

with open("non_linear_samples.pkl", 'rb') as f: 
    dict = pickle.load(f)


print(dict)
A_points = dict["A"]
X = A_points[0]
print(A_points)