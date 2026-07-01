from uqregressors.conformal import cqr 
import sys 
import os 
project_root = os.path.abspath(os.path.join(os.getcwd(), '..'))
if project_root not in sys.path: 
    sys.path.append(project_root)

from UBAS.generators.central_peak_generator import CentralPeakGenerator
from UBAS.generators.input_generator import InputGenerator 
import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np 
import matplotlib.pyplot as plt
from uqregressors.bayesian.deep_ens import DeepEnsembleRegressor
from uqregressors.conformal.k_fold_cqr import KFoldCQR
import pandas as pd 
from pandas.plotting import scatter_matrix
from uqregressors.metrics.metrics import compute_all_metrics
from scipy.stats import norm
from scipy.optimize import minimize

def plot_fit(net, X_train, y_train, ndim_to_plot=2, n_random_test_points=50):
    """
    Plots:
    - 1D prediction if input is 1D
    - Otherwise, shows scatter matrix of [X, y_pred, width] for train and random test points
    """
    X_train = np.array(X_train)
    y_train = np.array(y_train).flatten()

    ndim = X_train.shape[1]

    # === 1D Plot ===
    if ndim == 1:
        x_test = np.linspace(-1, 1, 200).reshape(-1, 1)
        y_preds = net.predict(x_test)
        plt.figure()
        if isinstance(y_preds, tuple):
            mean = y_preds[0]
            lower = y_preds[1]
            upper = y_preds[2]
            plt.plot(x_test, mean, label='Predictions')
            plt.plot(x_test, lower, linestyle="--", color="black", label="90% CI")
            plt.plot(x_test, upper, linestyle="--", color="black")
        else:
            plt.plot(x_test, y_preds, label='Predictions')

        plt.scatter(X_train, y_train, label='Training Points', color='blue')
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.title("1D Fit")
        plt.show()
        return

    # === NDIM >= 2: Use scatter matrix ===
    # Create test points for comparison
    rng = np.random.default_rng(44)
    X_test = rng.uniform(-1, 1, size=(n_random_test_points, ndim))
    preds = net.predict(X_test)

    if not isinstance(preds, tuple):
        raise ValueError("Expected model to return (mean, lower, upper) tuple.")

    mean_test, lower_test, upper_test = preds
    width_test = (upper_test - lower_test).flatten()
    mean_test = mean_test.flatten()

    df_test = pd.DataFrame(X_test, columns=[f"x{i}" for i in range(ndim)])
    df_test["y_pred"] = mean_test
    df_test["width"] = width_test
    df_test["source"] = "Test"

    # Training predictions
    train_preds = net.predict(X_train)
    mean_train, lower_train, upper_train = train_preds
    width_train = (upper_train - lower_train).flatten()
    mean_train = mean_train.flatten()

    df_train = pd.DataFrame(X_train, columns=[f"x{i}" for i in range(ndim)])
    df_train["y_pred"] = mean_train
    df_train["width"] = width_train
    df_train["source"] = "Train"

    # Combine
    df_all = pd.concat([df_train, df_test], ignore_index=True)

    # Plot scatter matrix
    color_map = {'Train': 'blue', 'Test': 'red'}
    colors = df_all['source'].map(color_map)

    scatter_matrix(df_all.drop(columns=["source"]), figsize=(12, 12), diagonal='kde', color=colors, alpha=0.8)
    plt.suptitle("Scatter Matrix: Inputs, Predicted y, Widths")
    plt.show()
 

def orthogonal_component_DE(g_x, G):
    Q, _ = torch.linalg.qr(G.T, mode='reduced')  # Q: d x k
    proj = Q @ (Q.T @ g_x)
    residual = g_x - proj
    return residual

def plot_iteration_DE(candidate_X, acquisition_scores, selected_idx, net,
                   train_X, train_y, candidate_X_selected_indices, title=""):
    candidate_X = candidate_X.flatten()

    # Predict NN output over dense grid
    grid = np.linspace(-1, 1, 200).reshape(-1, 1)
    preds, lower, upper = net.predict(grid)

    fig, axs = plt.subplots(1, 2, figsize=(12, 4))

    # Acquisition Function Plot
    axs[0].plot(candidate_X, acquisition_scores, label="Acquisition Score")
    axs[0].scatter(candidate_X[selected_idx], acquisition_scores[selected_idx],
                   color='red', label='Current Selected Point')
    axs[0].set_title("Acquisition Function")
    axs[0].legend()
    axs[0].set_xlabel("Input x")
    axs[0].set_ylabel("Acquisition Score")

    # Neural Net Prediction Plot
    axs[1].plot(grid, preds.reshape(-1, 1), label="NN Prediction", color='blue')
    axs[1].plot(grid, lower.reshape(-1, 1), label="NN_bounds", color='black', linestyle='--')
    axs[1].plot(grid, upper.reshape(-1, 1), color='black', linestyle='--')

    axs[1].scatter(train_X, train_y, color='black', label="Training Points")

    # Previously selected candidate points (excluding current one)
    if candidate_X_selected_indices:
        prev_points = [candidate_X[i] for i in candidate_X_selected_indices[:-1]]
        prev_outputs, _, _ = net.predict((np.array(prev_points)).reshape(-1, 1))
        axs[1].scatter(prev_points, prev_outputs,
                       color='purple', marker='x', label="Previously Selected")

    # Currently selected point
    current_x = candidate_X[selected_idx]
    current_y, _ , _ = net.predict([[current_x]])
    axs[1].scatter([current_x], [current_y], color='red', label="Current Selected")

    axs[1].set_title("Neural Network Prediction")
    axs[1].legend()
    axs[1].set_xlabel("Input x")
    axs[1].set_ylabel("Output y")

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

    from mpl_toolkits.mplot3d import Axes3D

def plot_iteration_2D(candidate_X, acquisition_scores, selected_idx,
                      selected_indices, title="Acquisition Function (2D)"):

    candidate_X = np.array(candidate_X)
    acquisition_scores = np.array(acquisition_scores)

    x1 = candidate_X[:, 0]
    x2 = candidate_X[:, 1]

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(121, projection='3d')

    # 3D scatter of acquisition scores
    ax.scatter(x1, x2, acquisition_scores, c=acquisition_scores, cmap='viridis')
    ax.set_title(title)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("Acquisition Score")

    # 2D view with selected points
    ax2 = fig.add_subplot(122)
    sc = ax2.scatter(x1, x2, c=acquisition_scores, cmap='viridis')
    plt.colorbar(sc, ax=ax2, label="Acquisition Score")

    if selected_indices:
        selected = candidate_X[selected_indices]
        ax2.scatter(selected[:, 0], selected[:, 1], color='red', label="Selected Points")
    ax2.scatter(candidate_X[selected_idx][0], candidate_X[selected_idx][1],
                color='black', marker='x', s=100, label="Current Point")

    ax2.set_title("Selected Candidates")
    ax2.set_xlabel("x1")
    ax2.set_ylabel("x2")
    ax2.legend()
    plt.tight_layout()
    plt.show()

def acquisition_fn(x_tensor, net, G_selected_per_model): 
    x_tensor = x_tensor.detach().clone().requires_grad_(True) 

    grads_per_model = net.return_last_layer_grads(x_tensor.detach().cpu().numpy())

    residual_norms = [] 
    for m, g_x in enumerate(grads_per_model): 
        if len(G_selected_per_model[m]) == 0: 
            residual = g_x  
        
        else: 
            G_mat = torch.stack(G_selected_per_model[m], dim=0)
            residual = orthogonal_component_DE(g_x, G_mat)

        residual_norms.append(residual.norm().item())

    avg_residual_norm = sum(residual_norms) / len(residual_norms)
    mean, lower, upper = net.predict(x_tensor)

    z = norm.ppf(1 - net.alpha / 2) 
    std = (upper - lower) / (2 * z) / net.output_scaler.std_.item() * 10 ** 2
    var = std ** 2

    return float(avg_residual_norm * var)

def optimize_acquisition(net, dim, G_selected_per_model, n_restarts=10): 
    best_score = -np.inf 
    best_x = None 

    bounds = [(-1.0, 1.0)] * dim 

    for _ in range(n_restarts): 
        x0 = np.random.uniform(-1, 1, size=(dim,))
        res = minimize(lambda x: -acquisition_fn(torch.tensor(x, dtype=torch.float32, device=net.device).unsqueeze(0), 
                                                 net, G_selected_per_model), 
                                                 x0, 
                                                 bounds=bounds, 
                                                 method='L-BFGS-B')
        
        if not res.success: 
            continue 

        score = -res.fun 
        if score > best_score: 
            best_score = score 
            best_x = res.x 

    return best_x

def select_batch_DE_optimize(net, candidate_X, train_X, train_y, batch_size): 
    selected_xs = [] 
    num_models = len(net.models)
    G_selected = [[] for _ in range(num_models)]
    dim = net.input_dim

    for i in range(batch_size):
        print(f"Selecting point: {i+1}/{batch_size}")
        best_x = optimize_acquisition(net, dim, G_selected, n_restarts=3)

        if best_x is None:
            print("Warning: Optimization failed, skipping point.")
            continue

        selected_xs.append(best_x)
        grads_sel = net.return_last_layer_grads(np.array(best_x).reshape(1, -1))

        for m in range(num_models):
            G_selected[m].append(grads_sel[m])

        # Optionally plot or print
        print(f"Selected point {i+1}: {best_x}")

    return np.array(selected_xs)

def select_batch_DE(net, candidate_X, train_X, train_y, batch_size):
    device = net.device
    candidate_X = torch.tensor(candidate_X, dtype=torch.float32).to(device)
    N = candidate_X.shape[0]

    selected_indices = []

    num_models = len(net.models)
    G_selected = [[] for _ in range(num_models)]

    
    for iter in range(batch_size):
        print(f"Selecting point:  {iter + 1}")
        acquisition_scores = []

        for idx in range(N):
            if idx in selected_indices:
                acquisition_scores.append(0)
                continue

            x = np.array([candidate_X[idx]]).reshape(1, -1)
            
            grads_per_model = net.return_last_layer_grads(x)

            residual_norms = [] 
            for m, g_x in enumerate(grads_per_model): 
                if len(G_selected[m]) == 0: 
                    residual = g_x  
                
                else: 
                    G_mat = torch.stack(G_selected[m], dim=0)
                    residual = orthogonal_component_DE(g_x, G_mat)

                residual_norms.append(residual.norm().item())

            avg_residual_norm = sum(residual_norms) / num_models 
            mean, lower, upper = net.predict(x)

            z = norm.ppf(1 - net.alpha / 2) 
            std = (upper - lower) / (2 * z) / net.output_scaler.std_.item() * 10 ** 2
            var = std ** 2
            acquisition_scores.append(avg_residual_norm * var)
            #acquisition_scores.append(avg_residual_norm)
        acquisition_scores = np.array(acquisition_scores)
        best_idx = int(np.argmax(acquisition_scores))
        selected_indices.append(best_idx)

        # Update selected gradients
        x_sel = np.array([candidate_X[best_idx]]).reshape(1, -1)
        grads_sel = net.return_last_layer_grads(x_sel)
        for m in range(num_models):
            G_selected[m].append(grads_sel[m])

        # Plotting (show all previously selected indices)

        if candidate_X.shape[1] == 1: 
            plot_iteration_DE(
                candidate_X.cpu().numpy(),
                acquisition_scores,
                best_idx,
                net,
                train_X,
                train_y,
                selected_indices,
                title=f"Iteration {iter + 1}"
            )

        elif candidate_X.shape[1] == 2: 
            plot_iteration_2D(
                candidate_X.cpu().numpy(), 
                acquisition_scores, 
                best_idx, 
                selected_indices, 
                title=f"Iteration {iter + 1}"
            )

    return selected_indices

def run_adaptive_sampling_round_DE_1D(net, inputs, outputs, batch_size): 
    # Suppose you have a trained model `mse_net`
    net.fit(inputs, outputs)
    plot_fit(net, X_train=inputs, y_train=outputs, ndim_to_plot=1)
    
    candidate_X = np.linspace(-1, 1, 100).reshape(-1, 1)  # shape (100, 1)

    #selected_idxs = select_batch_DE(net, candidate_X, inputs, outputs, batch_size=10)
    #selected_points = candidate_X[selected_idxs]
    selected_points = select_batch_DE_optimize(net, candidate_X, inputs, outputs, batch_size=batch_size)

    print("Selected inputs for labeling:", selected_points)
    selected_xs, selected_ys = CentralPeakGenerator(-5).generate(selected_points)
    post_fit_inputs = np.vstack((inputs.reshape(-1, 1), selected_xs.reshape(-1, 1)))
    post_fit_outputs = np.vstack((outputs.reshape(-1, 1), selected_ys.reshape(-1, 1)))

    return net, post_fit_inputs, post_fit_outputs

def run_adaptive_sampling_round_DE_nD(net, inputs, outputs, dim=2, candidate_points=1000, batch_size=10): 
    net.fit(inputs, outputs)
    plot_fit(net, X_train=inputs, y_train=outputs, ndim_to_plot=dim, n_random_test_points=0) 
    print(f"Error Metrics: n_points = {inputs.shape[0]}")
    compute_error_metrics(net, inputs, outputs)

    if dim == 1: 
        candidate_X = np.linspace(-1, 1, 100).reshape(-1, 1)
    else: 
        candidate_X = torch.rand(candidate_points, dim) * 2 - 1

    #selected_idxs = select_batch_DE(net, candidate_X, inputs, outputs, batch_size=batch_size)
    #selected_points = candidate_X[selected_idxs]
    selected_points = select_batch_DE_optimize(net, candidate_X, inputs, outputs, batch_size=batch_size)
    selected_xs, selected_ys = CentralPeakGenerator(-5).generate(selected_points)
    post_fit_inputs = np.vstack((inputs.reshape(-1, dim), selected_xs.reshape(-1, dim)))
    post_fit_outputs = np.vstack((outputs.reshape(-1, 1), selected_ys.reshape(-1, 1)))


    return net, post_fit_inputs, post_fit_outputs, dim, candidate_points, batch_size

def compute_error_metrics(net, inputs, outputs, test_points=500):
    test_xs = torch.rand(test_points, inputs.shape[1]) * 2 - 1 
    test_xs, test_ys = CentralPeakGenerator(-5).generate(test_xs)
    preds = net.predict(test_xs)
    metrics = compute_all_metrics(preds[0], preds[1], preds[2], test_ys, net.alpha)
    for k, v in metrics.items(): 
        print(f"{k}: {v}")

ndim = 1 
init_points = 5 
hidden_dim = 25
candidate_points=500
batch_size = 25

bounds = np.ones((2, ndim)) * np.array([[-1, 1]]).T 
input_sampler = InputGenerator(bounds, ndim, seed=42)
init_inputs, init_targets = CentralPeakGenerator(-5).generate(input_sampler.uniformly_sample(init_points))

deep_ens = DeepEnsembleRegressor(n_estimators=5, hidden_sizes=[100, 100], epochs=500, learning_rate=0.001, batch_size=32) 
#deep_ens.fit(init_inputs, init_targets)

#cqr = KFoldCQR(n_estimators = 2, hidden_sizes=[100, 100], epochs=100, learning_rate=0.001, batch_size=32)

#plot_fit(deep_ens, init_inputs, init_targets)

rounds = 5 
adaptive_sampling_inputs = (deep_ens, init_inputs, init_targets, ndim, candidate_points, batch_size)

for round in range(rounds): 
    adaptive_sampling_inputs = run_adaptive_sampling_round_DE_nD(*(adaptive_sampling_inputs))

