import numpy as np
import matplotlib.pyplot as plt
from UBAS.regressors.quant_nn_regressor import QuantNNRegressor
from UBAS.generators.central_peak_generator import CentralPeakGenerator
from UBAS.generators.input_generator import InputGenerator
import pandas as pd
import seaborn as sns
from UBAS.plotters.base_plotter import BasePlotter
import torch
import torch.nn as nn
import torch.optim as optim
from uqregressors.bayesian.deep_ens import DeepEnsembleRegressor
from uqregressors.bayesian.bbmm_gp import BBMM_GP

def analyze_quantiles():
    xs = np.linspace(-1, 1, 50)
    ys = np.exp(-5 * xs ** 2)
    training_xs = np.hstack((np.linspace(-1, -0.5, 10), [0], np.linspace(0.5, 1, 10)))
    training_ys = np.exp(-5 * training_xs ** 2)

    nn = QuantNNRegressor(neurons=100, n_epochs=2000, batch_size=training_xs.shape[0])
    nn.fit(training_xs.reshape(-1, 1), training_ys)
    preds = nn.predict(xs.reshape(-1, 1))
    print(preds)
    lb = preds[:, 0]
    ub = preds[:, 1]
    fig, ax = plt.subplots()
    plt.plot(xs, ys, linestyle='dashed', color='red', label='function')
    plt.scatter(training_xs, training_ys, color='blue', label = 'training_points')
    plt.plot(xs, lb, color='black', label='90% Confidence Bound')
    plt.plot(xs, ub, color='black')
    plt.legend()
    plt.show()
    fig, ax = plt.subplots()
    plt.plot(xs, (ub - lb) ** 5)
    plt.show()
    return nn


def analyze_quantiles_dim(ndim=2, init_points=20, scatter_plot=True):
    bounds = np.ones((2, ndim)) * np.array([[-1, 1]]).T
    input_sampler = InputGenerator(bounds, ndim, seed=42)
    init_inputs, init_targets = CentralPeakGenerator(-5).generate(input_sampler.uniformly_sample(init_points))
    nn = QuantNNRegressor(quantiles=[0.05, 0.95], layers=7, neurons=128, n_epochs=2000, batch_size=int(init_points / 3))
    nn.fit(init_inputs, init_targets)
    if scatter_plot == True:
        X = np.linspace(np.min(bounds), np.max(bounds), 50)
        Y = np.linspace(np.min(bounds), np.max(bounds), 50)
        Xmesh, Ymesh = np.meshgrid(X, Y)
        Z_bounds = nn.predict(np.c_[np.ravel(Xmesh), np.ravel(Ymesh)])
        widths = Z_bounds[:, 1] - Z_bounds[:, 0]
        Wmesh = np.reshape(widths, Xmesh.shape)

        plt.figure(figsize=(12, 12))
        cf = plt.contourf(Xmesh, Ymesh, Wmesh, levels=20)
        plt.colorbar(cf)
        plt.show()
    residuals = torch.absolute(torch.mean(nn.predict(init_inputs), axis=1) - init_targets)
    largest_residual = 2 * torch.max(residuals).to(torch.float32)

    samples = np.random.rand(10000, ndim) * 2 - 1
    bounds = nn.predict(samples)
    widths = bounds[:, 1] - bounds[:, 0]
    max_width_pred = nn.predict(np.zeros(ndim).reshape([1, -1]))
    max_width = max_width_pred[0, 1] - max_width_pred[0, 0]
    print(max_width)
    normalized_widths = widths / largest_residual
    log_prob_distribution = np.exp(normalized_widths)
    df = pd.DataFrame(np.c_[samples, widths, normalized_widths, log_prob_distribution])
    sns.pairplot(pd.DataFrame(np.c_[widths, normalized_widths], columns=["Width", "Normalized_Width"]).sample(5000))
    plt.show()
    return nn, init_inputs, init_targets

def analyze_quantiles_2():
    # Generate sparse data from a deterministic function: y = sin(x)
    x_train = np.array([0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 4.75, 6.75, 7, 7.25, 7.5, 7.75, 8, 8.25, 8.5, 8.75, 10], dtype=np.float32)
    y_train = np.sin(x_train)

    x_train_tensor = torch.tensor(x_train).unsqueeze(1)
    y_train_tensor = torch.tensor(y_train).unsqueeze(1)

    # Define a simple neural network with 3 quantile outputs
    class QuantileNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.shared = nn.Sequential(
                nn.Linear(1, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU()
            )
            self.q10 = nn.Linear(64, 1)
            self.q50 = nn.Linear(64, 1)
            self.q90 = nn.Linear(64, 1)

        def forward(self, x):
            h = self.shared(x)
            return self.q10(h), self.q50(h), self.q90(h)

    # Quantile loss
    def quantile_loss(pred, target, q):
        e = target - pred
        return torch.max(q * e, (q - 1) * e).mean()

    # Initialize and train
    model = QuantileNet()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(2000):
        model.train()
        q10, q50, q90 = model(x_train_tensor)
        loss = (
            quantile_loss(q10, y_train_tensor, 0.1) +
            quantile_loss(q50, y_train_tensor, 0.5) +
            quantile_loss(q90, y_train_tensor, 0.9)
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Predict on test data
    x_test = np.linspace(0, 10, 200, dtype=np.float32)
    x_test_tensor = torch.tensor(x_test).unsqueeze(1)
    model.eval()
    with torch.no_grad():
        q10, q50, q90 = model(x_test_tensor)
        q10, q50, q90 = q10.squeeze().numpy(), q50.squeeze().numpy(), q90.squeeze().numpy()

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(x_test, np.sin(x_test), '--', label='True Function: sin(x)', color='gray')
    plt.scatter(x_train, y_train, color='black', label='Sparse Training Points')
    plt.plot(x_test, q10, label='10th Quantile', color='blue')
    plt.plot(x_test, q50, label='50th Quantile', color='green')
    plt.plot(x_test, q90, label='90th Quantile', color='red')
    plt.fill_between(x_test, q10, q90, color='orange', alpha=0.3, label='Quantile Range')
    plt.title('Quantile Regression on Sparse Deterministic Function')
    plt.xlabel('x'); plt.ylabel('y')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def analyze_conformal_quantiles(ndim=2, init_points=20, scatter_plot=True):
    bounds = np.ones((2, ndim)) * np.array([[-1, 1]]).T
    input_sampler = InputGenerator(bounds, ndim, seed=42)
    init_inputs, init_targets = CentralPeakGenerator(-5).generate(input_sampler.uniformly_sample(init_points))
    init_inputs = torch.tensor(init_inputs, dtype=torch.float32)
    init_targets = torch.tensor(init_targets, dtype=torch.float32)
    #qrnn = QuantNNRegressor(quantiles=[0.05, 0.95], layers=7, neurons=128, n_epochs=2000, batch_size=int(init_points / 3))
    #reg = ConformalKFoldQuantileRegressor(qrnn, n_splits=5, alpha=0.1, n_jobs=5)
    #reg = DeepEnsembleRegressor(n_estimators=3, epochs=500, requires_grad=True)
    reg = BBMM_GP(requires_grad=True)
    reg.fit(init_inputs, init_targets)
    if scatter_plot == True:
        X = np.linspace(np.min(bounds), np.max(bounds), 50)
        Y = np.linspace(np.min(bounds), np.max(bounds), 50)
        Xmesh, Ymesh = np.meshgrid(X, Y)
        Z_bounds = reg.predict(np.c_[np.ravel(Xmesh), np.ravel(Ymesh)])
        widths = Z_bounds[:, 1] - Z_bounds[:, 0]
        Wmesh = np.reshape(widths, Xmesh.shape)

        plt.figure(figsize=(12, 12))
        cf = plt.contourf(Xmesh, Ymesh, Wmesh, levels=20)
        plt.colorbar(cf)
        plt.show()
    mean, lower, upper = reg.predict(init_inputs)
    residuals = torch.absolute(mean - init_targets)
    largest_residual = 2 * torch.max(residuals).to(torch.float32)

    samples = np.random.rand(10000, ndim) * 2 - 1
    mean, lower, upper = reg.predict(samples)
    widths = upper - lower
    max_width_mean, max_width_lower, max_width_upper = reg.predict(np.zeros(ndim).reshape([1, -1]))
    max_width = max_width_upper - max_width_lower
    print(max_width)
    normalized_widths = widths / largest_residual
    log_prob_distribution = torch.exp(normalized_widths)
    #df = pd.DataFrame(np.c_[samples, widths, normalized_widths, log_prob_distribution])
    sns.pairplot(pd.DataFrame(torch.hstack([widths.reshape(-1, 1), normalized_widths.reshape(-1, 1)]).detach(), columns=["Width", "Normalized_Width"]).sample(5000))
    plt.show()
    return reg, init_inputs, init_targets

if __name__ == "__main__":
    analyze_quantiles_2()