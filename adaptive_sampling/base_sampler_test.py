from uqregressors.conformal import cqr 
import sys 
import os 
project_root = os.path.abspath(os.path.join(os.getcwd(), '..'))
if project_root not in sys.path: 
    sys.path.append(project_root)

from UBAS.generators.central_peak_generator import CentralPeakGenerator
from UBAS.plotters.base_plotter import BasePlotter
from UBAS.generators.input_generator import InputGenerator 
from UBAS.samplers.base_sampler import BaseSampler
from UBAS.samplers.badge_sampler import BADGESampler
import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np 
import matplotlib.pyplot as plt
from uqregressors.bayesian.deep_ens import DeepEnsembleRegressor
import pandas as pd 
from pandas.plotting import scatter_matrix
from uqregressors.metrics.metrics import compute_all_metrics
from scipy.stats import norm
from scipy.optimize import minimize

ndim = 8
init_points = 250 
hidden_dim = 25
candidate_points=500
batch_size = 25
test_points = 100000

bounds = np.ones((2, ndim)) * np.array([[-1, 1]]).T 
test_bounds = np.ones((2, ndim)) * np.array([[-0.25, 0.25]]).T
input_sampler = InputGenerator(bounds, ndim, seed=42)
test_sampler = InputGenerator(bounds, ndim, seed=42)
init_inputs, init_targets = CentralPeakGenerator(-5).generate(input_sampler.uniformly_sample(init_points))
test_inputs, test_targets = CentralPeakGenerator(-5).generate(test_sampler.uniformly_sample(test_points))

deep_ens = DeepEnsembleRegressor(n_estimators=5, hidden_sizes=[256, 256], epochs=1000, learning_rate=0.001, batch_size=32) 
#deep_ens.fit(init_inputs, init_targets)

plotter = BasePlotter("adaptive_sampling/base_test", plotting_interval=1)
sampler = BaseSampler("adaptive_sampling/base_test", deep_ens, CentralPeakGenerator(-5), bounds, 10, 50, init_inputs, init_targets, test_inputs, test_targets, plotter=plotter)

#sampler.sample()

plotter = BasePlotter("adaptive_sampling/grubs_test/8D_Central_Peak", plotting_interval=1)
sampler = BADGESampler("adaptive_sampling/grubs_test/8D_Central_Peak", deep_ens, CentralPeakGenerator(-5), bounds, 10, 50, init_inputs, init_targets, test_inputs, test_targets, plotter=plotter)

sampler.sample(track_values=["rmse"])