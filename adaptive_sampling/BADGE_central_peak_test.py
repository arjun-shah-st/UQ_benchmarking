from uqregressors.conformal.k_fold_cqr import KFoldCQR 
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
import pandas as pd 
from pandas.plotting import scatter_matrix
from uqregressors.metrics.metrics import compute_all_metrics
from scipy.stats import norm
from scipy.optimize import minimize

ndim = 3
init_points = 250
hidden_dim = 25
candidate_points=500
batch_size = 75
#test_points = 100
test_points=100

Re_bounds = np.array([10 ** 6, 10 ** 8])
alpha_bounds = np.array([0, 15])
e_bounds = np.array([0, 0.08])
p_bounds = np.array([0.2, 0.6])
t_bounds = np.array([0.02, 0.15])

#bounds = np.c_[e_bounds, p_bounds, t_bounds, Re_bounds, alpha_bounds]
bounds = np.ones((2, ndim)) * np.array([[-1, 1]]).T 

input_sampler = InputGenerator(bounds, ndim, seed=42)
test_sampler = InputGenerator(bounds, ndim, seed=1)
init_inputs, init_targets = CentralPeakGenerator(-5).generate(input_sampler.uniformly_sample(init_points))
test_inputs, test_targets = CentralPeakGenerator(-5).generate(test_sampler.uniformly_sample(test_points))

deep_ens = KFoldCQR(n_estimators=2, hidden_sizes=[256, 256], epochs=1500, learning_rate=0.008, batch_size=64) 
#deep_ens = DeepEnsembleRegressor(n_estimators=3, hidden_sizes=[256, 256], epochs=500, learning_rate=0.008, batch_size=64) 

#deep_ens.fit(init_inputs, init_targets)

plotter = BasePlotter("adaptive_sampling/testing/GRUBS_KFOLD_CQR_CENTRAL_PEAK", plotting_interval=1)
sampler = BADGESampler("adaptive_sampling/testing/GRUBS_KFOLD_CQR_CENTRAL_PEAK", deep_ens, CentralPeakGenerator(-5), bounds, 10, 25, init_inputs, init_targets, test_inputs, test_targets, plotter=plotter, random_seed=1089384)

sampler.sample(track_values=["rmse"])

#plotter = BasePlotter("adaptive_sampling/CD_5D", plotting_interval=1)
#sampler = BADGESampler("adaptive_sampling/CD_5D", deep_ens, NFGenerator(qoi="CD"), bounds, 10, 75, init_inputs, init_targets, test_inputs, test_targets, plotter=plotter)

#sampler.sample(track_values=["rmse"])