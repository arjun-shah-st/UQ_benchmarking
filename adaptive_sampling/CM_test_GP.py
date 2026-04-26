from uqregressors.bayesian.bbmm_gp import BBMM_GP
import sys 
import os 
project_root = os.path.abspath(os.path.join(os.getcwd(), '..'))
if project_root not in sys.path: 
    sys.path.append(project_root)

import gpytorch
from UBAS.generators.neural_foil_generator import NFGenerator
from UBAS.plotters.base_plotter import BasePlotter
from UBAS.generators.input_generator import InputGenerator 
from UBAS.samplers.base_sampler import BaseSampler
from UBAS.samplers.uncertainty_sampler import UncertaintySampler
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

ndim = 5
init_points = 250
hidden_dim = 25
candidate_points=500
batch_size = 25
#test_points = 100
test_points=10000

Re_bounds = np.array([10 ** 6, 10 ** 8])
alpha_bounds = np.array([0, 15])
e_bounds = np.array([0, 0.08])
p_bounds = np.array([0.2, 0.6])
t_bounds = np.array([0.02, 0.15])

bounds = np.c_[e_bounds, p_bounds, t_bounds, Re_bounds, alpha_bounds]

input_sampler = InputGenerator(bounds, ndim, seed=42)
test_sampler = InputGenerator(bounds, ndim, seed=1)

init_inputs, init_targets = NFGenerator(qoi="CL").generate(input_sampler.uniformly_sample(init_points))
test_inputs, test_targets = NFGenerator(qoi="CL").generate(test_sampler.uniformly_sample(test_points))
bbmm_gpr = BBMM_GP(
    kernel=gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel()) + gpytorch.kernels.ConstantKernel(), epochs=1450, learning_rate=0.0088, 
)

plotter = BasePlotter("adaptive_sampling/CL_5D/US_GP", plotting_interval=75)
sampler = UncertaintySampler("adaptive_sampling/CL_5D/US_GP", bbmm_gpr, NFGenerator(qoi="CL"), bounds, 750, 1, init_inputs, init_targets, test_inputs, test_targets, plotter=plotter, random_seed=1089384)

sampler.sample(track_values=["rmse"])

init_inputs, init_targets = NFGenerator(qoi="CM").generate(input_sampler.uniformly_sample(init_points))
test_inputs, test_targets = NFGenerator(qoi="CM").generate(test_sampler.uniformly_sample(test_points))
bbmm_gpr = BBMM_GP(
    kernel=gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel()) + gpytorch.kernels.ConstantKernel(), epochs=1230, learning_rate=0.00835, 
)

plotter = BasePlotter("adaptive_sampling/CM_5D/US_GP", plotting_interval=75)
sampler = UncertaintySampler("adaptive_sampling/CM_5D/US_GP", bbmm_gpr, NFGenerator(qoi="CM"), bounds, 750, 1, init_inputs, init_targets, test_inputs, test_targets, plotter=plotter, random_seed=1089384)

sampler.sample(track_values=["rmse"])

init_inputs, init_targets = NFGenerator(qoi="CD").generate(input_sampler.uniformly_sample(init_points))
test_inputs, test_targets = NFGenerator(qoi="CD").generate(test_sampler.uniformly_sample(test_points))

bbmm_gpr = BBMM_GP(
    kernel=gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel()) + gpytorch.kernels.ConstantKernel(), epochs=1450, learning_rate=0.0075, 
)

plotter = BasePlotter("adaptive_sampling/CD_5D/US_GP", plotting_interval=75)
sampler = UncertaintySampler("adaptive_sampling/CD_5D/US_GP", bbmm_gpr, NFGenerator(qoi="CD"), bounds, 750, 1, init_inputs, init_targets, test_inputs, test_targets, plotter=plotter, random_seed=1089384)

sampler.sample(track_values=["rmse"])

#plotter = BasePlotter("adaptive_sampling/CD_5D", plotting_interval=1)
#sampler = BADGESampler("adaptive_sampling/CD_5D", deep_ens, NFGenerator(qoi="CD"), bounds, 10, 75, init_inputs, init_targets, test_inputs, test_targets, plotter=plotter)

#sampler.sample(track_values=["rmse"])

#init_inputs, init_targets = NFGenerator(qoi="CL").generate(input_sampler.uniformly_sample(init_points))
#test_inputs, test_targets = NFGenerator(qoi="CL").generate(test_sampler.uniformly_sample(test_points))

#plotter = BasePlotter("adaptive_sampling/CL_5D/RAND/p_775", plotting_interval=1)
#sampler = BaseSampler("adaptive_sampling/CL_5D/RAND/p_775", deep_ens, NFGenerator(qoi="CL"), bounds, 10, 75, init_inputs, init_targets, test_inputs, test_targets, plotter=plotter, random_seed=4208343)

#sampler.sample(track_values=["rmse"])

#plotter = BasePlotter("adaptive_sampling/CL_5D", plotting_interval=1)
#sampler = BADGESampler("adaptive_sampling/CL_5D", deep_ens, NFGenerator(qoi="CL"), bounds, 10, 75, init_inputs, init_targets, test_inputs, test_targets, plotter=plotter)

#sampler.sample(track_values=["rmse"])