from uqregressors.conformal.k_fold_cqr import KFoldCQR
import sys 
import os 
project_root = os.path.abspath(os.path.join(os.getcwd(), '..'))
if project_root not in sys.path: 
    sys.path.append(project_root)

from UBAS.generators.neural_foil_generator import NFGenerator
from UBAS.plotters.base_plotter import BasePlotter
from UBAS.generators.input_generator import InputGenerator 
from UBAS.samplers.base_sampler import BaseSampler
from UBAS.samplers.grubs_sampler import GRUBSSampler
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
init_points = 100
hidden_dim = 25
batch_size = 50
n_iterations = 15
#test_points = 100
test_points=5000

Re_bounds = np.array([10 ** 6, 10 ** 8])
alpha_bounds = np.array([0, 15])
e_bounds = np.array([0, 0.08])
p_bounds = np.array([0.2, 0.6])
t_bounds = np.array([0.02, 0.15])

bounds = np.c_[e_bounds, p_bounds, t_bounds, Re_bounds, alpha_bounds]

## CD 
input_sampler = InputGenerator(bounds, ndim, seed=42)
test_sampler = InputGenerator(bounds, ndim, seed=43)
init_inputs, init_targets = NFGenerator(qoi="CD").generate(input_sampler.uniformly_sample(init_points))
test_inputs, test_targets = NFGenerator(qoi="CD").generate(test_sampler.uniformly_sample(test_points))

deep_ens = DeepEnsembleRegressor(n_estimators=5, hidden_sizes=[128, 128], epochs=1500, learning_rate=0.008, batch_size=64, 
                                 scale_data=True,
                                 scheduler_cls=torch.optim.lr_scheduler.CosineAnnealingLR, 
                                 requires_grad=False, 
                                 random_seed=42, 
                                 verbose=True) 
#deep_ens = DeepEnsembleRegressor(n_estimators=3, hidden_sizes=[256, 256], epochs=500, learning_rate=0.008, batch_size=64) 
#k_fold_cqr = KFoldCQR(n_estimators=5, hidden_sizes=[256, 256], epochs=1650, learning_rate=0.0044, batch_size=64)

#deep_ens.fit(init_inputs, init_targets)

plotter = BasePlotter("adaptive_sampling/CD_5D/DE_GRUBS", plotting_interval=1)
sampler = GRUBSSampler("adaptive_sampling/CD_5D/DE_GRUBS", deep_ens, NFGenerator(qoi="CD"), bounds, n_iterations,
                        batch_size, init_inputs, init_targets, test_inputs, test_targets, plotter=plotter, 
                        random_seed=44, verbose=True)

sampler.sample(track_values=["rmse"])

## CD RAND
input_sampler = InputGenerator(bounds, ndim, seed=42)
test_sampler = InputGenerator(bounds, ndim, seed=43)
init_inputs, init_targets = NFGenerator(qoi="CD").generate(input_sampler.uniformly_sample(init_points))
test_inputs, test_targets = NFGenerator(qoi="CD").generate(test_sampler.uniformly_sample(test_points))

deep_ens = DeepEnsembleRegressor(n_estimators=5, hidden_sizes=[128, 128], epochs=1500, learning_rate=0.008, batch_size=64, 
                                 scale_data=True,
                                 scheduler_cls=torch.optim.lr_scheduler.CosineAnnealingLR, 
                                 requires_grad=False, 
                                 random_seed=42, 
                                 verbose=True) 
#deep_ens = DeepEnsembleRegressor(n_estimators=3, hidden_sizes=[256, 256], epochs=500, learning_rate=0.008, batch_size=64) 
#k_fold_cqr = KFoldCQR(n_estimators=5, hidden_sizes=[256, 256], epochs=1650, learning_rate=0.0044, batch_size=64)

#deep_ens.fit(init_inputs, init_targets)

plotter = BasePlotter("adaptive_sampling/CD_5D/RAND", plotting_interval=1)
sampler = BaseSampler("adaptive_sampling/CD_5D/RAND", deep_ens, NFGenerator(qoi="CD"), bounds, n_iterations,
                        batch_size, init_inputs, init_targets, test_inputs, test_targets, plotter=plotter, 
                        random_seed=44)

sampler.sample(track_values=["rmse"])

## CL_GRUBS 
input_sampler = InputGenerator(bounds, ndim, seed=42)
test_sampler = InputGenerator(bounds, ndim, seed=43)
init_inputs, init_targets = NFGenerator(qoi="CL").generate(input_sampler.uniformly_sample(init_points))
test_inputs, test_targets = NFGenerator(qoi="CL").generate(test_sampler.uniformly_sample(test_points))

deep_ens = DeepEnsembleRegressor(n_estimators=5, hidden_sizes=[128, 128], epochs=750, learning_rate=0.003, batch_size=64, 
                                 scale_data=True,
                                 scheduler_cls=torch.optim.lr_scheduler.CosineAnnealingLR, 
                                 requires_grad=False, 
                                 random_seed=42, 
                                 verbose=True) 

plotter = BasePlotter("adaptive_sampling/CL_5D/DE_GRUBS", plotting_interval=1)
sampler = GRUBSSampler("adaptive_sampling/CL_5D/DE_GRUBS", deep_ens, NFGenerator(qoi="CL"), bounds, n_iterations,
                        batch_size, init_inputs, init_targets, test_inputs, test_targets, plotter=plotter, 
                        random_seed=44, verbose=True)

sampler.sample(track_values=["rmse"])

## CL_RANDOM 
input_sampler = InputGenerator(bounds, ndim, seed=42)
test_sampler = InputGenerator(bounds, ndim, seed=43)
init_inputs, init_targets = NFGenerator(qoi="CL").generate(input_sampler.uniformly_sample(init_points))
test_inputs, test_targets = NFGenerator(qoi="CL").generate(test_sampler.uniformly_sample(test_points))

deep_ens = DeepEnsembleRegressor(n_estimators=5, hidden_sizes=[128, 128], epochs=750, learning_rate=0.003, batch_size=64, 
                                 scale_data=True,
                                 scheduler_cls=torch.optim.lr_scheduler.CosineAnnealingLR, 
                                 requires_grad=False, 
                                 random_seed=42, 
                                 verbose=True) 

plotter = BasePlotter("adaptive_sampling/CL_5D/RAND", plotting_interval=1)
sampler = BaseSampler("adaptive_sampling/CL_5D/RAND", deep_ens, NFGenerator(qoi="CL"), bounds, n_iterations,
                        batch_size, init_inputs, init_targets, test_inputs, test_targets, plotter=plotter, 
                        random_seed=44)

sampler.sample(track_values=["rmse"])

## CM_GRUBS 
input_sampler = InputGenerator(bounds, ndim, seed=42)
test_sampler = InputGenerator(bounds, ndim, seed=43)
init_inputs, init_targets = NFGenerator(qoi="CM").generate(input_sampler.uniformly_sample(init_points))
test_inputs, test_targets = NFGenerator(qoi="CM").generate(test_sampler.uniformly_sample(test_points))

deep_ens = DeepEnsembleRegressor(n_estimators=5, hidden_sizes=[128, 128], epochs=750, learning_rate=0.003, batch_size=64, 
                                 scale_data=True,
                                 scheduler_cls=torch.optim.lr_scheduler.CosineAnnealingLR, 
                                 requires_grad=False, 
                                 random_seed=42, 
                                 verbose=True) 

plotter = BasePlotter("adaptive_sampling/CM_5D/DE_GRUBS", plotting_interval=1)
sampler = GRUBSSampler("adaptive_sampling/CM_5D/DE_GRUBS", deep_ens, NFGenerator(qoi="CM"), bounds, n_iterations,
                        batch_size, init_inputs, init_targets, test_inputs, test_targets, plotter=plotter, 
                        random_seed=44, verbose=True)

sampler.sample(track_values=["rmse"])

## CL_RANDOM 
input_sampler = InputGenerator(bounds, ndim, seed=42)
test_sampler = InputGenerator(bounds, ndim, seed=43)
init_inputs, init_targets = NFGenerator(qoi="CM").generate(input_sampler.uniformly_sample(init_points))
test_inputs, test_targets = NFGenerator(qoi="CM").generate(test_sampler.uniformly_sample(test_points))

deep_ens = DeepEnsembleRegressor(n_estimators=5, hidden_sizes=[128, 128], epochs=1489, learning_rate=0.0064, batch_size=64, 
                                 scale_data=True,
                                 scheduler_cls=torch.optim.lr_scheduler.CosineAnnealingLR, 
                                 requires_grad=False, 
                                 random_seed=42, 
                                 verbose=True) 

plotter = BasePlotter("adaptive_sampling/CM_5D/RAND", plotting_interval=1)
sampler = BaseSampler("adaptive_sampling/CM_5D/RAND", deep_ens, NFGenerator(qoi="CM"), bounds, n_iterations,
                        batch_size, init_inputs, init_targets, test_inputs, test_targets, plotter=plotter, 
                        random_seed=44)

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