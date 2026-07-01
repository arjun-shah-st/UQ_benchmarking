from uqregressors.conformal.k_fold_cqr import KFoldCQR
from uqregressors.bayesian.deep_ens import DeepEnsembleRegressor
from uqregressors.bayesian.bbmm_gp import BBMM_GP

import sys 
import os 
project_root = os.path.abspath(os.path.join(os.getcwd(), '..'))
if project_root not in sys.path: 
    sys.path.append(project_root)

from UBAS.generators.shift_wing_generator import ShiftWingGenerator
from UBAS.generators.input_generator import InputGenerator 
from UBAS.plotters.base_plotter import BasePlotter 
from UBAS.samplers.base_sampler import BaseSampler 
from UBAS.samplers.badge_sampler import BADGESampler 
from UBAS.samplers.uncertainty_sampler import UncertaintySampler
import gpytorch

import numpy as np

ndim = 9 
init_points = 250 
n_iterations = 10
batch_size = 75
plot_interval = 1 

test_points = 2000 

AR_bounds = np.array([7.5, 11])
Lambda_c_4_bounds = np.array([25, 37.5])
c_r_extension_factor_bounds = np.array([1, 1.4])
D_f_bounds = np.array([6.09, 6.56])
twist_r_bounds = np.array([3, 9])
delta_twist_k_bounds = [-7, -3] 
delta_twist_t_bounds = [-7.5, -1.5]
alpha_bounds = [0, 4]
mach_bounds = [0.75, 0.85]

bounds = np.c_[AR_bounds, Lambda_c_4_bounds, c_r_extension_factor_bounds, D_f_bounds, 
               twist_r_bounds, delta_twist_k_bounds, delta_twist_t_bounds, alpha_bounds, 
               mach_bounds]

input_sampler = InputGenerator(bounds, ndim, seed=42)
test_sampler = InputGenerator(bounds, ndim, seed=1)

cl_gen = ShiftWingGenerator(qoi="CL")
init_inputs, init_targets = cl_gen.generate(input_sampler.uniformly_sample(init_points))
test_inputs, test_targets = cl_gen.generate(test_sampler.uniformly_sample(test_points))

#deep_ens = DeepEnsembleRegressor(n_estimators=5, hidden_sizes=[256, 256], epochs=820, learning_rate=0.00754, batch_size=64)
gp = BBMM_GP(kernel=gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel()) + gpytorch.kernels.ConstantKernel(), epochs=772, learning_rate=0.001045)

plotter = BasePlotter("adaptive_sampling/SHIFTWING_CL/US_GP", plotting_interval=plot_interval)
sampler = UncertaintySampler("adaptive_sampling/SHIFTWING_CL/US_GP", gp, cl_gen, bounds, n_iterations, batch_size, 
                       init_inputs, init_targets, test_inputs, test_targets, plotter=plotter, 
                       random_seed=1064)

sampler.sample(track_values=['rmse'])

## CM 
cm_gen = ShiftWingGenerator(qoi="CM")
init_inputs, init_targets = cm_gen.generate(input_sampler.uniformly_sample(init_points))
test_inputs, test_targets = cm_gen.generate(test_sampler.uniformly_sample(test_points))

#deep_ens = DeepEnsembleRegressor(n_estimators=5, hidden_sizes=[256, 256], epochs=1985, learning_rate=0.00973, batch_size=64)
gp = BBMM_GP(kernel=gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel()) + gpytorch.kernels.ConstantKernel(), epochs=1054, learning_rate=0.001126)

plotter = BasePlotter("adaptive_sampling/SHIFTWING_CM/US_GP", plotting_interval=plot_interval)
sampler = UncertaintySampler("adaptive_sampling/SHIFTWING_CM/US_GP", gp, cm_gen, bounds, n_iterations, batch_size, 
                       init_inputs, init_targets, test_inputs, test_targets, plotter=plotter, 
                       random_seed=1064)

sampler.sample(track_values=['rmse'])

## CD 
cd_gen = ShiftWingGenerator(qoi="CD")
init_inputs, init_targets = cd_gen.generate(input_sampler.uniformly_sample(init_points))
test_inputs, test_targets = cd_gen.generate(test_sampler.uniformly_sample(test_points))

#deep_ens = DeepEnsembleRegressor(n_estimators=5, hidden_sizes=[256, 256], epochs=1825, learning_rate=0.00969, batch_size=64)
gp = BBMM_GP(kernel=gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel()) + gpytorch.kernels.ConstantKernel(), epochs=1381, learning_rate=0.00311)


plotter = BasePlotter("adaptive_sampling/SHIFTWING_CD/US_GP", plotting_interval=plot_interval)
sampler = UncertaintySampler("adaptive_sampling/SHIFTWING_CD/US_GP", gp, cd_gen, bounds, n_iterations, batch_size, 
                       init_inputs, init_targets, test_inputs, test_targets, plotter=plotter, 
                       random_seed=1064)

sampler.sample(track_values=['rmse'])

