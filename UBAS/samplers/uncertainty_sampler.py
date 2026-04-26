import numpy as np 
from UBAS.samplers.base_sampler import BaseSampler 
from numpy.typing import NDArray 
import pandas as pd 
import matplotlib.pyplot as plt
import torch 
from scipy.stats import norm 
from scipy.optimize import minimize
from torch.quasirandom import SobolEngine

class UncertaintySampler(BaseSampler): 
    def __init__(self, directory, regressor, generator, bounds, n_iterations, n_batch_points,
                 initial_inputs, initial_targets, test_inputs=None, test_targets=None, plotter=None, 
                 save_interval=1, random_seed=42, n_initial_samples=1000, n_to_optimize=5, optimization_steps=10, opt_lr=0.001): 
        
        super().__init__(directory, regressor, generator, bounds, n_iterations, n_batch_points, 
                         initial_inputs, initial_targets, test_inputs, test_targets, plotter, 
                         save_interval, random_seed)
        self.n_initial_samples = n_initial_samples 
        self.n_to_optimize = n_to_optimize 
        self.optimization_steps = optimization_steps 
        self.opt_lr = opt_lr
        
    def sample_step(self): 
        selected_xs = [] 
        i = 0
        dim = self.regressor.input_dim
        while i < self.n_batch_points: 
            print(f"Selecting point: {i+1}/{self.n_batch_points}")
            best_x = self._optimize_acquisition(dim, self.n_initial_samples, self.n_to_optimize, self.optimization_steps, self.opt_lr)

            if best_x is None: 
                print("Warning: Optimization failed: trying again.")
                continue 

            selected_xs.append(best_x)

            print(f"Selected point {i+1}: {best_x}")
            i +=1
        return np.array(selected_xs)
    
    
    def _acquisition_fn(self, x_tensor): 
        x_tensor = x_tensor.unsqueeze(0)
        mean, lower, upper = self.regressor.predict(x_tensor)

        z = norm.ppf(1 - self.regressor.alpha / 2) 
        std = (upper - lower) / (2 * z) * 10 ** 2 
        var = std ** 2
        return var
    
    def _optimize_acquisition(self, dim, n_initial_samples=100000, n_to_optimize=3, n_steps=10, lr=0.001):
        best_score = -float('inf')
        best_x = None
        requires_grad_clone = False 
        if self.regressor.requires_grad: 
            requires_grad_clone = True

        self.regressor.requires_grad = True 

        # Use Sobol for better coverage (or fallback to random)
        sobol = SobolEngine(dimension=dim, scramble=True)
        x0_samples = sobol.draw(n_initial_samples)
        acq_values = []
        for x0 in x0_samples: 
            acq_values.append(self._acquisition_fn(x0))
        acq_values = torch.stack(acq_values)

        top_vals, top_idxs = torch.topk(acq_values, k=n_to_optimize)
        x0_topk = x0_samples[top_idxs]
        
        for x0 in x0_topk:
            x = x0.clone().detach().to(self.regressor.device).requires_grad_(True)

            optimizer = torch.optim.Adam([x], lr=lr)

            for _ in range(n_steps):
                optimizer.zero_grad()
                acq_value = self._acquisition_fn(x)
                loss = -acq_value
                loss.backward(retain_graph=True)
                optimizer.step()
                print(f"optimizer iteration: {_}")
                print(loss.item())

                # Clamp to bounds after step
                with torch.no_grad():
                    x.clamp_(0.0, 1.0)

            # Evaluate final acquisition
            final_score = self._acquisition_fn(x)

            if final_score > best_score:
                best_score = final_score
                best_x = x.detach().cpu().numpy()

        self.regressor.requires_grad = requires_grad_clone
        return best_x