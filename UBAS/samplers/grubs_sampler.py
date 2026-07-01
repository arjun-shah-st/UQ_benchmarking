import numpy as np 
from UBAS.samplers.base_sampler import BaseSampler 
from numpy.typing import NDArray 
import pandas as pd 
import matplotlib.pyplot as plt
import torch 
from scipy.optimize import minimize
from torch.quasirandom import SobolEngine
from scipy.stats import norm

class GRUBSSampler(BaseSampler): 
    def __init__(self, directory, regressor, generator, bounds, n_iterations, n_batch_points,
                 initial_inputs, initial_targets, test_inputs=None, test_targets=None, plotter=None, 
                 save_interval=1, random_seed=42, n_initial_samples=1000, n_to_optimize=5, optimization_steps=10, opt_lr=0.001, 
                 mode="GRUBS", verbose=False): 
        
        super().__init__(directory, regressor, generator, bounds, n_iterations, n_batch_points, 
                         initial_inputs, initial_targets, test_inputs, test_targets, plotter, 
                         save_interval, random_seed)
        self.n_initial_samples = n_initial_samples 
        self.n_to_optimize = n_to_optimize 
        self.optimization_steps = optimization_steps 
        self.opt_lr = opt_lr
        self.mode = mode
        self.verbose = verbose
        
    def sample_step(self): 
        selected_xs = [] 
        if hasattr(self.regressor, "n_gradients"): 
            num_models = self.regressor.n_gradients
        elif hasattr(self.regressor, "models"): 
            num_models = len(self.regressor.models)
        elif hasattr(self.regressor, "model"):
            num_models = 1

        else: 
            raise ValueError(f"Regressor of type {self.regressor.__class__} does not have model or models attribute to run GRUBS on")
        
        G_selected = [[] for _ in range(num_models)]
        Q = [None for _ in range(num_models)]
        dim = self.regressor.input_dim 
        i = 0

        while i < self.n_batch_points: 
            print(f"Selecting point: {i+1}/{self.n_batch_points}")
            best_x = self._optimize_acquisition(dim, Q, self.n_initial_samples, self.n_to_optimize, self.optimization_steps, self.opt_lr)

            if best_x is None: 
                print("Warning: Optimization failed: trying again.")
                continue 

            selected_xs.append(best_x)
            grads_sel = self.regressor.return_last_layer_grads(best_x)

            for m in range(num_models): 
                G_selected[m].append(grads_sel[m])
                G_mat = torch.stack(G_selected[m], dim=0)
                Q_new, _ = torch.linalg.qr(G_mat.T, mode='reduced')
                Q[m] = Q_new

            print(f"Selected point {i+1}: {best_x}")
            i +=1
        return np.array(selected_xs)
    
    def _orthogonal_comp(self, g_x, Q): 
        proj = Q @ (Q.T @ g_x)
        residual = g_x - proj 
        return residual 
    
    def _acquisition_fn(self, var, grads_per_model, Q): 
        for g in grads_per_model: 
            g.retain_grad()

        residual_norms = [] 
        for m, g_x in enumerate(grads_per_model): 
            if Q[m] is None: 
                residual = g_x 
            else: 
                residual = self._orthogonal_comp(g_x, Q[m])

            residual_norms.append(residual.norm())

        avg_residual_norm = sum(residual_norms) / len(residual_norms)
        if self.mode == "GRUBS": 
            acquisition_value = avg_residual_norm * var
        elif self.mode == "UNC": 
            acquisition_value = var 
        elif self.mode == "DIV": 
            acquisition_value = avg_residual_norm
        return acquisition_value
    
    def _determine_var_grad(self, x): 
        mean, lower, upper = self.regressor.predict(x)
        z = norm.ppf(1-self.regressor.alpha / 2)
        std = (upper - lower) / (2 * z) 
        var = std ** 2 
        grad_per_models = self.regressor.return_last_layer_grads(x)
        return var, [G.squeeze() for G in grad_per_models]
    
    def _optimize_acquisition(self, dim, Q, n_initial_samples=10000, n_to_optimize=3, n_steps=10, lr=0.001):
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
        means, lowers, uppers = self.regressor.predict(x0_samples)
        z = norm.ppf(1 - self.regressor.alpha / 2)
        stds = (uppers - lowers) / (2 * z)
        vars = stds ** 2
        grad_per_models = self.regressor.return_last_layer_grads(x0_samples)

        for i in range(len(x0_samples)): 
            acq_values.append(self._acquisition_fn(vars[i], [G[i] for G in grad_per_models], Q))
        acq_values = torch.stack(acq_values)

        top_vals, top_idxs = torch.topk(acq_values, k=n_to_optimize)
        x0_topk = x0_samples[top_idxs]
        
        for x0 in x0_topk:
            x = x0.clone().detach().to(self.regressor.device).requires_grad_(True)

            optimizer = torch.optim.Adam([x], lr=lr)

            for _ in range(n_steps):
                optimizer.zero_grad()
                var, grad_per_model = self._determine_var_grad(x.unsqueeze(dim=0))
                acq_value = self._acquisition_fn(var, grad_per_model, Q)
                loss = -acq_value
                loss.backward(retain_graph=True)
                optimizer.step()
                if self.verbose: 
                    print(f"optimizer iteration: {_}")
                    print(loss.item())

                # Clamp to bounds after step
                with torch.no_grad():
                    x.clamp_(0.0, 1.0)

            # Evaluate final acquisition
            var, grad_per_model = self._determine_var_grad(x.unsqueeze(dim=0))
            final_score = self._acquisition_fn(var, grad_per_model, Q)

            if final_score > best_score:
                best_score = final_score
                best_x = x.detach().cpu().numpy()

        self.regressor.requires_grad = requires_grad_clone
        return best_x