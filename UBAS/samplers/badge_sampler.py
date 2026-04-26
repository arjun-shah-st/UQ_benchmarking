import numpy as np 
from UBAS.samplers.base_sampler import BaseSampler 
from numpy.typing import NDArray 
import pandas as pd 
import matplotlib.pyplot as plt
import torch 
from scipy.stats import norm 
from scipy.optimize import minimize
from torch.quasirandom import SobolEngine

class BADGESampler(BaseSampler): 
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
        print(self.regressor.models)
        if hasattr(self.regressor, "n_gradients"): 
            num_models = self.regressor.n_gradients
            print("Using n_gradients attribute. Num_Models: ")
            print(num_models)
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
    
    def _acquisition_fn(self, x_tensor, Q): 
        x_tensor = x_tensor.unsqueeze(0)
        grads_per_model = self.regressor.return_last_layer_grads(x_tensor)
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
        mean, lower, upper = self.regressor.predict(x_tensor)

        z = norm.ppf(1 - self.regressor.alpha / 2) 
        std = (upper - lower) / (2 * z) * 10 ** 2 
        var = std ** 2
        #return var.squeeze()
        #return float(var)#float(avg_residual_norm * var)

        acquisition_value = avg_residual_norm * var
        return acquisition_value
    
    def _optimize_acquisition(self, dim, Q, n_initial_samples=100000, n_to_optimize=3, n_steps=10, lr=0.001):
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
            acq_values.append(self._acquisition_fn(x0, Q))
        acq_values = torch.stack(acq_values)

        top_vals, top_idxs = torch.topk(acq_values, k=n_to_optimize)
        x0_topk = x0_samples[top_idxs]
        
        for x0 in x0_topk:
            x = x0.clone().detach().to(self.regressor.device).requires_grad_(True)

            optimizer = torch.optim.Adam([x], lr=lr)

            for _ in range(n_steps):
                optimizer.zero_grad()
                acq_value = self._acquisition_fn(x, Q)
                loss = -acq_value
                loss.backward(retain_graph=True)
                optimizer.step()
                print(f"optimizer iteration: {_}")
                print(loss.item())

                # Clamp to bounds after step
                with torch.no_grad():
                    x.clamp_(0.0, 1.0)

            # Evaluate final acquisition
            final_score = self._acquisition_fn(x, Q)

            if final_score > best_score:
                best_score = final_score
                best_x = x.detach().cpu().numpy()

        self.regressor.requires_grad = requires_grad_clone
        return best_x