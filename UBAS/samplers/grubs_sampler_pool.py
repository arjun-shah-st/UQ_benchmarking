import numpy as np
from UBAS.samplers.grubs_sampler import GRUBSSampler
import torch
from torch.quasirandom import SobolEngine
from scipy.stats import norm

class GRUBSSamplerPool(GRUBSSampler):
    def __init__(self, directory, regressor, generator, bounds, n_iterations, n_batch_points,
                 initial_inputs, initial_targets, candidate_x_values, test_inputs=None, test_targets=None, plotter=None,
                 save_interval=1, random_seed=42, n_initial_samples=1000, n_to_optimize=5, optimization_steps=10, opt_lr=0.001,
                 mode="GRUBS", verbose=False):

        super().__init__(directory, regressor, generator, bounds, n_iterations, n_batch_points,
                         initial_inputs, initial_targets, test_inputs, test_targets, plotter,
                         save_interval, random_seed, n_initial_samples, n_to_optimize, optimization_steps, opt_lr,
                         mode, verbose)
        
        self.candidate_x_values = candidate_x_values  # In original space
        self.candidate_pool = None  # Will be set to scaled candidates when first used

    def _optimize_acquisition(self, dim, Q, n_initial_samples=10000, n_to_optimize=3, n_steps=10, lr=0.001):
        """
        Instead of optimizing, evaluate acquisition on candidate set and pick the best.
        Removes the chosen point from the candidate pool.
        """
        best_score = -float('inf')
        best_x = None
        requires_grad_clone = False
        if self.regressor.requires_grad:
            requires_grad_clone = True

        self.regressor.requires_grad = True

        # Initialize candidate pool if not done
        if self.candidate_pool is None:
            self.candidate_pool = self.input_scaler.transform(self.candidate_x_values)

        # Use current candidate pool
        candidate_samples = torch.tensor(self.candidate_pool, dtype=torch.float32)

        # Evaluate acquisition on all candidates
        acq_values = []
        means, lowers, uppers = self.regressor.predict(candidate_samples)
        z = norm.ppf(1 - self.regressor.alpha / 2)
        stds = (uppers - lowers) / (2 * z)
        vars = stds ** 2
        grad_per_models = self.regressor.return_last_layer_grads(candidate_samples)

        for i in range(len(candidate_samples)):
            acq_values.append(self._acquisition_fn(vars[i], [G[i] for G in grad_per_models], Q))
        acq_values = torch.stack(acq_values)

        # Find the best candidate
        best_idx = torch.argmax(acq_values)
        best_x = candidate_samples[best_idx].detach().cpu().numpy()

        # Remove the chosen point from the candidate pool
        self.candidate_pool = np.delete(self.candidate_pool, best_idx, axis=0)

        self.regressor.requires_grad = requires_grad_clone
        return best_x
