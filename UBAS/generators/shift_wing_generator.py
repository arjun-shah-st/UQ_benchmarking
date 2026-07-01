import numpy as np 
from UBAS.generators.base_generator import BaseGenerator
import pandas as pd 

class ShiftWingGenerator(BaseGenerator): 
    def __init__(self, qoi="CM"): 
        self.qoi = qoi 

        df = pd.read_csv("UBAS/generators/shift_wing_data/shift_wing_cleaned_from_npz.csv")

        self.X = df.iloc[:, 0:8].values 

        if qoi == "CD": 
            self.y = df.iloc[:, 8].values 
        elif qoi == "CF": 
            self.y = df.iloc[:, 9].values 
        elif qoi == "CL": 
            self.y = df.iloc[:, 10].values
        elif qoi == "CM": 
            self.y = df.iloc[:, 11].values
        else: 
            raise ValueError("qoi must be one of ['CM', 'CL', 'CD']")
        
        self.mean = self.X.mean(axis=0, keepdims=True)
        self.std = self.X.std(axis=0, keepdims=True)
        self.X_scaled = (self.X - self.mean) / self.std 

        self.used = np.zeros(len(self.X), dtype=bool)

    def generate(self, x, replace=False, exact_match=False, *args, **kwargs):
        x = np.asarray(x) 
        if x.ndim == 1: 
            x = x[None, :] 

        n_queries = x.shape[0]

        chosen_indices = []

        if exact_match:
            for i in range(n_queries):
                xq = x[i]
                matches = np.all(self.X == xq, axis=1)
                if np.any(matches):
                    idx = np.where(matches)[0][0]
                    if not replace and self.used[idx]:
                        raise ValueError(f"Exact match at index {idx} already used")
                    chosen_indices.append(idx)
                    if not replace:
                        self.used[idx] = True
                else:
                    raise ValueError(f"No exact match found for query {i}: {xq}")
        else:
            Xq_scaled = (x - self.mean) / self.std
            for i in range(n_queries): 
                xq = Xq_scaled[i]

                diffs = self.X_scaled - xq 
                dists = np.einsum("ij, ij->i", diffs, diffs)

                dists[self.used] = np.inf 

                idx = np.argmin(dists)

                chosen_indices.append(idx)

                if replace == False: 
                    self.used[idx] = True 

        chosen_indices = np.array(chosen_indices)

        X_selected = self.X[chosen_indices]
        y_selected = self.y[chosen_indices]

        return X_selected, y_selected

