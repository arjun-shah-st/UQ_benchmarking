import numpy as np 
from UBAS.generators.base_generator import BaseGenerator
import pandas as pd 

class ShiftWingGenerator(BaseGenerator): 
    def __init__(self, qoi="CM"): 
        self.qoi = qoi 

        df = pd.read_csv("UBAS/generators/shift_wing_data/shift_wing_cleaned.csv")

        self.X = df.iloc[:, 0:9].values 

        if qoi == "CL": 
            self.y = df.iloc[:, 9].values 
        elif qoi == "CM": 
            self.y = df.iloc[:, 10].values 
        elif qoi == "CD": 
            self.y = df.iloc[:, 11].values
        else: 
            raise ValueError("qoi must be one of ['CM', 'CL', 'CD']")
        
        self.mean = self.X.mean(axis=0, keepdims=True)
        self.std = self.X.std(axis=0, keepdims=True)
        self.X_scaled = (self.X - self.mean) / self.std 

        self.used = np.zeros(len(self.X), dtype=bool)

    def generate(self, x, * args, **kwargs):
        x = np.asarray(x) 
        if x.ndim == 1: 
            x = x[None, :] 

        Xq_scaled = (x - self.mean) / self.std
        n_queries = Xq_scaled.shape[0]

        chosen_indices = []

        for i in range(n_queries): 
            xq = Xq_scaled[i]

            diffs = self.X_scaled - xq 
            dists = np.einsum("ij, ij->i", diffs, diffs)

            dists[self.used] = np.inf 

            idx = np.argmin(dists)

            chosen_indices.append(idx)
            self.used[idx] = True 

        chosen_indices = np.array(chosen_indices)

        X_selected = self.X[chosen_indices]
        y_selected = self.y[chosen_indices]

        return X_selected, y_selected

