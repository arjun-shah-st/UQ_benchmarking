import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import KFold
import concurrent.futures
import numpy as np
from copy import deepcopy
import pandas as pd

class ConformalKFoldQuantileRegressor:
    def __init__(self, model, n_splits=5, alpha=0.1, random_state=None, n_jobs=1):
        self.model = model
        self.n_splits = n_splits
        self.alpha = alpha
        self.random_state = random_state
        self.n_jobs = n_jobs

        self.models = []
        self.residuals = []
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _train_single_fold(self, train_idx, cal_idx, X, y, **fit_kwargs):
        model = deepcopy(self.model) #.to(self.device)

        X_train, y_train = X[train_idx], y[train_idx]
        X_cal, y_cal = X[cal_idx], y[cal_idx]

        model.fit(X_train, y_train, **fit_kwargs)
        preds = model.predict(X_cal)

        lower_pred, upper_pred = preds[:, 0], preds[:, 1]
        res = torch.max(lower_pred - y_cal, y_cal - upper_pred)

        return model, res

    def _check_and_prepare_inputs(self, X, y=None):
        """
        X: input features
        y: (optional) target values
        Converts numpy to torch, moves to device, checks dimensions.
        """
        if isinstance(X, (np.ndarray, pd.DataFrame)):
            X = torch.from_numpy(np.asarray(X)).float()
        elif not isinstance(X, torch.Tensor):
            raise TypeError(f"X must be a torch.Tensor or np.ndarray, got {type(X)}")

        if y is not None:
            if isinstance(y, (np.ndarray, pd.Series, pd.DataFrame)):
                y = torch.from_numpy(np.asarray(y)).float()
            elif not isinstance(y, torch.Tensor):
                raise TypeError(f"y must be a torch.Tensor or np.ndarray, got {type(y)}")

            if len(X) != len(y):
                raise ValueError(f"X and y must have the same number of samples. Got {len(X)} and {len(y)}.")

        # Force 2D for X
        if X.dim() == 1:
            X = X.unsqueeze(1)

        # Force y to be 1D (targets usually are)
        if y is not None and y.dim() > 1 and y.shape[1] == 1:
            y = y.squeeze(1)

        X = X.to(self.device)
        if y is not None:
            y = y.to(self.device)
            return X, y
        else:
            return X

    def fit(self, X, y, **fit_kwargs):
        X, y = self._check_and_prepare_inputs(X, y)
        kf = KFold(n_splits = self.n_splits, shuffle=True, random_state=self.random_state)
        splits = list(kf.split(X))

        if self.n_jobs == 1:
            results = [self._train_single_fold(train_idx, cal_idx, X, y, **fit_kwargs) for train_idx, cal_idx in splits]

        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
                futures = [
                    executor.submit(self._train_single_fold, train_idx, val_idx, X, y, **fit_kwargs)
                    for train_idx, val_idx in splits
                ]
                results = [future.result() for future in concurrent.futures.as_completed(futures)]

        self.models, residuals_list = zip(*results)
        self.residuals = torch.cat(residuals_list)

        n = len(self.residuals)
        q = int((1 - self.alpha) * (n+1)) - 1
        q = min(max(q, 0), n-1)
        self.quantile = torch.topk(self.residuals, n-q)[0][-1]

    def predict(self, X, **predict_kwargs):
        X = self._check_and_prepare_inputs(X)
        preds = []
        for model in self.models:
            pred = model.predict(X, **predict_kwargs)
            preds.append(pred.unsqueeze(0))

        preds = torch.cat(preds, dim=0)
        lower_preds = preds[:, :, 0]
        upper_preds = preds[:, :, 1]

        lb = torch.mean(lower_preds, dim=0) - self.quantile
        ub = torch.mean(upper_preds, dim=0) + self.quantile

        return torch.stack([lb, ub], dim=-1)


