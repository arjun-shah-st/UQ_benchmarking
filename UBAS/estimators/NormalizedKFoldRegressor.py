import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
import concurrent.futures
from copy import deepcopy
from sklearn.model_selection import train_test_split


class NormalizedConformalRes:
    def __init__(self, model, alpha=0.1, res_model=None, random_state=None, cal_size=0.2, res_size=0.2):
        self.model = model
        self.res_model = res_model if res_model is not None else deepcopy(model)
        self.alpha = alpha
        self.random_state = random_state
        self.cal_size = cal_size
        self.res_size = res_size

        self.normalization_values = None
        self.quantile = None

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # For target normalization
        self.y_min = None
        self.y_max = None

    def _check_and_prepare_inputs(self, X, y=None):
        if isinstance(X, (np.ndarray, pd.DataFrame)):
            X = torch.from_numpy(np.asarray(X)).float()
        elif not isinstance(X, torch.Tensor):
            raise TypeError(f"X must be a torch.Tensor or np.ndarray, got {type(X)}")
        if X.dim() == 1:
            X = X.unsqueeze(1)

        if y is not None:
            if isinstance(y, (np.ndarray, pd.Series, pd.DataFrame)):
                y = torch.from_numpy(np.asarray(y)).float()
            elif not isinstance(y, torch.Tensor):
                raise TypeError(f"y must be a torch.Tensor or np.ndarray, got {type(y)}")
            if y.dim() > 1 and y.shape[1] == 1:
                y = y.squeeze(1)
            if len(X) != len(y):
                raise ValueError("X and y must have same length")

        return (X.to(self.device), y.to(self.device)) if y is not None else X.to(self.device)

    def _normalize_targets(self, y):
        self.y_min = y.min()
        self.y_max = y.max()
        return (y - self.y_min) / (self.y_max - self.y_min + 1e-8)

    def _unnormalize_targets(self, y_scaled):
        return y_scaled * (self.y_max - self.y_min + 1e-8) + self.y_min

    def fit(self, X, y, res_net_kwargs, **fit_kwargs):
        X, y = self._check_and_prepare_inputs(X, y)
        y = self._normalize_targets(y)

        X_train, X_cal_full, y_train, y_cal_full = train_test_split(X, y, test_size=self.cal_size,
                                                                    random_state=self.random_state)
        X_cal, X_res, y_cal, y_res = train_test_split(X_cal_full, y_cal_full, test_size=self.res_size,
                                                      random_state=self.random_state)
        model = deepcopy(self.model)
        model.fit(X_train, y_train, **fit_kwargs)
        with torch.no_grad():
            preds = model.predict(X_cal).squeeze()
        residuals = torch.abs(preds - y_cal)
        residual_model = deepcopy(self.res_model)
        with torch.no_grad():
            res_preds = model.predict(X_res).squeeze()
        res_res = torch.abs(res_preds - y_res)
        residual_model.fit(X_res, torch.log(torch.clamp(res_res, min=1e-6)), **res_net_kwargs)
        with torch.no_grad():
            sigma = torch.exp(residual_model.predict(X_cal).squeeze())
            sigma = torch.clamp(sigma, min=1e-6)
        self._res_model = residual_model

        print (f"residuals: {residuals}")
        print(f"sigma: {sigma}")
        scores = residuals / sigma
        n = len(scores)
        q = int((1-self.alpha) * (n + 1)) - 1
        self.quantile = torch.topk(scores, n - q)[0][-1]
        print(scores)
        print(self.quantile)
        self.normalization_values = sigma
        self._model = model

    def predict(self, X, **predict_kwargs):
        X = self._check_and_prepare_inputs(X)
        preds = self._model.predict(X).squeeze()
        sigma = torch.exp(self._res_model.predict(X).squeeze())

        sigma = torch.clamp(sigma, min=1e-6)
        lb = preds - self.quantile * sigma
        ub = preds + self.quantile * sigma

        lb = self._unnormalize_targets(lb)
        ub = self._unnormalize_targets(ub)

        return torch.stack([lb, ub], dim=-1)


class NormalizedConformalEns:
    def __init__(self, model, n_ensemble=5, alpha=0.1, random_state=None, n_jobs=1, cal_size=0.2):
        self.model = model
        self.n_ensemble = n_ensemble
        self.alpha = alpha
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.cal_size = cal_size

        self.models = []
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _train_single_fold(self, X_train, y_train, X_cal, **fit_kwargs):
        X, y = self._check_and_prepare_inputs(X_train, y_train)
        model = deepcopy(self.model)  # .to(self.device)

        model.fit(X, y, **fit_kwargs)
        preds = model.predict(X_cal)

        return model, preds

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
        X_train, X_cal, y_train, y_cal = train_test_split(X, y, test_size=self.cal_size, random_state=self.random_state)

        if self.n_jobs == 1:
            results = [self._train_single_fold(X_train, y_train, X_cal, **fit_kwargs)
                    for i in range(self.n_ensemble)]

        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
                futures = [
                    executor.submit(self._train_single_fold, X_train, y_train, X_cal, **fit_kwargs)
                    for i in range(self.n_ensemble)
                ]
                results = [future.result() for future in concurrent.futures.as_completed(futures)]

        self.models, preds_list = zip(*results)
        preds_list = torch.stack(preds_list, dim=0)
        mean_preds = torch.mean(preds_list, dim=0)
        var_preds = torch.std(preds_list, dim=0)

        residuals = torch.abs(y_cal - mean_preds)
        scores = residuals / var_preds

        n = len(scores)
        q = int((1 - self.alpha) * (n + 1)) - 1
        q = min(max(q, 0), n - 1)
        self.quantile = torch.topk(scores, n - q)[0][-1]

    def predict(self, X, **predict_kwargs):
        X = self._check_and_prepare_inputs(X)
        preds = []
        for model in self.models:
            pred = model.predict(X, **predict_kwargs)
            preds.append(pred.unsqueeze(0))

        preds = torch.cat(preds, dim=0)
        mean_preds = torch.mean(preds, dim=0)
        var_preds = torch.std(preds, dim=0)
        lb = mean_preds - self.quantile * var_preds
        ub = mean_preds + self.quantile * var_preds

        return torch.stack([lb, ub], dim=-1)
