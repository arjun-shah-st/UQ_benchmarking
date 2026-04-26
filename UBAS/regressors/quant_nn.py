"""
quant_regressor.py
==================
A simple regressor class which defines a neural net quantile regressor using skorch
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from skorch import NeuralNetRegressor
from skorch.callbacks import ProgressBar, LRScheduler
from sklearn.base import BaseEstimator, RegressorMixin
import pickle
import os
import torch


if torch.cuda.is_available():
    device = torch.device('cuda')
    print('CUDA is available. Using GPU.')
    print('Device name:', torch.cuda.get_device_name(0))
    print('Number of GPUs available:', torch.cuda.device_count())
else:
    device = torch.device('cpu')
    print('CUDA is not available. Using CPU.')

class QuantileLoss(nn.Module):
    """Tilted loss function for quantile regression"""
    def __init__(self, quantiles, device):
        super().__init__()
        self.quantiles = torch.tensor(quantiles, device=device)

    def forward(self, y_pred, y_true):
        error = y_true.view(-1, 1) - y_pred
        return torch.mean(torch.max(self.quantiles * error, (self.quantiles - 1) * error))


class MLP(nn.Module):
    """Constructor class for a basic MLP"""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim)]
        for i in range(num_layers-1):
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Linear(hidden_dim, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, X):
        return self.model(X)


class QuantNN(BaseEstimator, RegressorMixin):
    """A Neural Network which can perform quantile regression"""
    def __init__(self, input_dim, hidden_dim=64, output_dim=1, num_layers=3, quantiles=np.array([0.05, 0.95]), lr=0.01,
                 max_epochs=500, batch_size=32, patience=10):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.quantiles = quantiles
        self.lr = lr
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.patience = patience
        self.device = 'cuda' if torch.cuda.is_available() else'cpu'

        self.net = NeuralNetRegressor(
            module=MLP,
            module__input_dim=self.input_dim,
            module__hidden_dim=self.hidden_dim,
            module__output_dim=self.output_dim,
            module__num_layers=self.num_layers,
            criterion=QuantileLoss,
            criterion__quantiles=self.quantiles,
            criterion__device=self.device,
            optimizer=optim.Adam,
            optimizer__lr=self.lr,
            max_epochs=self.max_epochs,
            batch_size=self.batch_size,
            iterator_train__shuffle=True,
            train_split=None,
            callbacks=[LRScheduler(policy=ReduceLROnPlateau, factor=0.2, min_lr=1e-5, patience=50)],
            verbose=True,
            device=self.device
        )

    def fit(self, X, y):
        """Fit method to conform to scikit-api"""
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float32, device=self.device)
        else:
            X = QuantNN.to_device(X, self.device)
        if isinstance(y, np.ndarray):
            y = torch.tensor(y, dtype=torch.float32, device=self.device)
        else:
            y = QuantNN.to_device(y, self.device)
        self.net.fit(X, y)
        return self

    def predict(self, X):
        """Predict method to conform to scikit-api"""
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float32, device=self.device)
        else:
            X = QuantNN.to_device(X, self.device)
        return self.net.predict(X)

    def save(self, filename):
        """Method to serialize the Neural Network"""
        path = os.path.join(filename)
        os.makedirs(path, exist_ok=True)

        instance_path = os.path.join(path, 'instance.pkl')
        params_path = os.path.join(path, 'model.pkl')
        opt_path = os.path.join(path, 'opt.pkl')
        hist_path = os.path.join(path, 'history.json')

        self.net.save_params(
            f_params=params_path, f_optimizer=opt_path, f_history=hist_path
        )

        instance_dict = {}
        for key, value in self.__dict__.items():
            if key != "net":
                instance_dict[key] = value

        with open(instance_path, 'wb') as f:
            pickle.dump(instance_dict, f)

    @staticmethod
    def load(filename):
        """Method to load serialized Neural Network"""
        path = os.path.join(filename)
        os.makedirs(path, exist_ok=True)

        instance_path = os.path.join(path, 'instance.pkl')
        params_path = os.path.join(path, 'model.pkl')
        opt_path = os.path.join(path, 'opt.pkl')
        hist_path = os.path.join(path, 'history.json')

        with open(instance_path, 'rb') as f:
            instance_dict = pickle.load(f)
        q_nn = QuantNN(**instance_dict)
        q_nn.net.initialize()
        q_nn.net.load_params(
            f_params=params_path, f_optimizer=opt_path, f_history=hist_path
        )
        return q_nn

    @staticmethod
    def to_device(tensor, device):
        if tensor.device != device:
            return tensor.to(device)
        return tensor
