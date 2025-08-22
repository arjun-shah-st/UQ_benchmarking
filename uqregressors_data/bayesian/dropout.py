"""
Monte Carlo Dropout
-------------------

This module implements a Monte Carlo (MC) Dropout Regressor for regression on a one dimensional output
with uncertainty quantification. It estimates
predictive uncertainty by performing multiple stochastic forward passes through a dropout-enabled
neural network.

Key features are: 
    - Customizable neural network architecture 
    - Aleatoric uncertainty included with hyperparameter tau
    - Prediction Intervals based on Gaussian assumption 
    - Customizable optimizer and loss function
    - Optional Input/Output Normalization

!!! warning 
    Using hyperparameter optimization to optimize the aleatoric uncertainty hyperparameter tau is often necessary to obtain correct predictive intervals!
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.base import BaseEstimator, RegressorMixin
from uqregressors.utils.activations import get_activation
from uqregressors.utils.data_loader import validate_and_prepare_inputs, validate_X_input
from uqregressors.utils.torch_sklearn_utils import TorchStandardScaler
from uqregressors.utils.logging import Logger
from pathlib import Path 
import json 
import pickle
import scipy.stats as st


class MLP(nn.Module):
    """
    A simple feedforward neural network with dropout for regression.

    This MLP supports customizable hidden layer sizes, activation functions,
    and dropout. It outputs a single scalar per input — the predictive mean.

    Args:
        input_dim (int): Number of input features.
        hidden_sizes (list of int): Sizes of the hidden layers.
        dropout (float): Dropout rate (applied after each activation).
        activation (callable): Activation function (e.g., nn.ReLU).
    """
    def __init__(self, input_dim, hidden_sizes, dropout, activation):
        super().__init__()
        layers = []
        for h in hidden_sizes:
            layers.append(nn.Linear(input_dim, h))
            layers.append(activation())
            layers.append(nn.Dropout(dropout))
            input_dim = h
        layers.append(nn.Linear(hidden_sizes[-1], 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class MCDropoutRegressor(BaseEstimator, RegressorMixin):
    """ 
    MC Dropout Regressor with uncertainty estimation using neural networks. 

    This class trains a dropout MLP and takes stochastic forward passes to provide predictive uncertainty intervals.
    It makes a Gaussian assumption on the output distribution, and often requires tuning of the hyperparameter tau

    Args:
        name (str): Name of the model instance.
        hidden_sizes (List[int]): Hidden layer sizes for the MLP.
        dropout (float): Dropout rate to apply after each hidden layer.
        tau (float): Precision parameter (used in predictive variance).
        use_paper_weight_decay (bool): Whether to use paper's theoretical weight decay.
        alpha (float): Significance level (1 - confidence level) for prediction intervals.
        requires_grad (bool): Whether to track gradients in prediction output.
        activation_str (str): Activation function name (e.g., "ReLU", "Tanh").
        prior_length_scale (float): Prior length scale for weight decay (1e-2 in paper implementation).
        n_samples (int): Number of stochastic forward passes for prediction.
        learning_rate (float): Learning rate for the optimizer.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        optimizer_cls (Optimizer): PyTorch optimizer class.
        optimizer_kwargs (dict): Optional kwargs to pass to optimizer.
        scheduler_cls (Optional[Callable]): Optional learning rate scheduler class.
        scheduler_kwargs (dict): Optional kwargs for the scheduler.
        loss_fn (Callable): Loss function for training (default: MSE).
        device (str): Device to run training/prediction on ("cpu" or "cuda").
        use_wandb (bool): If True, logs training to Weights & Biases.
        wandb_project (str): W&B project name.
        wandb_run_name (str): W&B run name.
        random_seed (Optional[int]): Seed for reproducibility.
        scale_data (bool): Whether to standardize inputs and outputs.
        input_scaler (Optional[TorchStandardScaler]): Custom input scaler.
        output_scaler (Optional[TorchStandardScaler]): Custom output scaler.
        tuning_loggers (List[Logger]): External loggers returned from hyperparameter tuning.

    Attributes:
        model (MLP): Trained PyTorch MLP model.
        input_dim (int): Dimensionality of input features.
        _loggers (Logger): Training logger.
        training_logs: Logs from training.
        tuning_logs: Logs from hyperparameter tuning.
        fitted (bool): Whether fit has successfully been called.
    """
    def __init__(
        self,
        name="MC_Dropout_Regressor",
        hidden_sizes=[64, 64],
        dropout=0.1,
        tau=1.0e6,
        use_paper_weight_decay=True,
        prior_length_scale=1e-2,
        alpha=0.1,
        requires_grad=False, 
        activation_str="ReLU",
        n_samples=100,
        learning_rate=1e-3,
        epochs=200,
        batch_size=32,
        optimizer_cls=torch.optim.Adam,
        optimizer_kwargs=None,
        scheduler_cls=None,
        scheduler_kwargs=None,
        loss_fn=torch.nn.functional.mse_loss,
        device="cpu",
        use_wandb=False,
        wandb_project=None,
        wandb_run_name=None,
        random_seed=None,
        scale_data=True, 
        input_scaler=None,
        output_scaler=None, 
        tuning_loggers = []
    ):
        self.name=name
        self.hidden_sizes = hidden_sizes
        self.dropout = dropout
        self.tau = tau
        self.use_paper_weight_decay = use_paper_weight_decay
        self.prior_length_scale = prior_length_scale
        self.alpha = alpha
        self.requires_grad = requires_grad
        self.activation_str = activation_str
        self.n_samples = n_samples
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = optimizer_kwargs or {}
        self.scheduler_cls = scheduler_cls
        self.scheduler_kwargs = scheduler_kwargs or {}
        self.loss_fn = loss_fn

        self.device = device
        self.model = None

        self.use_wandb = use_wandb
        self.wandb_project = wandb_project
        self.wandb_run_name = wandb_run_name
        self.random_seed = random_seed
        self.input_dim = None

        self.scale_data = scale_data
        self.input_scaler = input_scaler or TorchStandardScaler()
        self.output_scaler = output_scaler or TorchStandardScaler()

        self._loggers = [] 
        self.training_logs = None
        self.tuning_loggers = tuning_loggers 
        self.tuning_logs = None
        self.fitted = False

    def fit(self, X, y): 
        """
        Fit the MC Dropout model on training data.

        Args:
            X (array-like): Training features of shape (n_samples, n_features).
            y (array-like): Target values of shape (n_samples,).
        Returns: 
            (MCDropoutRegressor): Fitted model.
        """
        X_tensor, y_tensor = validate_and_prepare_inputs(X, y, device=self.device)
        input_dim = X_tensor.shape[1]
        if self.use_paper_weight_decay: 
            l = self.prior_length_scale 
            N = len(X_tensor)
            p = 1 - self.dropout 
            weight_decay = (p * l ** 2) / (2 * N * self.tau)
            self.optimizer_kwargs["weight_decay"] = weight_decay
        self.input_dim = input_dim

        if self.scale_data: 
            X_tensor = self.input_scaler.fit_transform(X_tensor)
            y_tensor = self.output_scaler.fit_transform(y_tensor)

        model, logger = self._fit_single_model(X_tensor, y_tensor)
        self._loggers.append(logger)
        self.fitted = True
        return self 

    def _fit_single_model(self, X_tensor, y_tensor):
        if self.random_seed is not None: 
            torch.manual_seed(self.random_seed)
            np.random.seed(self.random_seed)
        
        config = {
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
        }

        logger = Logger(
            use_wandb=self.use_wandb,
            project_name=self.wandb_project,
            run_name=self.wandb_run_name,
            config=config,
        )

        activation = get_activation(self.activation_str)

        model = MLP(self.input_dim, self.hidden_sizes, self.dropout, activation)
        self.model = model.to(self.device)

        optimizer = self.optimizer_cls(
            self.model.parameters(), lr=self.learning_rate, **self.optimizer_kwargs
        )

        scheduler = None
        if self.scheduler_cls is not None:
            scheduler = self.scheduler_cls(optimizer, **self.scheduler_kwargs)

        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for xb, yb in dataloader: 
                optimizer.zero_grad()
                preds = self.model(xb)
                loss = self.loss_fn(preds, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss

            if scheduler is not None:
                scheduler.step()

            if epoch % (self.epochs / 20) == 0:
                logger.log({"epoch": epoch, "train_loss": epoch_loss})

        logger.finish()

        return self, logger

    def predict(self, X):
        """
        Predicts the target values with uncertainty estimates.

        Args:
            X (np.ndarray): Feature matrix of shape (n_samples, n_features).

        Returns:
            (Union[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]): Tuple containing:
                mean predictions,
                lower bound of the prediction interval,
                upper bound of the prediction interval.
        
        !!! note
            If `requires_grad` is False, all returned arrays are NumPy arrays.
            Otherwise, they are PyTorch tensors with gradients.
        """
        if not self.fitted: 
            raise ValueError("Model not yet fit. Please call fit() before predict().")

        X_tensor = validate_X_input(X, input_dim=self.input_dim, device=self.device, requires_grad=self.requires_grad)
        if self.random_seed is not None: 
            torch.manual_seed(self.random_seed)
            np.random.seed(self.random_seed)
        
        if self.scale_data: 
            X_tensor = self.input_scaler.transform(X_tensor)

        self.model.train()
        preds = []
        with torch.no_grad():
            for _ in range(self.n_samples):
                preds.append(self.model(X_tensor))
        preds = torch.stack(preds)
        mean = preds.mean(dim=0)
        variance = torch.var(preds, dim=0) + 1 / self.tau 
        std = variance.sqrt()
        std_mult = torch.tensor(st.norm.ppf(1 - self.alpha / 2), device=mean.device)
        lower = mean - std * std_mult
        upper = mean + std * std_mult

        if self.scale_data: 
            mean = self.output_scaler.inverse_transform(mean).squeeze()
            lower = self.output_scaler.inverse_transform(lower).squeeze()
            upper = self.output_scaler.inverse_transform(upper).squeeze()
        
        else: 
            mean = mean.squeeze() 
            lower = lower.squeeze() 
            upper = upper.squeeze() 

        if not self.requires_grad: 
            return mean.detach().cpu().numpy(), lower.detach().cpu().numpy(), upper.detach().cpu().numpy()

        else: 
            return mean, lower, upper

    def save(self, path):
        """
        Save model weights, config, and scalers to disk.

        Args:
            path (str or Path): Directory to save model components.
        """
        if not self.fitted: 
            raise ValueError("Model not yet fit. Please call fit() before save().")
        
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save config (exclude non-serializable or large objects)
        config = {
            k: v for k, v in self.__dict__.items()
            if k not in ["optimizer_cls", "optimizer_kwargs", "scheduler_cls", "scheduler_kwargs", "input_scaler", 
                         "output_scaler", "_loggers", "training_logs", "tuning_loggers", "tuning_logs"]
            and not callable(v)
            and not isinstance(v, (torch.nn.Module,))
        }
        
        config["optimizer"] = self.optimizer_cls.__class__.__name__ if self.optimizer_cls is not None else None
        config["scheduler"] = self.scheduler_cls.__class__.__name__ if self.scheduler_cls is not None else None
        config["input_scaler"] = self.input_scaler.__class__.__name__ if self.input_scaler is not None else None 
        config["output_scaler"] = self.output_scaler.__class__.__name__ if self.output_scaler is not None else None

        with open(path / "config.json", "w") as f:
            json.dump(config, f, indent=4)

        with open(path / "extras.pkl", 'wb') as f: 
            pickle.dump([self.optimizer_cls, 
                         self.optimizer_kwargs, self.scheduler_cls, self.scheduler_kwargs, self.input_scaler, self.output_scaler], f)

        # Save model weights
        torch.save(self.model.state_dict(), path / f"model.pt")

        for i, logger in enumerate(getattr(self, "_loggers", [])):
            logger.save_to_file(path, idx=i, name="estimator")

        for i, logger in enumerate(getattr(self, "tuning_loggers", [])): 
            logger.save_to_file(path, name="tuning", idx=i)

    @classmethod
    def load(cls, path, device="cpu", load_logs=False):
        """
        Load a saved MC dropout regressor from disk.

        Args:
            path (str or pathlib.Path): Directory path to load the model from.
            device (str or torch.device): Device to load the model onto.
            load_logs (bool): Whether to load training and tuning logs.

        Returns:
            (MCDropoutRegressor): Loaded model instance.
        """
        path = Path(path)

        # Load config
        with open(path / "config.json", "r") as f:
            config = json.load(f)
        config["device"] = device

        config.pop("optimizer", None)
        config.pop("scheduler", None)
        config.pop("input_scaler", None)
        config.pop("output_scaler", None)
        
        input_dim = config.pop("input_dim", None)
        fitted = config.pop("fitted", False)
        model = cls(**config)

        with open(path / "extras.pkl", 'rb') as f: 
            optimizer_cls, optimizer_kwargs, scheduler_cls, scheduler_kwargs, input_scaler, output_scaler = pickle.load(f)

        # Recreate models
        model.input_dim = input_dim
        activation = get_activation(config["activation_str"])

        model.model = MLP(model.input_dim, config["hidden_sizes"], model.dropout, activation).to(device)
        model.model.load_state_dict(torch.load(path / f"model.pt", map_location=device))

        model.optimizer_cls = optimizer_cls 
        model.optimizer_kwargs = optimizer_kwargs 
        model.scheduler_cls = scheduler_cls 
        model.scheduler_kwargs = scheduler_kwargs
        model.input_scaler = input_scaler 
        model.output_scaler = output_scaler
        model.fitted = fitted

        if load_logs: 
            logs_path = path / "logs"
            training_logs = [] 
            tuning_logs = []
            if logs_path.exists() and logs_path.is_dir(): 
                estimator_log_files = sorted(logs_path.glob("estimator_*.log"))
                for log_file in estimator_log_files:
                    with open(log_file, "r", encoding="utf-8") as f:
                        training_logs.append(f.read())

                tuning_log_files = sorted(logs_path.glob("tuning_*.log"))
                for log_file in tuning_log_files: 
                    with open(log_file, "r", encoding="utf-8") as f: 
                        tuning_logs.append(f.read())

            model.training_logs = training_logs
            model.tuning_logs = tuning_logs
        
        return model
    