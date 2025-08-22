"""
Deep Ensembles
--------------

This module implements Deep Ensemble Regressors for regression of a one dimensional output. 

Key features are: 
    - Customizable neural network architecture 
    - Prediction Intervals based on Gaussian assumption 
    - Parallel training of ensemble members with Joblib
    - Customizable optimizer and loss function
    - Optional Input/Output Normalization
"""

import numpy as np
import torch
import torch.nn as nn
import scipy.stats as st
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, RegressorMixin 
from uqregressors.utils.activations import get_activation
from uqregressors.utils.logging import Logger
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path
import json
import pickle
import torch.multiprocessing as mp 
import torch.nn.functional as F
from uqregressors.utils.torch_sklearn_utils import TorchStandardScaler
from uqregressors.utils.data_loader import validate_and_prepare_inputs, validate_X_input

mp.set_start_method('spawn', force=True)

class MLP(nn.Module): 
    """
    A simple multi-layer perceptron which outputs a mean and a positive variance per input sample.
    
    Args:
        input_dim (int): Number of input features.
        hidden_sizes (list of int): List of hidden layer sizes.
        activation (torch.nn.Module): Activation function class (e.g., nn.ReLU).
    """
    def __init__(self, input_dim, hidden_sizes, activation):
        super().__init__()
        layers = []
        for h in hidden_sizes:
            layers.append(nn.Linear(input_dim, h))
            layers.append(activation())
            input_dim = h
        output_layer = nn.Linear(hidden_sizes[-1], 2)
        layers.append(output_layer)
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        outputs = self.model(x)
        means = outputs[:, 0]
        unscaled_variances = outputs[:, 1]
        scaled_variance = F.softplus(unscaled_variances) + 1e-6
        scaled_outputs = torch.cat((means.unsqueeze(dim=1), scaled_variance.unsqueeze(dim=1)), dim=1)

        return scaled_outputs
    
class DeepEnsembleRegressor(BaseEstimator, RegressorMixin): 
    """
    Deep Ensemble Regressor with uncertainty estimation using neural networks.

    Trains an ensemble of MLP models to predict both mean and variance for regression tasks,
    and provides predictive uncertainty intervals.

    Args:
        name (str): Name of the regressor for config files.
        n_estimators (int): Number of ensemble members.
        hidden_sizes (list of int): List of hidden layer sizes for each MLP.
        alpha (float): Significance level for prediction intervals (e.g., 0.1 for 90% interval).
        requires_grad (bool): If True, returned predictions require gradients.
        activation_str (str): Name of activation function to use (e.g., 'ReLU').
        learning_rate (float): Learning rate for optimizer.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        optimizer_cls (torch.optim.Optimizer): Optimizer class.
        optimizer_kwargs (dict): Additional kwargs for optimizer.
        scheduler_cls (torch.optim.lr_scheduler._LRScheduler or None): Learning rate scheduler class.
        scheduler_kwargs (dict): Additional kwargs for scheduler.
        loss_fn (callable): Loss function accepting (preds, targets).
        device (str or torch.device): Device to run training on ('cpu' or 'cuda').
        use_wandb (bool): Whether to use Weights & Biases logging.
        wandb_project (str or None): WandB project name.
        wandb_run_name (str or None): WandB run name prefix.
        n_jobs (int): Number of parallel jobs to train ensemble members.
        random_seed (int or None): Seed for reproducibility.
        scale_data (bool): Whether to scale input and output data.
        input_scaler (object or None): Scaler for input features.
        output_scaler (object or None): Scaler for target values.
        tuning_loggers (list): List of tuning loggers.

    Attributes:
        models (list): List of trained PyTorch MLP models.
        input_dim (int): Dimensionality of input features.
        _loggers (list): Training loggers for each model.
        training_logs: Logs from training.
        tuning_logs: Logs from hyperparameter tuning.
        fitted (bool): Whether fit has been successfully called
    """
    def __init__(
        self,
        name = "Deep_Ensemble_Regressor",
        n_estimators=5,
        hidden_sizes=[64, 64],
        alpha=0.1,
        requires_grad=False,
        activation_str="ReLU",
        learning_rate=1e-3,
        epochs=200,
        batch_size=32,
        optimizer_cls=torch.optim.Adam,
        optimizer_kwargs=None,
        scheduler_cls=None,
        scheduler_kwargs=None,
        loss_fn=None,
        device="cpu",
        use_wandb=False,
        wandb_project=None,
        wandb_run_name=None,
        n_jobs=1,
        random_seed=None,
        scale_data=True, 
        input_scaler=None,
        output_scaler=None, 
        tuning_loggers = [],
    ):
        self.name=name
        self.n_estimators = n_estimators
        self.hidden_sizes = hidden_sizes
        self.alpha = alpha
        self.requires_grad = requires_grad
        self.activation_str = activation_str
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = optimizer_kwargs or {}
        self.scheduler_cls = scheduler_cls
        self.scheduler_kwargs = scheduler_kwargs or {}
        self.loss_fn = loss_fn or self.nll_loss
        self.device = device

        self.use_wandb = use_wandb
        self.wandb_project = wandb_project
        self.wandb_run_name = wandb_run_name

        self.n_jobs = n_jobs
        self.random_seed = random_seed
        self.models = []
        self.input_dim = None

        self.scale_data = scale_data

        if scale_data: 
            self.input_scaler = input_scaler or TorchStandardScaler()
            self.output_scaler = output_scaler or TorchStandardScaler()

        self._loggers = []
        self.training_logs = None
        self.tuning_loggers = tuning_loggers
        self.tuning_logs = None
        self.fitted = False

    def nll_loss(self, preds, y): 
        """
        Negative log-likelihood loss assuming Gaussian outputs.

        Args:
            preds (torch.Tensor): Predicted means and variances, shape (batch_size, 2).
            y (torch.Tensor): True target values, shape (batch_size,).

        Returns:
            (torch.Tensor): Scalar loss value.
        """
        means = preds[:, 0]
        variances = preds[:, 1]
        precision = 1 / variances
        squared_error = (y.view(-1) - means) ** 2
        nll = 0.5 * (torch.log(variances) + precision * squared_error)
        return nll.mean()

    def _train_single_model(self, X_tensor, y_tensor, input_dim, idx): 
        """
        Train a single ensemble member.

        Args:
            X_tensor (torch.Tensor): Input tensor.
            y_tensor (torch.Tensor): Target tensor.
            input_dim (int): Number of input features.
            idx (int): Index of the model (for seeding and logging).

        Returns:
            (Tuple[MLP, Logger]): (trained model, logger instance)
        """
        X_tensor = X_tensor.to(self.device)
        y_tensor = y_tensor.to(self.device)

        if self.random_seed is not None: 
            torch.manual_seed(self.random_seed + idx)
            np.random.seed(self.random_seed + idx)

        activation = get_activation(self.activation_str)
        model = MLP(input_dim, self.hidden_sizes, activation).to(self.device)

        optimizer = self.optimizer_cls(
            model.parameters(), lr=self.learning_rate, **self.optimizer_kwargs
        )
        scheduler = None 
        if self.scheduler_cls: 
            scheduler = self.scheduler_cls(optimizer, **self.scheduler_kwargs)

        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        logger = Logger(
            use_wandb=self.use_wandb,
            project_name=self.wandb_project,
            run_name=self.wandb_run_name + str(idx) if self.wandb_run_name is not None else None,
            config={"n_estimators": self.n_estimators, "learning_rate": self.learning_rate, "epochs": self.epochs},
            name=f"Estimator-{idx}"
        )
        
        for epoch in range(self.epochs): 
            model.train()
            epoch_loss = 0.0 
            for xb, yb in dataloader: 
                optimizer.zero_grad() 
                preds = model(xb)
                loss = self.loss_fn(preds, yb)
                loss.backward() 
                optimizer.step() 
                epoch_loss += loss.item()
            
            if epoch % (self.epochs / 20) == 0:
                logger.log({"epoch": epoch, "train_loss": epoch_loss})

            if scheduler: 
                scheduler.step()

        logger.finish()
        return model, logger
    
    def fit(self, X, y): 
        """
        Fit the ensemble on training data.

        Args:
            X (array-like or torch.Tensor): Training inputs.
            y (array-like or torch.Tensor): Training targets.

        Returns:
            (DeepEnsembleRegressor): Fitted estimator.
        """
        X_tensor, y_tensor = validate_and_prepare_inputs(X, y, device=self.device)
        input_dim = X_tensor.shape[1]
        self.input_dim = input_dim
        
        if self.scale_data: 
            X_tensor = self.input_scaler.fit_transform(X_tensor)
            y_tensor = self.output_scaler.fit_transform(y_tensor)

        results = Parallel(n_jobs=self.n_jobs)(
            delayed(self._train_single_model)(X_tensor, y_tensor, input_dim, i)
            for i in range(self.n_estimators)
        )

        self.models, self._loggers = zip(*results)

        self.fitted = True
        return self
    
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
        if self.scale_data: 
            X_tensor = self.input_scaler.transform(X_tensor)

        preds = [] 

        for model in self.models: 
            model.eval()
            pred = model(X_tensor)
            preds.append(pred)

        preds = torch.stack(preds)

        means = preds[:, :, 0]
        variances = preds[:, :, 1]

        mean = means.mean(dim=0)
        variance = torch.mean(variances + means ** 2, dim=0) - mean ** 2
        std = variance.sqrt()

        std_mult = torch.tensor(st.norm.ppf(1 - self.alpha / 2), device=mean.device)

        lower = mean - std * std_mult 
        upper = mean + std * std_mult 

        if self.scale_data: 
            mean = self.output_scaler.inverse_transform(mean.view(-1, 1)).squeeze()
            lower = self.output_scaler.inverse_transform(lower.view(-1, 1)).squeeze()
            upper = self.output_scaler.inverse_transform(upper.view(-1, 1)).squeeze() 

        if not self.requires_grad: 
            return mean.detach().cpu().numpy(), lower.detach().cpu().numpy(), upper.detach().cpu().numpy()

        else: 
            return mean, lower, upper

    def save(self, path):
        """
        Save the trained ensemble to disk.

        Args:
            path (str or pathlib.Path): Directory path to save the model and metadata.
        """
        if not self.fitted: 
            raise ValueError("Model not yet fit. Please call fit() before save().")
        
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save config (exclude non-serializable or large objects)
        config = {
            k: v for k, v in self.__dict__.items()
            if k not in ["models", "optimizer_cls", "optimizer_kwargs", "scheduler_cls", "scheduler_kwargs", 
                         "input_scaler", "output_scaler", "_loggers", "training_logs", "tuning_loggers", 
                         "tuning_logs"]
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
        for i, model in enumerate(self.models):
            torch.save(model.state_dict(), path / f"model_{i}.pt")

        for i, logger in enumerate(getattr(self, "_loggers", [])):
            logger.save_to_file(path, idx=i, name="estimator")

        for i, logger in enumerate(getattr(self, "tuning_loggers", [])): 
            logger.save_to_file(path, name="tuning", idx=i)

    @classmethod
    def load(cls, path, device="cpu", load_logs=False):
        """
        Load a saved ensemble regressor from disk.

        Args:
            path (str or pathlib.Path): Directory path to load the model from.
            device (str or torch.device): Device to load the model onto.
            load_logs (bool): Whether to load training and tuning logs.

        Returns:
            (DeepEnsembleRegressor): Loaded model instance.
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

        # Recreate models
        model.input_dim = input_dim
        activation = get_activation(config["activation_str"])
        model.models = []
        for i in range(config["n_estimators"]):
            m = MLP(model.input_dim, config["hidden_sizes"], activation).to(device)
            m.load_state_dict(torch.load(path / f"model_{i}.pt", map_location=device))
            model.models.append(m)

        with open(path / "extras.pkl", 'rb') as f: 
            optimizer_cls, optimizer_kwargs, scheduler_cls, scheduler_kwargs, input_scaler, output_scaler = pickle.load(f)

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


