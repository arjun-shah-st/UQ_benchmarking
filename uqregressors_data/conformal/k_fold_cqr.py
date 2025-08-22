"""
K-Fold-CQR
----------

This module implements conformal quantile regression in a K-fold manner for regression of a one dimensional output. 

Key features are: 
    - Customizable neural network architecture
    - Tunable quantiles of the underyling regressors
    - Prediction intervals without distributional assumptions 
    - Parallel training of ensemble models with Joblib 
    - Customizable optimizer and loss function 
    - Optional Input/Output Normalization 
"""
import numpy as np 
import torch 
import torch.nn as nn 
from torch.utils.data import TensorDataset, DataLoader 
from sklearn.base import BaseEstimator, RegressorMixin 
from uqregressors.utils.activations import get_activation 
from uqregressors.utils.logging import Logger
from uqregressors.utils.data_loader import validate_and_prepare_inputs, validate_X_input
from uqregressors.utils.torch_sklearn_utils import TorchStandardScaler, TorchKFold
from joblib import Parallel, delayed 
from pathlib import Path 
import json 
import pickle

class QuantNN(nn.Module): 
    """
    A simple quantile neural network that estimates the lower and upper quantile when trained
    with a pinball loss function. 

    Args: 
        input_dim (int): Number of input features 
        hidden_sizes (list of int): List of hidden layer sizes
        dropout (None or float): The dropout probability - None if no dropout
        activation (torch.nn.Module): Activation function class (e.g., nn.ReLU).
    """
    def __init__(self, input_dim, hidden_sizes, dropout, activation): 
        super().__init__()
        layers = []
        for h in hidden_sizes:
            layers.append(nn.Linear(input_dim, h))
            layers.append(activation())
            if dropout is not None: 
                layers.append(nn.Dropout(dropout))
            input_dim = h
        output_layer = nn.Linear(hidden_sizes[-1], 2)
        layers.append(output_layer)
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
    

class KFoldCQR(BaseEstimator, RegressorMixin): 
    """
    K-Fold Conformalized Quantile Regressor for uncertainty estimation in regression tasks.

    This class trains an ensemble of quantile neural networks using K-Fold cross-validation,
    and applies conformal prediction to calibrate prediction intervals.

    Args:
        name (str): Name of the model.
        n_estimators (int): Number of K-Fold models to train.
        hidden_sizes (list): Sizes of the hidden layers for each quantile regressor.
        dropout (float or None): Dropout rate for the neural network layers.
        alpha (float): Miscoverage rate (1 - confidence level).
        requires_grad (bool): Whether inputs should require gradient.
        tau_lo (float): Lower quantile, defaults to alpha/2.
        tau_hi (float): Upper quantile, defaults to 1 - alpha/2.
        n_jobs (int): Number of parallel jobs for training.
        activation_str (str): String identifier of the activation function.
        learning_rate (float): Learning rate for training.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        optimizer_cls (type): Optimizer class.
        optimizer_kwargs (dict): Keyword arguments for optimizer.
        scheduler_cls (type or None): Learning rate scheduler class.
        scheduler_kwargs (dict): Keyword arguments for scheduler.
        loss_fn (callable or None): Loss function, defaults to quantile loss.
        device (str): Device to use for training and inference.
        use_wandb (bool): Whether to log training with Weights & Biases.
        wandb_project (str or None): wandb project name.
        wandb_run_name (str or None): wandb run name.
        scale_data (bool): Whether to normalize input/output data.
        input_scaler (TorchStandardScaler): Scaler for input features.
        output_scaler (TorchStandardScaler): Scaler for target outputs.
        random_seed (int or None): Random seed for reproducibility.
        tuning_loggers (list): Optional list of loggers for tuning.

    Attributes: 
        quantiles (Tensor): The lower and upper quantiles for prediction.
        models (list[QuantNN]): A list of the models in the ensemble.
        residuals (Tensor): The combined residuals on the calibration sets. 
        conformal_width (Tensor): The width needed to conformalize the quantile regressor, q. 
        _loggers (list[Logger]): Training loggers for each ensemble member. 
        fitted (bool): Whether fit has been successfully called. 
    """
    def __init__(
            self, 
            name="K_Fold_CQR_Regressor",
            n_estimators=5,
            hidden_sizes=[64, 64], 
            dropout = None,
            alpha=0.1, 
            requires_grad=False,
            tau_lo = None, 
            tau_hi = None, 
            n_jobs=1, 
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
            scale_data = True, 
            input_scaler = None, 
            output_scaler = None,
            random_seed=None, 
            tuning_loggers = []
    ):
        self.name = name
        self.n_estimators = n_estimators
        self.hidden_sizes = hidden_sizes
        self.dropout = dropout
        self.alpha = alpha
        self.requires_grad = requires_grad
        self.tau_lo = tau_lo or alpha / 2 
        self.tau_hi = tau_hi or 1 - alpha / 2
        self.activation_str = activation_str
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = optimizer_kwargs or {}
        self.scheduler_cls = scheduler_cls
        self.scheduler_kwargs = scheduler_kwargs or {}
        self.loss_fn = loss_fn or self.quantile_loss
        self.device = device

        self.use_wandb = use_wandb
        self.wandb_project = wandb_project
        self.wandb_run_name = wandb_run_name

        self.n_jobs = n_jobs
        self.random_seed = random_seed
        self.quantiles = torch.tensor([self.tau_lo, self.tau_hi], device=self.device)
        self.models = []
        self.residuals = []
        self.conformal_width = None
        self.input_dim = None
        if self.n_estimators == 1: 
            raise ValueError("n_estimators set to 1. To use a single Quantile Regressor, use a non-ensembled Quantile Regressor class")
        self.scale_data = scale_data 
        self.input_scaler = input_scaler or TorchStandardScaler() 
        self.output_scaler = output_scaler or TorchStandardScaler()

        self._loggers = []
        self.training_logs = None
        self.tuning_loggers = tuning_loggers
        self.tuning_logs = None
        self.fitted = False 

    def quantile_loss(self, preds, y): 
        """
        Quantile loss used for training the quantile regressors.

        Args:
            preds (Tensor): Predicted quantiles, shape (batch_size, 2).
            y (Tensor): True target values, shape (batch_size,).

        Returns:
            (Tensor): Scalar loss.
        """
        error = y.view(-1, 1) - preds
        return torch.mean(torch.max(self.quantiles * error, (self.quantiles - 1) * error))

    def _train_single_model(self, X_tensor, y_tensor, input_dim, train_idx, cal_idx, model_idx): 
        if self.random_seed is not None: 
            torch.manual_seed(self.random_seed + model_idx)
            np.random.seed(self.random_seed + model_idx)

        activation = get_activation(self.activation_str)
        model = QuantNN(input_dim, self.hidden_sizes, self.dropout, activation).to(self.device)

        optimizer = self.optimizer_cls(
            model.parameters(), lr=self.learning_rate, **self.optimizer_kwargs
        )
        scheduler = None 
        if self.scheduler_cls: 
            scheduler = self.scheduler_cls(optimizer, **self.scheduler_kwargs)

        X_train = X_tensor.detach()[train_idx]
        y_train = y_tensor.detach()[train_idx]
        dataset = TensorDataset(X_train, y_train)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        logger = Logger(
            use_wandb=self.use_wandb,
            project_name=self.wandb_project,
            run_name=self.wandb_run_name + str(model_idx) if self.wandb_run_name is not None else None,
            config={"n_estimators": self.n_estimators, "learning_rate": self.learning_rate, "epochs": self.epochs},
            name=f"Estimator-{model_idx}"
        )
        
        model.train()
        for epoch in range(self.epochs): 
            model.train()
            epoch_loss = 0.0 
            for xb, yb in dataloader: 
                optimizer.zero_grad() 
                preds = model(xb)
                loss = self.loss_fn(preds, yb)
                loss.backward() 
                optimizer.step() 
                epoch_loss += loss 
            
            if epoch % (self.epochs / 20) == 0:
                logger.log({"epoch": epoch, "train_loss": epoch_loss})

            if scheduler: 
                scheduler.step()

        model.eval()
        test_X = X_tensor[cal_idx]
        test_y = y_tensor[cal_idx]
        oof_preds = model(test_X)
        loss_matrix =(oof_preds - test_y) * torch.tensor([1.0, -1.0], device=self.device)
        residuals = torch.max(loss_matrix, dim=1).values
        logger.finish()
        return model, residuals, logger
    
    def fit(self, X, y): 
        """
        Fit the ensemble on training data.

        Args:
            X (array-like or torch.Tensor): Training inputs.
            y (array-like or torch.Tensor): Training targets.

        Returns:
            (KFoldCQR): Fitted estimator.
        """
        X_tensor, y_tensor = validate_and_prepare_inputs(X, y, device=self.device, requires_grad=self.requires_grad)
        input_dim = X_tensor.shape[1]
        self.input_dim = input_dim


        if self.scale_data:
            X_tensor = self.input_scaler.fit_transform(X_tensor)
            y_tensor = self.output_scaler.fit_transform(y_tensor)

        kf = TorchKFold(n_splits=self.n_estimators, shuffle=True)

        results = Parallel(n_jobs=self.n_jobs)(
            delayed(self._train_single_model)(X_tensor, y_tensor, input_dim, train_idx, cal_idx, i)
            for i, (train_idx, cal_idx) in enumerate(kf.split(X_tensor))
        )

        self.models = [result[0] for result in results]
        self.residuals = torch.cat([result[1] for result in results], dim=0).ravel()
        self._loggers = [result[2] for result in results]

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
        n = len(self.residuals)
        q = int((1 - self.alpha) * (n + 1))
        q = min(q, n-1)

        res_quantile = n-q
    
        self.conformal_width = torch.topk(self.residuals, res_quantile).values[-1]

        if self.scale_data: 
            X_tensor = self.input_scaler.transform(X_tensor)

        preds = [] 

        with torch.no_grad(): 
            for model in self.models: 
                model.eval()
                pred = model(X_tensor)
                preds.append(pred)

        preds = torch.stack(preds)

        means = torch.mean(preds, dim=2) 
        mean = torch.mean(means, dim=0)
 
        lower_cq = torch.mean(preds[:, :, 0], dim=0)
        upper_cq = torch.mean(preds[:, :, 1], dim=0)

        lower = lower_cq - self.conformal_width
        upper = upper_cq + self.conformal_width

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
        Save the trained model and associated configuration to disk.

        Args:
            path (str or Path): Directory to save model files.
        """
        if not self.fitted: 
            raise ValueError("Model not yet fit. Please call fit() before save().")
        
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save config (exclude non-serializable or large objects)
        config = {
            k: v for k, v in self.__dict__.items()
            if k not in ["models", "quantiles", "residuals", "conformal_width", "optimizer_cls", "optimizer_kwargs", "scheduler_cls", "scheduler_kwargs", 
                         "input_scaler", "output_scaler", "_loggers", "training_logs", "tuning_loggers", "tuning_logs"]
            and not callable(v)
            and not isinstance(v, (torch.nn.Module,))
        }

        config["optimizer"] = self.optimizer_cls.__class__.__name__ if self.optimizer_cls is not None else None
        config["scheduler"] = self.scheduler_cls.__class__.__name__ if self.scheduler_cls is not None else None
        config["input_scaler"] = self.input_scaler.__class__.__name__ if self.input_scaler is not None else None 
        config["output_scaler"] = self.output_scaler.__class__.__name__ if self.output_scaler is not None else None

        with open(path / "config.json", "w") as f:
            json.dump(config, f, indent=4)

        # Save model weights
        for i, model in enumerate(self.models):
            torch.save(model.state_dict(), path / f"model_{i}.pt")

        # Save residuals and conformity score
        torch.save({
            "conformal_width": self.conformal_width, 
            "residuals": self.residuals,
            "quantiles": self.quantiles,
        }, path / "extras.pt")

        with open(path / "extras.pkl", 'wb') as f: 
            pickle.dump([self.optimizer_cls, 
                        self.optimizer_kwargs, self.scheduler_cls, self.scheduler_kwargs, self.input_scaler, self.output_scaler], f)

        for i, logger in enumerate(getattr(self, "_loggers", [])):
            logger.save_to_file(path, idx=i, name="estimator")

        for i, logger in enumerate(getattr(self, "tuning_loggers", [])): 
            logger.save_to_file(path, name="tuning", idx=i)

    @classmethod
    def load(cls, path, device="cpu", load_logs=False):
        """
        Load a saved KFoldCQR model from disk.

        Args:
            path (str or Path): Directory containing saved model files.
            device (str): Device to load the model on ("cpu" or "cuda").
            load_logs (bool): Whether to also load training logs.

        Returns:
            (KFoldCQR): The loaded model instance.
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
            m = QuantNN(model.input_dim, config["hidden_sizes"], config["dropout"], activation).to(device)
            m.load_state_dict(torch.load(path / f"model_{i}.pt", map_location=device))
            model.models.append(m)

        # Load extras
        extras_path = path / "extras.pt"
        if extras_path.exists():
            extras = torch.load(extras_path, map_location=device, weights_only=False)
            model.conformal_width = extras.get("conformal_width", None)
            model.residuals = extras.get("residuals", None)
            model.quantiles = extras.get("quantiles", None)
        else:
            model.conformal_width = None
            model.residuals = None
            model.quantiles = None

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