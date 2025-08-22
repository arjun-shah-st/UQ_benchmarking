import gpytorch
import torch
from uqregressors.utils.logging import Logger
import scipy.stats as st
from pathlib import Path
import json
import pickle
from uqregressors.utils.data_loader import validate_and_prepare_inputs, validate_X_input

class ExactGP(gpytorch.models.ExactGP): 
    """
    A custom GPyTorch Exact Gaussian Process model using a constant mean and a user-specified kernel.

    Args:
        kernel (gpytorch.kernels.Kernel): Kernel defining the covariance structure of the GP.
        train_x (torch.Tensor): Training inputs of shape (n_samples, n_features).
        train_y (torch.Tensor): Training targets of shape (n_samples,).
        likelihood (gpytorch.likelihoods.Likelihood): Likelihood function (e.g., GaussianLikelihood).
    """
    def __init__(self, kernel, train_x, train_y, likelihood):
        super(ExactGP, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = kernel
    
    def forward(self, x): 
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class BBMM_GP: 
    """
    A wrapper around GPyTorch's ExactGP for regression with uncertainty quantification.

    Supports custom kernels, optimizers, learning schedules, and logging.
    Outputs mean predictions and confidence intervals using predictive variance.

    Args:
        name (str): Name of the model instance.
        kernel (gpytorch.kernels.Kernel): Covariance kernel.
        likelihood (gpytorch.likelihoods.Likelihood): Likelihood function used in GP.
        alpha (float): Significance level for predictive intervals (e.g. 0.1 = 90% CI).
        requires_grad (bool): If True, returns tensors requiring gradients during prediction.
        learning_rate (float): Optimizer learning rate.
        epochs (int): Number of training epochs.
        optimizer_cls (Callable): Optimizer class (e.g., torch.optim.Adam).
        optimizer_kwargs (dict): Extra keyword arguments for the optimizer.
        scheduler_cls (Callable or None): Learning rate scheduler class.
        scheduler_kwargs (dict): Extra keyword arguments for the scheduler.
        loss_fn (Callable or None): Custom loss function. Defaults to negative log marginal likelihood.
        device (str): Device to train the model on ("cpu" or "cuda").
        use_wandb (bool): If True, enables wandb logging.
        wandb_project (str or None): Name of the wandb project.
        wandb_run_name (str or None): Name of the wandb run.
        random_seed (int or None): Random seed for reproducibility.
        tuning_loggers (List[Logger]): Optional list of loggers from hyperparameter tuning.

    Attributes: 
        _loggers (list): Logger of training loss.
        fitted (bool): Whether the fit method has been successfully called.
    """
    def __init__(self, 
                 name="BBMM_GP_Regressor",
                 kernel=gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel()), 
                 likelihood=gpytorch.likelihoods.GaussianLikelihood(), 
                 alpha=0.1,
                 requires_grad=False,
                 learning_rate=1e-3,
                 epochs=200, 
                 optimizer_cls=torch.optim.Adam,
                 optimizer_kwargs=None,
                 scheduler_cls=None,
                 scheduler_kwargs=None,
                 loss_fn=None, 
                 device="cpu", 
                 use_wandb=False,
                 wandb_project=None,
                 wandb_run_name=None, 
                 random_seed=None, 
                 tuning_loggers=[],
            ):
        self.name = name
        self.kernel = kernel 
        self.likelihood = likelihood
        self.alpha = alpha 
        self.requires_grad = requires_grad
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.optimizer_cls = optimizer_cls 
        self.optimizer_kwargs = optimizer_kwargs or {} 
        self.scheduler_cls = scheduler_cls
        self.scheduler_kwargs = scheduler_kwargs or {}
        self.loss_fn = loss_fn
        self.device = device 
        self.use_wandb = use_wandb
        self.wandb_project = wandb_project 
        self.wandb_run_name = wandb_run_name
        self.model = None
        self.random_seed = random_seed

        self._loggers = []
        self.training_logs = None 
        self.tuning_loggers = tuning_loggers 
        self.tuning_logs = None

        self.train_X = None 
        self.train_y = None

        self.fitted = False

    def fit(self, X, y): 
        """
        Fits the GP model to training data.

        Args:
            X (np.ndarray or torch.Tensor): Training features of shape (n_samples, n_features).
            y (np.ndarray or torch.Tensor): Training targets of shape (n_samples,).
        """
        X_tensor, y_tensor = validate_and_prepare_inputs(X, y, device=self.device, requires_grad=self.requires_grad)
        y_tensor = y_tensor.view(-1)

        self.train_X = X_tensor 
        self.train_y = y_tensor

        if self.random_seed is not None: 
            torch.manual_seed(self.random_seed)

        config = {
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
        }

        logger = Logger(
            use_wandb=self.use_wandb,
            project_name=self.wandb_project,
            run_name=self.wandb_run_name,
            config=config,
        )

        model = ExactGP(self.kernel, X_tensor, y_tensor, self.likelihood)
        self.model = model.to(self.device)

        self.model.train()
        self.likelihood.train()

        if self.loss_fn == None: 
            self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, model)
            self.loss_fn = self.mll_loss

        optimizer = self.optimizer_cls(
            model.parameters(), lr=self.learning_rate, **self.optimizer_kwargs
        )

        scheduler = None
        if self.scheduler_cls is not None:
            scheduler = self.scheduler_cls(optimizer, **self.scheduler_kwargs)

        for epoch in range(self.epochs): 
            optimizer.zero_grad()
            preds = model(X_tensor)
            loss = self.loss_fn(preds, y_tensor)
            loss.backward()
            optimizer.step() 

            if scheduler is not None:
                scheduler.step()
            if epoch % (self.epochs / 20) == 0:
                logger.log({"epoch": epoch, "train_loss": loss})
        
        self._loggers.append(logger)
        self.fitted=True

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
        
        X_tensor = validate_X_input(X, device=self.device, requires_grad=True)
        self.model.eval()
        self.likelihood.eval() 

        with torch.no_grad(), gpytorch.settings.fast_pred_var(): 
            preds = self.likelihood(self.model(X_tensor))

        with torch.no_grad(): 
            mean = preds.mean
            lower_2std, upper_2std = preds.confidence_region() 
            low_std, up_std = (mean - lower_2std) / 2, (upper_2std - mean) / 2 

        z_score = st.norm.ppf(1 - self.alpha / 2)
        lower = mean - z_score * low_std
        upper = mean + z_score * up_std

        if not self.requires_grad: 
            return mean.detach().cpu().numpy(), lower.detach().cpu().numpy(), upper.detach().cpu().numpy()

        else: 
            return mean, lower, upper
    
    def mll_loss(self, preds, y): 
        """
        Computes the negative log marginal likelihood (default loss function).

        Args:
            preds (gpytorch.distributions.MultivariateNormal): GP predictive distribution.
            y (torch.Tensor): Ground truth targets.

        Returns:
            (torch.Tensor): Negative log marginal likelihood loss.
        """
        return -torch.sum(self.mll(preds, y))
    

    def save(self, path):
        """
        Saves model configuration, weights, and training data to disk.

        Args:
            path (Union[str, Path]): Path to save directory.
        """
        if not self.fitted: 
            raise ValueError("Model not yet fit. Please call fit() before save().")
        
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save config (exclude non-serializable or large objects)
        config = {
            k: v for k, v in self.__dict__.items()
            if k not in ["model", "kernel", "likelihood", "optimizer_cls", "optimizer_kwargs", "scheduler_cls", "scheduler_kwargs", 
                         "_loggers", "training_logs", "tuning_loggers", "tuning_logs", "train_X", "train_y"]
            and not callable(v)
            and not isinstance(v, (torch.nn.Module, torch.Tensor))
        }
        config["optimizer"] = self.optimizer_cls.__class__.__name__ if self.optimizer_cls is not None else None
        config["scheduler"] = self.optimizer_cls.__class__.__name__ if self.scheduler_cls is not None else None

        with open(path / "config.json", "w") as f:
            json.dump(config, f, indent=4)

        with open(path / "extras.pkl", 'wb') as f: 
            pickle.dump([self.kernel, self.likelihood, self.optimizer_cls, 
                         self.optimizer_kwargs, self.scheduler_cls, self.scheduler_kwargs], f)

        # Save model weights
        torch.save(self.model.state_dict(), path / f"model.pt")
        torch.save([self.train_X, self.train_y], path / f"train.pt")

        for i, logger in enumerate(getattr(self, "_loggers", [])):
            logger.save_to_file(path, idx=i, name="estimator")

        for i, logger in enumerate(getattr(self, "tuning_loggers", [])): 
            logger.save_to_file(path, name="tuning", idx=i)

    @classmethod
    def load(cls, path, device="cpu", load_logs=False):
        """
        Loads a saved BBMM_GP model from disk.

        Args:
            path (Union[str, Path]): Path to saved model directory.
            device (str): Device to map model to ("cpu" or "cuda").
            load_logs (bool): If True, also loads training/tuning logs.

        Returns:
            (BBMM_GP): Loaded model instance.
        """
        path = Path(path)

        # Load config
        with open(path / "config.json", "r") as f:
            config = json.load(f)
        config["device"] = device

        config.pop("optimizer", None)
        config.pop("scheduler", None)
        fitted = config.pop("fitted", False)
        model = cls(**config)

        with open(path / "extras.pkl", 'rb') as f: 
            kernel, likelihood, optimizer_cls, optimizer_kwargs, scheduler_cls, scheduler_kwargs = pickle.load(f)

        train_X, train_y = torch.load(path / f"train.pt")
        model.model = ExactGP(kernel, train_X, train_y, likelihood)
        model.model.load_state_dict(torch.load(path / f"model.pt", map_location=device))

        model.optimizer_cls = optimizer_cls 
        model.optimizer_kwargs = optimizer_kwargs 
        model.scheduler_cls = scheduler_cls 
        model.scheduler_kwargs = scheduler_kwargs
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