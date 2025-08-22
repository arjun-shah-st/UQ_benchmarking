from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
import sklearn
import scipy.stats as st
from pathlib import Path 
import torch 
import json
import pickle 

class GPRegressor: 
    """
    A wrapper for scikit-learn's GaussianProcessRegressor with prediction intervals.

    This class provides a simplified interface to fit a Gaussian Process (GP) regressor,
    make predictions with uncertainty intervals, and save/load the model configuration.

    Args:
            name (str): Name of the model.
            kernel (sklearn.gaussian_process.kernels.Kernel): Kernel to use for the GP model.
            alpha (float): Significance level for the prediction interval.
            gp_kwargs (dict, optional): Additional keyword arguments for GaussianProcessRegressor.

    Attributes:
        name (str): Name of the model.
        kernel (sklearn.gaussian_process.kernels.Kernel): Kernel to use in the GP model.
        alpha (float): Significance level for confidence intervals (e.g., 0.1 for 90% CI).
        gp_kwargs (dict): Additional keyword arguments for the GaussianProcessRegressor.
        model (GaussianProcessRegressor): Fitted scikit-learn GP model.
        fitted (bool): Whether fit has been successfully called. 
    """
    def __init__(self, name="GP_Regressor", kernel = RBF(), 
                 alpha=0.1, 
                 gp_kwargs=None):

        self.name = name
        self.kernel = kernel 
        self.alpha = alpha 
        self.gp_kwargs = gp_kwargs or {}
        self.model = None
        self.fitted = False

    def fit(self, X, y): 
        """
        Fits the GP model to the input data.

        Args:
            X (np.ndarray): Feature matrix of shape (n_samples, n_features).
            y (np.ndarray): Target values of shape (n_samples,).
        Returns: 
            (GPRegressor): Fitted model.
        """
        model = GaussianProcessRegressor(kernel=self.kernel, **self.gp_kwargs)
        model.fit(X, y)
        self.model = model
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
        
        preds, std = self.model.predict(X, return_std=True)
        z_score = st.norm.ppf(1 - self.alpha / 2)
        mean = preds
        lower = mean - z_score * std
        upper = mean + z_score * std
        return mean, lower, upper
    
    def save(self, path): 
        """
        Saves the model and its configuration to disk.

        Args:
            path (Union[str, Path]): Directory where model and config will be saved.
        """
        if not self.fitted: 
            raise ValueError("Model not yet fit. Please call fit() before save().")
        
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True) 

        config = {
            k: v for k, v in self.__dict__.items()
            if k not in ["kernel", "model"]
            and not callable(v)
            and not isinstance(v, ())
        }
        config["kernel"] = self.kernel.__class__.__name__

        with open(path / "config.json", "w") as f:
            json.dump(config, f, indent=4)

        with open(path / "model.pkl", 'wb') as file: 
            pickle.dump(self, file)

    @classmethod
    def load(cls, path, device="cpu", load_logs=False): 
        """
        Loads a previously saved GPRegressor from disk.

        Args:
            path (Union[str, Path]): Path to the directory containing the saved model.
            device (str, optional): Unused, included for compatibility. Defaults to "cpu".
            load_logs (bool, optional): Unused, included for compatibility. Defaults to False.

        Returns:
            (GPRegressor): The loaded model instance.
        """
        path = Path(path)

        with open(path / "model.pkl", 'rb') as file: 
            model = pickle.load(file)
        
        return model
