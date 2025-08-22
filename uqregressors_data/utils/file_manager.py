"""
file_manager 
------------
Handles saving paths, including saving and loading models and plots. 

Examples: 
    >>> from uqregressors.bayesian.deep_ens import DeepEnsembleRegressor
    >>> from uqregressors.metrics.metrics import compute_all_metrics
    >>> from uqregressors.utils.file_manager import FileManager

    >>> # Instantiate File Manager
    >>> BASE_PATH = "C:/.uqregressors"
    >>> fm = FileManager(BASE_PATH)  # Replace with desired base path

    >>> # Fit a model and compute metrics
    >>> reg = DeepEnsembleRegressor()
    >>> reg.fit(X_train, y_train)
    >>> mean, lower, upper = reg.predict(X_test)
    >>> metrics = compute_all_metrics(
    ...     mean, lower, upper, y_test, reg.alpha
    ... )

    >>> # Save model and metrics
    >>> save_path = fm.save_model(
    ...     name="Deep_Ens",
    ...     model=reg,
    ...     metrics=metrics,
    ...     X_train=X_train,
    ...     y_train=y_train,
    ...     X_test=X_test,
    ...     y_test=y_test
    ... )
    >>> # Will save to: BASE_PATH/models/Deep_Ens

    >>> # Alternatively, specify full path directly
    >>> save_path = fm.save_model(path="SAVE_PATH", model=reg, ...)

    >>> # Load model and metrics
    >>> load_dict = fm.load_model(
    ...     reg.__class__, save_path, load_logs=True
    ... )
    >>> metrics = load_dict["metrics"]
    >>> loaded_model = load_dict["model"]
    >>> X_test = load_dict["X_test"]
"""
from pathlib import Path
from datetime import datetime
import json
import numpy as np
import warnings
import matplotlib.pyplot as plt 

class FileManager:
    """
    FileManager class to handle paths and saving.

    Args: 
        base_dir (str): Base directory to save files to. Defaults to creating a folder ".uqregressors" within the Users home path. 

    Attributes: 
        base_dir (Path): The base directory as a Path object.
        model_dir (Path): The directory "models" within the base_dir, where models will be saved and loaded.
    """
    def __init__(self, base_dir=Path.home() / ".uqregressors"):
        self.base_dir = Path(base_dir)
        self.model_dir = self.base_dir / "models"
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def get_timestamped_path(self, model_class_name: str) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return self.model_dir / f"{model_class_name}_{timestamp}"

    def get_named_path(self, name: str) -> Path:
        return self.model_dir / name

    def save_model(
        self, model, name=None, path=None, metrics=None, X_train=None, y_train=None, X_test=None, y_test=None
    ) -> Path:
        """
        Saves a model, along with metrics, and training and testing data

        Args: 
            model (BaseEstimator): The regressor to save. Note that it must implement the save method.
            name (str): The name of the model for directory purposes. If given, the model will be saved wihin the directory: fm.base_dir/models/name.
            path (str): The path to the directory where the model should be saved. Only one of name or path should be given. If neither are given,
                        a directory with the model class and timestamp is created. 
            metrics (dict): A dictionary of metrics to store. Can be used with uqregressors.metrics.metrics.compute_all_metrics.
            X_train (array-like): Training features. 
            y_train (array-like): Training targets. 
            X_test (array-like): Testing features. 
            y_test (array-like): Testing targets.
        """
        if name is not None:
            if path is not None:
                warnings.warn(f"Both name and path given. Using named path: {path}")
            else: 
                path = self.get_named_path(name)
        elif path is None:
            path = self.get_timestamped_path(model.__class__.__name__)
        else:
            path = Path(path)

        path.mkdir(parents=True, exist_ok=True)

        if not hasattr(model, "save") or not callable(model.save):
            raise AttributeError(f"{model.__class__.__name__} must implement `save(path)`")
        model.save(path)

        if metrics:
            with open(path / "metrics.json", "w") as f:
                json.dump(metrics, f, indent=2)

        for name, array in [("X_train", X_train), ("y_train", y_train), ("X_test", X_test), ("y_test", y_test)]:
            if array is not None:
                np.save(path / f"{name}.npy", np.array(array))

        print(f"Model and additional artifacts saved to: {path}")
        return path

    def load_model(self, model_class, path=None, name=None, device="cpu", load_logs=False):
        """
        Loads a model and associated metadata from path

        Args: 
            model_class (BaseEstimator): The class of the model to be loaded. This should match with the class of model which was saved. 
            path (str): The path to the directory in which the model and associated metadata is stored. If not given, name must be given. 
            name (str): The name if the directory containing the model is fm.base_dir/models/{name}. If not given, the path must be given. 
            device (str): The device, "cpu" or "cuda" to load the model with. 
            load_logs (bool): Whether training and hyperparameter logs should be loaded along with the model so they can be accessed by code.

        Returns: 
            (dict): Dictionary of loaded objects with the following keys: 

                    model: The loaded model,

                    metrics: The loaded metrics or None if there is no metrics.json file, 

                    X_train: The loaded training features or None if there is no X_train.npy file, 

                    y_train: The loaded training targets or None if there is no y_train.npy file,

                    X_test: The loaded testing features or None if there is no X_test.npy file, 

                    y_test: The loaded testing targets or None if there is no y_test.npy file.
        """
        if name:
            path = self.get_named_path(name)
        elif path:
            path = Path(path)
        else:
            raise ValueError("Either `name` or `path` must be specified.")
        
        if not path.exists():
            raise FileNotFoundError(f"Path {path} does not exist")

        if not hasattr(model_class, "load") or not callable(model_class.load):
            raise AttributeError(f"{model_class.__name__} must implement `load(path)`")

        from torch.serialization import safe_globals
        with safe_globals([np._core.multiarray._reconstruct, np.ndarray, np.dtype]):
            model = model_class.load(path, device=device, load_logs=load_logs)

        def try_load(name):
            f = path / f"{name}.npy"
            return np.load(f) if f.exists() else None

        metrics = None
        metrics_path = path / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path) as f:
                metrics = json.load(f)

        return {
            "model": model,
            "metrics": metrics,
            "X_train": try_load("X_train"),
            "y_train": try_load("y_train"),
            "X_test": try_load("X_test"),
            "y_test": try_load("y_test"),
        }
    
    def save_plot(self, fig, model_path, filename="plot.png", show=True, subdir="plots"):
        """
        A helper method to save plots to a subdirectory within the directory in which the model would be saved. 

        Args: 
            fig (matplotlib.figure.Figure): The figure to be saved. 
            model_path (str): The directory in which to create a "plots" subdirectory where the image will be saved. 
            filename (str): The filename of the plot to be saved, including the file extension.
            show (bool): Whether to display the plot after saving it. 
            subdir (str): The subdirectory in which the plot will be saved, each image will be saved to model_path/subdir/filename .
        """

        path = Path(model_path)
        plot_dir = path / subdir
        plot_dir.mkdir(parents=True, exist_ok=True)
        save_path = plot_dir / filename

        plt.figure(fig.number)
        fig.savefig(save_path, bbox_inches='tight')
        
        if show:
            plt.show(fig)
        plt.close(fig)
        print(f"Plot saved to {save_path}")
        return save_path

