"""
Tuning contains the helper function tune_hyperparams which uses the Bayesian hyperparameter optimization framework, [Optuna](https://optuna.org/)
as well as some examples of potential scoring functions. 

Important features of this hyperparameter optimization method are: 
    - Customizable scoring function 
    - Customizable hyperparameters to be tuned 
    - Customizable number of tuning iterations, search space 
    - Support for cross-validation 
"""
from skopt import BayesSearchCV
from sklearn.metrics import make_scorer
from sklearn.model_selection import train_test_split, KFold
import numpy as np
import warnings
from scipy.stats import norm
import tempfile
import optuna 
import copy 

warnings.filterwarnings("ignore")

def interval_score(estimator, X, y): 
    """
    Example of interval_score scoring function for hyperparameter tuning, greater=False
    """
    alpha = estimator.alpha
    _, lower, upper = estimator.predict(X)
    width = upper - lower
    penalty_lower = (2 / alpha) * (lower - y) * (y < lower)
    penalty_upper = (2 / alpha) * (y - upper) * (y > upper)
    return np.mean(width + penalty_lower + penalty_upper)

def interval_width(estimator, X, y): 
    """
    Example of minimizing interval width scoring function for hyperparameter tuning, greater=False
    """
    _, lower, upper = estimator.predict(X)
    return np.mean(upper - lower)

def log_likelihood(estimator, X, y): 
    """
    Example of maximizing log likelihood scoring function for hyperparameter tuning, greater=True
    """
    mean, lower, upper = estimator.predict(X)
    alpha = estimator.alpha 
    z = norm.ppf(1 - alpha / 2)
    std = (upper - lower) / (2 * z)
    std = np.clip(std, 1e-6, None)

    log_likelihoods = -0.5 * np.log(2 * np.pi * std**2) - 0.5 * ((y.ravel() - mean) / std) ** 2
    return np.mean(log_likelihoods)

def tune_hyperparams(
    regressor,
    param_space,
    X,
    y,
    score_fn,
    greater_is_better,
    initial_params = None,
    n_trials=20,
    n_splits=3,
    random_state=42,
    test_size=0.2,
    verbose=True,
):
    """
    Optimizes a scikit-learn-style regressor using Optuna.

    Supports CV when n_splits > 1, otherwise uses train/val split.

    Args:
        regressor (BaseEstimator): An instance of a base regressor (must have .fit and .predict).
        param_space (dict): Dict mapping param name → optuna suggest function (e.g., lambda t: t.suggest_float(...)).
        X (Union[torch.Tensor, np.ndarray, pd.DataFrame, pd.Series]): Training inputs.
        y (Union[torch.Tensor, np.ndarray, pd.DataFrame, pd.Series]): Training targets.
        score_fn (Callable(estimator, X_val, y_val) → float): Scoring function.
        greater_is_better (bool): Whether score_fn should be maximized (True) or minimized (False).
        initial_params (dict): A dictionary of the initial params to start with. Otherwise, chosen by Optuna algorithm.
        n_trials (int): Number of Optuna trials.
        n_splits (int): If >1, uses KFold CV; otherwise single train/val split.
        random_state (int): For reproducibility.
        test_size (float): Proportion of training examples to allocate to validation set; between 0 and 1. 
        verbose (bool): Print status messages.
    
    Returns:
        (Tuple[BaseEstimator, float, optuna.study]): A tuple containing the estimator with the optimized 
                                                     hyperparameters, the minimum score, and the optuna study object
    """

    direction = "maximize" if greater_is_better else "minimize"

    def objective(trial):
        # Sample hyperparameters
        trial_params = {k: suggest_fn(trial) for k, suggest_fn in param_space.items()}

        scores = []

        if n_splits == 1:
            # Single train/val split
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )

            # Copy the regressor if it has already been fit
            if regressor.fitted: 
                with tempfile.TemporaryDirectory() as tmpdir: 
                    regressor.save(tmpdir)
                    estimator = regressor.__class__.load(tmpdir)
            else: 
                estimator = regressor

            for param_name, param_value in trial_params.items():
                setattr(estimator, param_name, param_value)

            estimator.fit(X_train, y_train)
            score = score_fn(estimator, X_val, y_val)
            scores.append(score)
        else:
            # K-fold CV
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            for train_idx, val_idx in kf.split(X):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                # Copy the regressor if it has already been fit 
                if regressor.fitted: 
                    with tempfile.TemporaryDirectory() as tmpdir: 
                        regressor.save(tmpdir)
                        estimator = regressor.__class__.load(tmpdir)

                else: 
                    estimator = regressor
                
                for param_name, param_value in trial_params.items():
                    setattr(estimator, param_name, param_value)

                estimator.fit(X_train, y_train)
                score = score_fn(estimator, X_val, y_val)
                scores.append(score)

        mean_score = np.mean(scores)

        if verbose:
            print(f"Trial params: {trial_params} -> Score: {mean_score:.4f}")

        return mean_score

    study = optuna.create_study(direction=direction)

    if initial_params is not None: 
        study.enqueue_trial(initial_params)

    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    # Re-train on full data with best hyperparameters
    best_params = study.best_params

    with tempfile.TemporaryDirectory() as tmpdir: 
        regressor.save(tmpdir)
        best_estimator = regressor.__class__.load(tmpdir)

    for k, v in best_params.items():
        setattr(best_estimator, k, v)
    best_estimator.fit(X, y)

    if verbose:
        print("Best score:", study.best_value)
        print("Best hyperparameters:", best_params)

    return best_estimator, study.best_value, study