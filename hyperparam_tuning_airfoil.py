import numpy as np
import os
import pandas as pd
import joblib
import urllib.request
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from uqregressors.tuning.tuning import tune_hyperparams, log_likelihood, interval_score
from uqregressors.metrics.metrics import compute_all_metrics 
from uqregressors.plotting.plotting import plot_pred_vs_true
from uqregressors.utils.logging import set_logging_config
from uqregressors.utils.file_manager import FileManager
from uqregressors.bayesian.dropout import MCDropoutRegressor
from uqregressors.bayesian.deep_ens import DeepEnsembleRegressor
from uqregressors.bayesian.temp_scaled_deep_ens import TemperatureScaledDeepEnsembleRegressor
from uqregressors.conformal.cqr import ConformalQuantileRegressor
from uqregressors.conformal.k_fold_cqr import KFoldCQR
from uqregressors.conformal.conformal_ens import ConformalEnsRegressor
from uqregressors.bayesian.bbmm_gp import BBMM_GP
from uqregressors.bayesian.gaussian_process import GPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
import gpytorch
import pickle as pkl
import torch
import time
from torch.profiler import profile, record_function, ProfilerActivity
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingLR

device = "cpu"

set_logging_config(print=True)
seed = 42
# Helper to subsample datasets

def subsample(X, y, max_samples=5000, random_state=42):
    if len(X) > max_samples:
        idx = np.random.RandomState(random_state).choice(len(X), max_samples, replace=False)
        return X[idx], y[idx]
    return X, y

# Custom dataset loaders
def load_abalone():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data"
    col_names = [
        'Sex', 'Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight',
        'Viscera weight', 'Shell weight', 'Rings'
    ]
    df = pd.read_csv(url, header=None, names=col_names)
    # One-hot encode 'Sex'
    df = pd.get_dummies(df, columns=['Sex']).astype(float)
    X = df.drop('Rings', axis=1).values
    y = df['Rings'].values
    return X, y

def load_concrete():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls"
    df = pd.read_excel(url, dtype=float)
    X = df.drop('Concrete compressive strength(MPa, megapascals) ', axis=1).values
    y = df['Concrete compressive strength(MPa, megapascals) '].values
    return X, y

def load_blog():
    data_dir = 'blog_feedback_data'
    csv_path = os.path.join(data_dir, 'blogData_train.csv')
    df = pd.read_csv(csv_path, header=None, dtype=float)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    return X, y

def load_wine_red():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    df = pd.read_csv(url, sep=';', dtype=float)
    X = df.drop('quality', axis=1).values
    y = df['quality'].values
    return X, y

def load_all_datasets():
    datasets = {}
    #datasets['abalone'] = load_abalone()
    #datasets['concrete'] = load_concrete()
    #datasets['blog'] = load_blog()
    #datasets['wine_red'] = load_wine_red()
    datasets['Cd_nf_4D'] = load_nf_data("2D_data_neural_foil.pkl", 5)
    datasets['Cl_nf_4D'] = load_nf_data("2D_data_neural_foil.pkl", 4)
    datasets['Cm_nf_4D'] = load_nf_data("2D_data_neural_foil.pkl", 6)
    for name, (X, y) in datasets.items():
        X, y = subsample(np.array(X), np.array(y))
        datasets[name] = (X, y)
    return datasets

def analyze_datasets(): 
    datasets = load_all_datasets()
    for name, (X, y) in datasets.items():
        combined_data = np.c_[X, y]
        df = pd.DataFrame(combined_data)
        pd.plotting.scatter_matrix(df, figsize=(8,8), diagonal='kde', alpha=0.7)
        plt.title(name)
        plt.show() 

def load_nf_data(filename, output_col=4): 
    with open(filename, 'rb') as f: 
        data = pkl.load(f)
        df = pd.DataFrame(data)
        X = df.iloc[:, :4].values  # First 4 columns are input features
        y = df.iloc[:, output_col].values  # User specified output column
        return subsample(X, y, max_samples=1000)

dropout = MCDropoutRegressor(
    hidden_sizes=[256, 256], 
    dropout=0.1, 
    tau=1e6, 
    alpha=0.1, 
    learning_rate=0.001, 
    epochs=1, 
    batch_size=64,
    scale_data=True, 
    scheduler_cls = CosineAnnealingLR, 
    device=device
)

dropout_param_space = {
    "epochs": lambda trial:trial.suggest_int("epochs", 500, 2000),
    "learning_rate": lambda trial:trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
    "tau": lambda trial:trial.suggest_float("tau", 1, 1e6, log=True),
    "weight_decay": lambda trial:trial.suggest_float("weight_decay", 1e-4, 1, log=True)
}

deep_ens = DeepEnsembleRegressor(
    n_estimators=5, 
    hidden_sizes=[256, 256], 
    alpha = 0.1, 
    epochs = 1, 
    learning_rate=0.001, 
    batch_size=64,  # Increased from 64 to better utilize GPU
    scale_data=True, 
    n_jobs=5, 
    scheduler_cls = CosineAnnealingLR, 
    device=device
)

deep_ens_param_space = {
    "epochs": lambda trial:trial.suggest_int("epochs", 500, 2000),
    "learning_rate": lambda trial:trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
    "weight_decay": lambda trial:trial.suggest_float("weight_decay", 1e-4, 1, log=True)
}

temp_scale_deep_ens = TemperatureScaledDeepEnsembleRegressor(
    n_estimators=5, 
    hidden_sizes=[256, 256], 
    alpha=0.1, 
    epochs = 1, 
    learning_rate = 0.001,
    tau_lr=1e-1,
    temp_scaling_epochs=1000,
    batch_size = 64, 
    scale_data = True, 
    n_jobs = 5, 
    scheduler_cls = CosineAnnealingLR, 
    device=device 
)

temp_scale_deep_ens_param_space = {
    "epochs": lambda trial:trial.suggest_int("epochs", 500, 2000), 
    "learning_rate": lambda trial:trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True), 
    "val_size": lambda trial: trial.suggest_discrete_uniform("val_size", 0.1, 0.3, 0.1), 
    "lam": lambda trial: trial.suggest_float("lam", 0, 15)
}

cqr = ConformalQuantileRegressor(
    hidden_sizes=[256, 256], 
    cal_size=0.2, 
    alpha=0.1, 
    tau_lo=0.05, 
    epochs=1, 
    learning_rate=0.001, 
    batch_size=64,  # Increased from 64 to better utilize GPU
    scale_data=True, 
    scheduler_cls = CosineAnnealingLR, 
    device=device
)

cqr_param_space = {
    "epochs": lambda trial:trial.suggest_int("epochs", 500, 2000), 
    "learning_rate": lambda trial:trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True), 
    "tau_lo": lambda trial:trial.suggest_float("tau_lo", 0.04, 0.1),
    "weight_decay": lambda trial:trial.suggest_float("weight_decay", 1e-4, 1, log=True)
}


k_fold_cqr = KFoldCQR(
    n_estimators=5, 
    hidden_sizes=[256, 256], 
    alpha=0.1, 
    tau_lo=0.05,  # tau_hi will be automatically set to 0.95
    epochs=1, 
    learning_rate=0.001,
    batch_size=64, 
    scale_data=True, 
    n_jobs=5, 
    scheduler_cls = CosineAnnealingLR, 
    device=device
)

k_fold_cqr_param_space = {
    "epochs": lambda trial:trial.suggest_int("epochs", 500, 2000), 
    "learning_rate": lambda trial:trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True), 
    "tau_lo": lambda trial:trial.suggest_float("tau_lo", 0.04, 0.1),
    "weight_decay": lambda trial:trial.suggest_float("weight_decay", 1e-4, 1, log=True)
}

conformal_ens = ConformalEnsRegressor(
    n_estimators=5, 
    hidden_sizes=[256, 256], 
    alpha=0.1, 
    cal_size=0.2, 
    epochs=1, 
    learning_rate=0.001,
    batch_size=64, 
    scale_data=True, 
    n_jobs=5, 
    scheduler_cls = CosineAnnealingLR, 
    device=device
)

conformal_ens_param_space = {
    "epochs": lambda trial:trial.suggest_int("epochs", 500, 2000),
    "learning_rate": lambda trial:trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
    "weight_decay": lambda trial:trial.suggest_float("weight_decay", 1e-4, 1, log=True)
}

bbmm_gpr = BBMM_GP(
    kernel=gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel()) + gpytorch.kernels.ConstantKernel(),
    alpha=0.1, 
    epochs=1, 
    learning_rate=0.001, 
    device="cpu", 
    use_wandb=False, 
    scale_data=True
)
bbmm_gpr_param_space = {
    "epochs": lambda trial:trial.suggest_int("epochs", 500, 2000), 
    "learning_rate": lambda trial:trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
    "weight_decay": lambda trial:trial.suggest_float("weight_decay", 1e-4, 1, log=True)
}

UQ_REGRESSORS = {
    'MonteCarloDropout': (dropout, dropout_param_space, interval_score, False),
    'DeepEnsemble': (deep_ens, deep_ens_param_space, interval_score, False), 
    'ConformalQuantileRegression': (cqr, cqr_param_space, interval_score, False), 
    'KFoldQuantileRegression': (k_fold_cqr, k_fold_cqr_param_space, interval_score, False), 
    'NormalizedConformalEnsembles': (conformal_ens, conformal_ens_param_space, interval_score, False),
    #'BBMM_GPR': (bbmm_gpr, bbmm_gpr_param_space, interval_score, False)
}

def test_uqregressor_hyperparam_tuning(output_dir='uqregressor_results'):
    fm = FileManager(output_dir)
    datasets = load_all_datasets()
    for dataset_name, (X, y) in datasets.items():
        for reg_name, (reg_class, param_space, score_fn, greater) in UQ_REGRESSORS.items():
            print(f"Tuning {reg_name} on {dataset_name}...")
            if reg_name not in ['gpr']: 
                best_model, best_score, study = tune_hyperparams(reg_class, 
                                                                param_space, 
                                                                X, y, 
                                                                score_fn=score_fn, 
                                                                greater_is_better=greater, 
                                                                n_trials=30, 
                                                                n_splits=4)
                # Compute metrics 
            else: 
                best_model = reg_class.fit(X, y)

            mean, lower, upper = best_model.predict(X)
            metrics = compute_all_metrics(mean, lower, upper, y, alpha=best_model.alpha)

            print(metrics)

            # Save best model
            model_path = os.path.join(output_dir, f"{dataset_name}_{reg_name}")
            fm.save_model(best_model, 
                          path=model_path, 
                          metrics=metrics, 
                          X_train=X, 
                          y_train=y, 
                          X_test=X, 
                          y_test=y)

            if reg_name not in ['gpr']: 
                trials_df = study.trials_dataframe()
                study_path = os.path.join(output_dir, f"{dataset_name}_{reg_name}_optuna_trials.csv")
                trials_df.to_csv(study_path, index=False)
                print(f"Saved best model to {model_path} and Optuna trials to {study_path}")

            else: 
                print(f"Saved best model to {model_path}")

def test_gpr_fitting(output_dir='uqregressor_results'):
    """Fits a Gaussian Process Regressor to each dataset with proper scaling.
    
    Args:
        output_dir (str): Directory to save results
    """
    fm = FileManager(output_dir)
    datasets = load_all_datasets()
    
    # Initialize scalers
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    
    for dataset_name, (X, y) in datasets.items():
        print(f"\nFitting GPR on {dataset_name}...")
        
        # Create train/test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)
        
        # Scale the inputs and outputs
        X_train_scaled = x_scaler.fit_transform(X_train)
        y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
        
        X_test_scaled = x_scaler.transform(X_test)
        
        # Initialize and fit GPR on scaled data
        kernel = RBF(length_scale=1.0, length_scale_bounds=(1e-3, 1e3)) + WhiteKernel() # Initialize with length_scale=1.0
        #gp_kwargs = {'optimizer': 'fmin_l_bfgs_b', 'n_restarts_optimizer': 5}
        #gpr = GPRegressor(
        #    kernel=kernel,
        #    alpha=0.1,
        #    gp_kwargs=gp_kwargs
        #)
        gpr = GaussianProcessRegressor(kernel, optimizer='fmin_l_bfgs_b', n_restarts_optimizer=5)
        gpr = gpr.fit(X_train_scaled, y_train_scaled)
        
        # Print optimized kernel parameters
        print(f"Optimized kernel parameters: {gpr.kernel_.get_params()}")
        
        mean_scaled, std = gpr.predict(X_test_scaled, return_std=True)
        lower_scaled = mean_scaled - 1.645 * std 
        upper_scaled = mean_scaled + 1.645 * std

        # Get predictions in scaled space
        #mean_scaled, lower_scaled, upper_scaled = gpr.predict(X_test_scaled)
        
        # Transform predictions back to original space
        mean = y_scaler.inverse_transform(mean_scaled.reshape(-1, 1)).ravel()
        lower = y_scaler.inverse_transform(lower_scaled.reshape(-1, 1)).ravel()
        upper = y_scaler.inverse_transform(upper_scaled.reshape(-1, 1)).ravel()
    
        
        # Compute metrics in original space
        metrics = compute_all_metrics(mean, lower, upper, y_test, alpha=0.1)
        
        # Save results
        model_path = os.path.join(output_dir, f"{dataset_name}_GPR")
        #fm.save_model(gpr, 
        #             path=model_path,
        #             metrics=metrics,
        #             X_train=X_train,
        #             y_train=y_train,
        #             X_test=X_test,
        #             y_test=y_test)
        
        print(f"Saved GPR model and results to {model_path}")
        print("Metrics:", metrics)
        plot_pred_vs_true(mean, lower, upper, y_test, show=True)
        plt.show()
if __name__ == "__main__":
    # Run original hyperparameter tuning
    test_uqregressor_hyperparam_tuning(output_dir='results/airfoil_results_lr_sched/p_1000_runs')
    
    # Run GPR fitting
    #test_gpr_fitting(output_dir='results/airfoil_results/p_1000_runs')

    #analyze_datasets()