import numpy as np
from uqregressors.bayesian.bbmm_gp import BBMM_GP
from uqregressors.utils.file_manager import FileManager
from uqregressors.metrics.metrics import compute_all_metrics
import torch
import gpytorch
import pickle as pkl
from sklearn.model_selection import train_test_split
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_and_prepare_data(filename="2D_data_neural_foil.pkl", output_col=4, max_samples=250, seed=42):
    """Load and prepare data consistently"""
    with open(filename, 'rb') as f:
        data = pkl.load(f)
        df = pd.DataFrame(data)
        X = df.iloc[:, :4].values
        y = df.iloc[:, output_col].values
        
        if len(X) > max_samples:
            idx = np.random.RandomState(seed).choice(len(X), max_samples, replace=False)
            X, y = X[idx], y[idx]
            
        return train_test_split(X, y, test_size=0.3, random_state=seed)

def test_model_consistency():
    # Initialize model with fixed parameters
    kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel()) + gpytorch.kernels.ConstantKernel()
    model = BBMM_GP(
        kernel=kernel,
        alpha=0.1,
        epochs=1452,
        learning_rate=0.0083913,
        device="cpu",
        use_wandb=False,
        scale_data=True
    )
    
    # Load and prepare data
    X_train, X_test, y_train, y_test = load_and_prepare_data()
    
    print("Data shapes:")
    print(f"X_train: {X_train.shape}")
    print(f"X_test: {X_test.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"y_test: {y_test.shape}")
    
    # Train model and get initial predictions
    print("\nTraining model...")
    model.fit(X_train, y_train)
    
    print("\nGetting predictions after training...")
    mean1, lower1, upper1 = model.predict(X_test)
    metrics1 = compute_all_metrics(mean1, lower1, upper1, y_test, alpha=0.1)
    print("\nMetrics after training:")
    for metric_name, value in metrics1.items():
        print(f"{metric_name}: {value}")
    
    # Save model and data
    print("\nSaving model and data...")
    fm = FileManager('test_consistency_results')
    model_path = os.path.join('test_consistency_results', 'test_model')
    fm.save_model(model, 
                 path=model_path,
                 metrics=metrics1,
                 X_train=X_train,
                 y_train=y_train,
                 X_test=X_test,
                 y_test=y_test)
    
    # Load model and get predictions again
    print("\nLoading model...")
    result_dict = fm.load_model(BBMM_GP, path=model_path)
    loaded_model = result_dict['model']
    loaded_X_test = result_dict['X_test']
    loaded_y_test = result_dict['y_test']
    
    print("\nVerifying data consistency...")
    print(f"X_test data identical: {np.allclose(X_test, loaded_X_test)}")
    print(f"y_test data identical: {np.allclose(y_test, loaded_y_test)}")
    
    print("\nGetting predictions with loaded model...")
    mean2, lower2, upper2 = loaded_model.predict(loaded_X_test)
    metrics2 = compute_all_metrics(mean2, lower2, upper2, loaded_y_test, alpha=0.1)
    
    print("\nComparing predictions:")
    print(f"Mean predictions identical: {np.allclose(mean1, mean2)}")
    print(f"Lower bounds identical: {np.allclose(lower1, lower2)}")
    print(f"Upper bounds identical: {np.allclose(upper1, upper2)}")
    
    print("\nMetrics comparison:")
    for metric_name in metrics1.keys():
        print(f"\n{metric_name}:")
        print(f"  Original: {metrics1[metric_name]}")
        print(f"  Loaded:   {metrics2[metric_name]}")
        print(f"  Difference: {abs(metrics1[metric_name] - metrics2[metric_name])}")
        
    # Save detailed arrays for debugging if needed
    if not np.allclose(mean1, mean2) or not np.allclose(lower1, lower2) or not np.allclose(upper1, upper2):
        print("\nSaving detailed debug information...")
        debug_info = {
            'original_mean': mean1,
            'original_lower': lower1,
            'original_upper': upper1,
            'loaded_mean': mean2,
            'loaded_lower': lower2,
            'loaded_upper': upper2,
            'mean_diff': mean1 - mean2,
            'lower_diff': lower1 - lower2,
            'upper_diff': upper1 - upper2
        }
        with open('test_consistency_results/debug_arrays.pkl', 'wb') as f:
            pkl.dump(debug_info, f)
        print("Debug information saved to 'test_consistency_results/debug_arrays.pkl'")

if __name__ == "__main__":
    test_model_consistency()
