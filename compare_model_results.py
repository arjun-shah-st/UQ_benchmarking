import os
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from uqregressors.bayesian.dropout import MCDropoutRegressor
from uqregressors.bayesian.deep_ens import DeepEnsembleRegressor
from uqregressors.conformal.cqr import ConformalQuantileRegressor
from uqregressors.conformal.k_fold_cqr import KFoldCQR
from uqregressors.conformal.conformal_ens import ConformalEnsRegressor
from uqregressors.bayesian.bbmm_gp import BBMM_GP
from uqregressors.bayesian.temp_scaled_deep_ens import TemperatureScaledDeepEnsembleRegressor
from uqregressors.utils.file_manager import FileManager
from uqregressors.metrics.metrics import compute_all_metrics
from uqregressors.plotting.plotting import plot_metrics_comparisons, generate_cal_curve, plot_cal_curve, plot_pred_vs_true
from sklearn.preprocessing import StandardScaler
from scipy import stats
import pandas as pd
from itertools import combinations
from torch.optim.lr_scheduler import CosineAnnealingLR
from covmetrics import ERT 

# List of datasets and models we expect
DATASETS =['Cl_nf_4d', 'Cd_nf_4D', 'Cm_nf_4D']#, 'Cl_nf_4D', 'Cd_nf_4D']#['abalone', 'concrete', 'blog', 'wine_red']

# Map model names to their classes
MODEL_CLASSES = {
    'MonteCarloDropout': MCDropoutRegressor,
    'DeepEnsemble': DeepEnsembleRegressor,
    'SingleDeepEnsemble': DeepEnsembleRegressor,
    'ConformalQuantileRegression': ConformalQuantileRegressor,
    'KFoldQuantileRegression': KFoldCQR,
    'NormalizedConformalEnsembles': ConformalEnsRegressor,
    'BBMM_GPR': BBMM_GP, 
    'TempScaleDeepEns': TemperatureScaledDeepEnsembleRegressor, 
    'TempScaleDeepEns_scale': TemperatureScaledDeepEnsembleRegressor
}

# Models that support calibration with their settings
CALIBRATION_SETTINGS = {
    'MonteCarloDropout': {'refit': False, 'n_points': 20, 'alpha_range': (0.01, 0.9)},
    'DeepEnsemble': {'refit': False, 'n_points': 20, 'alpha_range': (0.01, 0.9)},
    'NormalizedConformalEnsembles': {'refit': False, 'n_points': 20, 'alpha_range': (0.01, 0.9)},
    'ConformalQuantileRegression': {'refit': True, 'n_points': 5, 'alpha_range': (0.01, 0.9)},
    'KFoldQuantileRegression': {'refit': True, 'n_points': 5, 'alpha_range': (0.01, 0.9)}
}

# Group models by their optimization objective
LIKELIHOOD_MODELS = []
INTERVAL_SCORE_MODELS = ['MonteCarloDropout', 'DeepEnsemble', 'TempScaleDeepEns', 'ConformalQuantileRegression', 'KFoldQuantileRegression', 'NormalizedConformalEnsembles']
FINAL_COMP_MODELS = ['BBMM_GPR', 'MonteCarloDropout', 'DeepEnsemble','KFoldQuantileRegression', 'ConformalQuantileRegression', 'NormalizedConformalEnsembles']

def plot_tuning_curves(results_dir, dataset, save_dir=None, figsize=(8, 5)):
    """Plot hyperparameter tuning curves for all models using interval score.
    
    Args:
        results_dir (str): Directory containing the Optuna trial results
        dataset (str): Name of the dataset
        save_dir (str): Directory to save the plots
        figsize (tuple): Figure size for each plot
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Collect all values to compute the 90th percentile
    all_values = []
    
    # First pass to collect all values
    for model in INTERVAL_SCORE_MODELS:
        trial_path = os.path.join(results_dir, f"{dataset}_{model}_optuna_trials.csv")
        if os.path.exists(trial_path):
            trials_df = pd.read_csv(trial_path)
            all_values.extend(trials_df['value'].values)
    
    # Compute 90th percentile for y-axis limit
    if all_values:
        ylim_max = np.percentile(all_values, 90)
    
    # Plot interval scores for all models
    ax.set_title(f"Interval Score Optimization\n{dataset} dataset")
    ax.set_xlabel("Trial number")
    ax.set_ylabel("Interval Score (lower is better)")
    
    for model in INTERVAL_SCORE_MODELS:
        trial_path = os.path.join(results_dir, f"{dataset}_{model}_optuna_trials.csv")
        if os.path.exists(trial_path):
            trials_df = pd.read_csv(trial_path)
            ax.plot(trials_df['number'], trials_df['value'], '-o', label=model, alpha=0.7)
    
    # Set y-axis limit to 90th percentile
    if all_values:
        ax.set_ylim(0, ylim_max)
    
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{dataset}_tuning_curves.png")
        plt.savefig(save_path)
        print(f"Saved tuning curves to {save_path}")
        plt.show()
        plt.close()
    else:
        plt.show()

def plot_bootstrap_metric_comparisons(dataset_results, y_test, alpha, n_bootstrap=100, save_dir=None, 
                                filename=".png", show=False, figsize=(10, 6)):
    """Generate comparison plots with bootstrap confidence intervals for each metric.
    
    Args:
        dataset_results (dict): Dictionary mapping model names to (mean, lower, upper) predictions
        y_test (array-like): True test set values
        alpha (float): Confidence level
        n_bootstrap (int): Number of bootstrap samples
        save_dir (str): Directory to save plots
        filename (str): Base filename for plots
        show (bool): Whether to display plots
        figsize (tuple): Figure size
    """
    n_samples = len(y_test)
    rng = np.random.RandomState(42)
    
    # Store bootstrap results for each model and metric
    bootstrap_metrics = defaultdict(lambda: defaultdict(list))
    
    # For each bootstrap iteration
    for i in range(n_bootstrap):
        # Sample indices with replacement
        indices = rng.randint(0, n_samples, n_samples)
        y_boot = y_test[indices]
        
        # Compute metrics for each model on this bootstrap sample
        for model_name, (mean, lower, upper) in dataset_results.items():
            metrics = compute_all_metrics(
                mean[indices], lower[indices], upper[indices],
                y_boot, alpha, excluded_metrics=["group_conditional_coverage"]
            )
            for metric_name, value in metrics.items():
                bootstrap_metrics[metric_name][model_name].append(value)
    
    # Compute mean and confidence intervals for each metric
    metric_stats = {}
    for metric_name in bootstrap_metrics.keys():
        metric_stats[metric_name] = {
            model_name: {
                'mean': np.mean(values),
                'ci_lower': np.percentile(values, 2.5),
                'ci_upper': np.percentile(values, 97.5)
            }
            for model_name, values in bootstrap_metrics[metric_name].items()
        }
    
    better_direction = {
        "rmse": "lower is better",
        "nll_gaussian": "lower is better",
        "interval_score": "lower is better",
        "coverage": f"closer to {1-alpha:.2f} is better",
        "average_interval_width": "lower is better",
        "error_width_corr": "higher is better",
        "RMSCD": "lower is better",
        "RMSCD_under": "lower is better",
        "lowest_group_coverage": "higher is better"
    }
    
    # Create plots for each metric with error bars
    for metric_name, model_stats in metric_stats.items():
        plt.figure(figsize=figsize)
        
        # Prepare data for plotting
        models = list(model_stats.keys())
        means = [model_stats[m]['mean'] for m in models]
        errors_low = [model_stats[m]['mean'] - model_stats[m]['ci_lower'] for m in models]
        errors_high = [model_stats[m]['ci_upper'] - model_stats[m]['mean'] for m in models]
        
        # Create bar plot with error bars
        plt.bar(models, means, alpha=0.8)
        plt.errorbar(models, means, yerr=[errors_low, errors_high], fmt='none', color='k', capsize=5)
        
        # Formatting
        plt.xticks(rotation=45, ha='right')
        plt.ylabel(metric_name)
        plt.title(f"{metric_name} ({better_direction.get(metric_name, '')})")
        
        if metric_name == "coverage":
            plt.axhline(1-alpha, color='red', linestyle='--', label='Nominal')
            plt.legend()
        
        plt.tight_layout()
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"{metric_name}_bootstrap_{filename}")
            plt.savefig(save_path)
            print(f"Saved bootstrap comparison plot to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()

def load_and_compare_results(results_dir='results/airfoil_results/p_250_runs', 
                         generate_calibration=True,
                         generate_pred_vs_true=True,
                         generate_comparisons=True,
                         generate_bootstrap_comparisons=False,
                         generate_tuning_curves=True,
                         n_bootstrap=100):
    """Load saved models and generate various plots for analysis.
    
    Args:
        results_dir (str): Directory containing saved models and where plots will be saved
        generate_calibration (bool): Whether to generate calibration curves
        generate_pred_vs_true (bool): Whether to generate prediction vs true value plots
        generate_comparisons (bool): Whether to generate model comparison plots
    """
    # Dictionary to store results for each dataset
    dataset_results = defaultdict(dict)
    fm = FileManager(results_dir)

    # Load all models and organize by dataset
    for dataset in DATASETS:
        print(f"\nProcessing dataset: {dataset}")
        for model in MODEL_CLASSES.keys():
            model_path = os.path.join(results_dir, f"{dataset}_{model}")
            if os.path.exists(model_path):
                try:
                    # Load model and its results
                    model_class = MODEL_CLASSES[model]
                    result_dict = fm.load_model(model_class, path=model_path)
                    
                    # Get model predictions on test set
                    loaded_model = result_dict['model']
                    X_test = result_dict['X_test']
                    mean, lower, upper = loaded_model.predict(X_test)

                    # Store predictions for comparison
                    dataset_results[dataset][model] = (mean, lower, upper)
                    print(f"Successfully loaded {model}")

                    # Generate prediction vs true plots
                    if generate_pred_vs_true:
                        print(f"Generating prediction vs true plot for {model} on {dataset}...")
                        pred_save_dir = os.path.join(results_dir, 'plots', f"{dataset}_pred_vs_true")
                        os.makedirs(pred_save_dir, exist_ok=True)
                        
                        plot_pred_vs_true(
                            mean, lower, upper,
                            result_dict['y_test'],
                            samples=200,
                            include_confidence=True,
                            show=True,
                            save_dir=pred_save_dir,
                            filename=f"{dataset}_{model}_pred_vs_true.png",
                            title=f"Predictions vs True Values for {model} on {dataset}",
                            alpha=loaded_model.alpha
                        )
                    
                    # Generate calibration curves for supported models
                    if generate_calibration and model in CALIBRATION_SETTINGS:
                        print(f"Generating calibration curve for {model} on {dataset}...")
                        cal_save_dir = os.path.join(results_dir, 'plots', f"{dataset}_calibration")
                        os.makedirs(cal_save_dir, exist_ok=True)
                        
                        # Get model-specific settings
                        settings = CALIBRATION_SETTINGS[model]
                        alpha_min, alpha_max = settings['alpha_range']
                        alphas = np.linspace(alpha_min, alpha_max, settings['n_points'])
                        
                        # Generate calibration data
                        desired_coverage, empirical_coverage, interval_widths = generate_cal_curve(
                            loaded_model, 
                            result_dict['X_test'],
                            result_dict['y_test'],
                            alphas=alphas,
                            refit=settings['refit'],
                            X_train=result_dict['X_train'] if settings['refit'] else None,
                            y_train=result_dict['y_train'] if settings['refit'] else None
                        )
                        
                        # Plot calibration curve
                        plot_cal_curve(
                            desired_coverage,
                            empirical_coverage,
                            show=False,
                            save_dir=cal_save_dir,
                            filename=f"{dataset}_{model}_calibration.png",
                            title=f"Calibration Curve for {model} on {dataset}"
                        )
                except Exception as e:
                    print(f"Error loading {model} for {dataset}: {str(e)}")
        
        # If we have results for this dataset, create comparison plots
        if dataset_results[dataset]:
            # Get y_test and alpha from any of the loaded models
            first_model = list(dataset_results[dataset].keys())[0]
            any_model_path = os.path.join(results_dir, f"{dataset}_{first_model}")
            result_dict = fm.load_model(MODEL_CLASSES[first_model], path=any_model_path)
            y_test = result_dict['y_test']
            alpha = result_dict['model'].alpha
            
            if generate_comparisons:
                print(f"\nGenerating comparison plots for {dataset}")
                # Generate comparison plots
                save_dir = os.path.join(results_dir, f"{dataset}_comparisons")
                plot_metrics_comparisons(
                    dataset_results[dataset],
                    y_test,
                    alpha,
                    show=False,
                    save_dir=save_dir,
                    filename="comparison.png"
                )
                print(f"Saved comparison plots to {save_dir}")
            
            if generate_bootstrap_comparisons:
                print(f"\nGenerating bootstrap comparison plots for {dataset}")
                save_dir = os.path.join(results_dir, 'plots', f"{dataset}_bootstrap_comparisons")
                plot_bootstrap_metric_comparisons(
                    dataset_results[dataset],
                    y_test,
                    alpha,
                    n_bootstrap=n_bootstrap,
                    save_dir=save_dir,
                    filename="comparison.png",
                    show=False
                )
                print(f"Saved bootstrap comparison plots to {save_dir}")
            
            if generate_tuning_curves:
                print(f"\nGenerating tuning curves for {dataset}")
                save_dir = os.path.join(results_dir, 'plots', "{dataset}_tuning_curves")
                plot_tuning_curves(results_dir, dataset, save_dir=save_dir)
        else:
            print(f"No results found for dataset {dataset}")

def evaluate_final_models(results_dir, n_splits=20, test_size=0.3):
    """Evaluate final models over multiple train-test splits and plot metric distributions.
    
    Args:
        results_dir (str): Directory containing the saved models
        n_splits (int): Number of different train-test splits to evaluate
        test_size (float): Proportion of data to use for testing
        save_dir (str): Directory to save the evaluation plots
        figsize (tuple): Figure size for plots
    """
    fm = FileManager(results_dir)
    metric_results = defaultdict(lambda: defaultdict(list))
    rng = np.random.RandomState(42)
    
    # For each dataset
    for dataset in DATASETS:
        print(f"\nEvaluating models on {dataset}")
        
        # Load any model to get the dataset
        first_model = FINAL_COMP_MODELS[0]
        model_path = os.path.join(results_dir, "models", f"{dataset}_{first_model}")
        result_dict = fm.load_model(MODEL_CLASSES[first_model], path=model_path)
        X = result_dict['X_train']
        y = result_dict['y_train']
        
        # For each split
        for split in range(n_splits):
            print(f"Processing split {split + 1}/{n_splits}")
            
            # Create new train-test split
            indices = np.arange(len(X))
            rng.shuffle(indices)
            split_idx = int(len(X) * (1 - test_size))
            train_idx, test_idx = indices[:split_idx], indices[split_idx:]
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # For each model
            for model_name in FINAL_COMP_MODELS:
                print(f"Evaluating {model_name}")
                try:
                    # Load original model to get hyperparameters
                    model_path = os.path.join(results_dir, "models", f"{dataset}_{model_name}")
                    result_dict = fm.load_model(MODEL_CLASSES[model_name], path=model_path)
                    model = result_dict['model']
                    # Train and evaluate
                    model.fit(X_train, y_train)
                    mean, lower, upper = model.predict(X_test)
                    metrics = compute_all_metrics(mean, lower, upper, y_test, alpha=model.alpha)
                    
                    # Store results
                    for metric_name, value in metrics.items():
                        metric_results[(dataset, metric_name)][model_name].append(value)
                    
                    # Save model and results for this split
                    cv_save_dir = os.path.join(results_dir, 'CV_models', dataset, model_name, f'split_{split}')
                    os.makedirs(cv_save_dir, exist_ok=True)
                    
                    fm.save_model(model=model, path=cv_save_dir, 
                                  metrics=metrics, 
                                  X_train=X_train, 
                                  y_train=y_train,
                                  X_test=X_test, 
                                  y_test=y_test)
                    print(model.optimizer_kwargs)
                    print(f"Saved model and results for split {split} to {cv_save_dir}")
                
                except Exception as e:
                    print(f"Error evaluating {model_name}: {str(e)}")

def plot_cv_results(results_dir, alpha=0.05, save_dir=None, figsize=(10, 6)): 
    fm = FileManager(results_dir)
    metric_results = defaultdict(lambda: defaultdict(list))
    os.makedirs(save_dir, exist_ok=True)
    for dataset in DATASETS: 
        for model_name in FINAL_COMP_MODELS: 
            print(dataset, model_name)
            cv_base_dir = os.path.join(results_dir, 'CV_models', dataset, model_name)
            if not os.path.exists(cv_base_dir):
                print(f"No CV results found for {model_name} on {dataset}")
                continue
                
            split_dirs = [d for d in os.listdir(cv_base_dir) if d.startswith('split_')]
            for split_dir in split_dirs: 
                try: 
                    result_dict = fm.load_model(MODEL_CLASSES[model_name], 
                                                path = os.path.join(cv_base_dir, split_dir))
                    metrics = result_dict['metrics']
                    for metric_name, value in metrics.items():
                        metric_results[(dataset, metric_name)][model_name].append(value)
                except Exception as e: 
                    print(f"Error loading split {split_dir} for {model_name}: {str(e)}")
    
    # Plot results for this dataset
        better_direction = {
            "rmse": "lower is better",
            "nll_gaussian": "lower is better",
            "interval_score": "lower is better",
            "coverage": f"closer to {0.90} is better",
            "average_interval_width": "lower is better",
            "error_width_corr": "higher is better",
            "RMSCD": "lower is better",
            "RMSCD_under": "lower is better",
            "lowest_group_coverage": "higher is better"
        }
        
        for metric_name in metrics.keys():
            plt.figure(figsize=figsize)
            
            # Prepare data for plotting
            models = FINAL_COMP_MODELS
            results = metric_results[(dataset, metric_name)]
            df = pd.DataFrame(results)

            # Create and save a summary plot
            plt.figure(figsize=figsize)
            
            # Create vertical stripplot with jittered points
            sns.swarmplot(data=df, orient='v', size=10, alpha=0.6, 
                         palette='deep')  # using 'deep' palette for distinct colors
            
            # Add mean points with 95% confidence intervals using t-distribution
            means = df.mean()
            n = len(df)  # number of samples
            dof = n - 1  # degrees of freedom
            
            # Calculate confidence intervals using t-distribution
            conf_intervals = []
            for col in df.columns:
                std_err = df[col].std() / np.sqrt(n)
                t_val = stats.t.ppf(0.975, dof)  # 97.5th percentile for 95% CI
                ci = t_val * std_err
                conf_intervals.append(ci)
            
            plt.errorbar(x=range(len(means)), y=means, yerr=conf_intervals,
                        fmt='o', color='black', capsize=5, 
                        markersize=10, markerfacecolor='white',
                        label='Mean ± 95% CI of the mean (t-dist)')
            
            if metric_name == "coverage":
                plt.axhline(0.90, color='red', linestyle='--', label='Nominal')
                plt.legend()

            plt.xticks(range(len(df.columns)), df.columns, rotation=45, ha='right')
            plt.ylabel(f'{metric_name} ({better_direction[metric_name]})')
            plt.title(f'Distribution of {metric_name} Across CV Splits\n{dataset}')
            plt.legend()
            
            # Add grid for better readability
            plt.grid(True, axis='y', alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"{dataset}_{metric_name}_comparison.png"))
           # plt.show()
            plt.close()
            

def analyze_cv_results(results_dir, alpha=0.05):
    """Perform statistical analysis on the cross-validation results.
    
    Args:
        results_dir (str): Directory containing the CV_models results
        alpha (float): Significance level for statistical tests
    """
    
    fm = FileManager(results_dir)
    
    # For each dataset
    for dataset in DATASETS:
        print(f"\nAnalyzing results for dataset: {dataset}")
        
        # Collect interval scores for each model and split
        scores = defaultdict(list)
        n_splits = None
        
        # Load results from each split
        for model_name in FINAL_COMP_MODELS:
            cv_base_dir = os.path.join(results_dir, 'CV_models', dataset, model_name)
            if not os.path.exists(cv_base_dir):
                print(f"No CV results found for {model_name} on {dataset}")
                continue
                
            split_dirs = [d for d in os.listdir(cv_base_dir) if d.startswith('split_')]
            n_splits = len(split_dirs)
            
            for split_dir in split_dirs:
                try:
                    result_dict = fm.load_model(MODEL_CLASSES[model_name], 
                                              path=os.path.join(cv_base_dir, split_dir))
                    scores[model_name].append(result_dict['metrics']['interval_score'])
                except Exception as e:
                    print(f"Error loading split {split_dir} for {model_name}: {str(e)}")
        
        if not scores:
            print("No results found for analysis")
            continue
            
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(scores)
        
        # 1. Perform repeated measures ANOVA
        f_stat, p_value = stats.f_oneway(*[df[model] for model in df.columns])
        print("\nRepeated Measures ANOVA Results:")
        print(f"F-statistic: {f_stat:.4f}")
        print(f"p-value: {p_value:.4f}")
        print(f"Overall significant difference: {p_value < alpha}")
        
        # 2. Perform pairwise t-tests with Bonferroni correction
        print("\nPairwise Comparisons (with Bonferroni correction):")
        n_comparisons = len(list(combinations(df.columns, 2)))
        alpha_corrected = alpha / n_comparisons
        
        # Create results table
        comparison_results = []
        
        for model1, model2 in combinations(df.columns, 2):
            t_stat, p_val = stats.ttest_rel(df[model1], df[model2])
            mean_diff = df[model1].mean() - df[model2].mean()
            significant = p_val < alpha_corrected
            
            comparison_results.append({
                'Model 1': model1,
                'Model 2': model2,
                'Mean Difference': mean_diff,
                't-statistic': t_stat,
                'p-value': p_val,
                'Significant': significant
            })
        
        # Convert to DataFrame and display
        results_df = pd.DataFrame(comparison_results)
        print("\nPairwise comparison results:")
        print(results_df.to_string(index=False))
        
        # 3. Effect sizes (Cohen's d)
        print("\nEffect Sizes (Cohen's d):")
        for model1, model2 in combinations(df.columns, 2):
            d = (df[model1].mean() - df[model2].mean()) / \
                np.sqrt((df[model1].std()**2 + df[model2].std()**2) / 2)
            print(f"{model1} vs {model2}: {d:.4f}")
        
        # 4. Summary statistics
        print("\nSummary Statistics:")
        summary_df = df.agg(['mean', 'std', 'min', 'max'])
        print(summary_df)
        
        # Save results to file
        if results_dir:
            save_dir = os.path.join(results_dir, 'statistical_analysis')
            os.makedirs(save_dir, exist_ok=True)
            
            # Save detailed results
            results_df.to_csv(os.path.join(save_dir, f"{dataset}_pairwise_tests.csv"), index=False)
            summary_df.to_csv(os.path.join(save_dir, f"{dataset}_summary_stats.csv"))
            
            # Create and save a summary plot
            plt.figure(figsize=(10, 6))
            
            # Create vertical stripplot with jittered points
            sns.swarmplot(data=df, orient='v', size=10, alpha=0.6, 
                         palette='deep')  # using 'deep' palette for distinct colors
            
            # Add mean points with 95% confidence intervals using t-distribution
            means = df.mean()
            n = len(df)  # number of samples
            dof = n - 1  # degrees of freedom
            
            # Calculate confidence intervals using t-distribution
            conf_intervals = []
            for col in df.columns:
                std_err = df[col].std() / np.sqrt(n)
                t_val = stats.t.ppf(0.975, dof)  # 97.5th percentile for 95% CI
                ci = t_val * std_err
                conf_intervals.append(ci)
            
            plt.errorbar(x=range(len(means)), y=means, yerr=conf_intervals,
                        fmt='o', color='black', capsize=5, 
                        markersize=10, markerfacecolor='white',
                        label='Mean ± 95% CI of the mean (t-dist)')
            
            plt.xticks(range(len(df.columns)), df.columns, rotation=45, ha='right')
            plt.ylabel('Interval Score (lower is better)')
            plt.title(f'Distribution of Interval Scores Across CV Splits\n{dataset}')
            plt.legend()
            
            # Add grid for better readability
            plt.grid(True, axis='y', alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"{dataset}_score_distribution.png"))
            plt.close()


def custom_pred_vs_true_plot(mean_train, lower_train, upper_train, y_train,
                          mean_test, lower_test, upper_test, y_test,
                          title, save_path, alpha=0.1, figsize=(10, 6)):
    """Create a custom prediction vs true plot showing both training and test data.
    
    Args:
        mean_train, lower_train, upper_train: Predictions on training data
        y_train: True training values
        mean_test, lower_test, upper_test: Predictions on test data
        y_test: True test values
        title: Plot title
        save_path: Path to save the figure
        alpha: Confidence level
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    
    # Calculate overall min/max for plot limits
    all_y = np.concatenate([y_train, y_test])
    all_mean = np.concatenate([mean_train, mean_test])
    y_min, y_max = min(all_y.min(), all_mean.min()), max(all_y.max(), all_mean.max())
    
    # Plot perfect prediction line
    plt.plot([y_min, y_max], [y_min, y_max], 'k--', alpha=0.5, label='Perfect Prediction')
    
    # Plot training data with transparency
    plt.scatter(y_train, mean_train, c='blue', alpha=0.2, s=20, label='Training Points')
    
    # Plot test data
    plt.scatter(y_test, mean_test, c='red', alpha=0.8, s=30, label='Test Points')
    
    # Plot confidence intervals for test points
    for i in range(len(y_test)):
        plt.plot([y_test[i], y_test[i]], [lower_test[i], upper_test[i]], 
                 color='red', alpha=0.3, linewidth=1)
    
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title(title)
    plt.legend()
    
    # Add text with coverage information for test set
    coverage = np.mean((y_test >= lower_test) & (y_test <= upper_test))
    avg_width = np.mean(upper_test - lower_test)
    plt.text(0.05, 0.95, 
             f'Test Coverage: {coverage:.3f}\nTarget: {1-alpha:.3f}\nAvg Width: {avg_width:.3f}',
             transform=plt.gca().transAxes,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(save_path)
    plt.close()

def generate_cv_pred_vs_true_plots(results_dir, model_name, figsize=(10, 6)):
    """Generate prediction vs true plots for all CV splits of a specific model.
    
    Args:
        results_dir (str): Base directory containing the CV_models results
        model_name (str): Name of the model to generate plots for
        figsize (tuple): Figure size for the plots
    """
    fm = FileManager(results_dir)
    
    # For each dataset
    for dataset in DATASETS:
        print(f"\nGenerating CV prediction plots for {model_name} on {dataset}")
        
        cv_base_dir = os.path.join(results_dir, 'CV_models', dataset, model_name)
        if not os.path.exists(cv_base_dir):
            print(f"No CV results found for {model_name} on {dataset}")
            continue
        
        split_dirs = [d for d in os.listdir(cv_base_dir) if d.startswith('split_')]
        
        for split_dir in split_dirs:
            try:
                # Load the model and results for this split
                result_dict = fm.load_model(
                    MODEL_CLASSES[model_name], 
                    path=os.path.join(cv_base_dir, split_dir)
                )
                
                # Get model and data
                model = result_dict['model']
                X_train = result_dict['X_train']
                y_train = result_dict['y_train']
                X_test = result_dict['X_test']
                y_test = result_dict['y_test']
                
                # Get predictions for both training and test sets
                mean_train, lower_train, upper_train = model.predict(X_train)
                mean_test, lower_test, upper_test = model.predict(X_test)
                
                # Generate and save the prediction vs true plot
                save_path = os.path.join(cv_base_dir, split_dir, "pred_vs_true.png")
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                
                custom_pred_vs_true_plot(
                    mean_train, lower_train, upper_train, y_train,
                    mean_test, lower_test, upper_test, y_test,
                    title=f"Predictions vs True Values\n{model_name} on {dataset} ({split_dir})",
                    save_path=save_path,
                    alpha=model.alpha,
                    figsize=figsize
                )
                print(f"Generated plot for {split_dir}")
                
            except Exception as e:
                print(f"Error processing {split_dir}: {str(e)}")

def evaluate_single_model_cv(results_dir, target_model_name, reference_model_name="BBMM_GPR", save_number="", inverse_scaling=False, splits = None):
    """Evaluate a single model using the train-test splits from an existing model's CV results.
    
    Args:
        results_dir (str): Directory containing the saved models
        target_model_name (str): Name of the model to evaluate
        reference_model_name (str): Name of the model whose splits to use (default: "BBMM_GPR")
    """
    fm = FileManager(results_dir)
    
    # For each dataset
    for dataset in DATASETS:
        print(f"\nEvaluating {target_model_name} on {dataset}")
        
        # Get the directory containing the reference model's CV splits
        ref_cv_base_dir = os.path.join(results_dir, 'CV_models', dataset, reference_model_name)
        if not os.path.exists(ref_cv_base_dir):
            print(f"No CV results found for reference model {reference_model_name} on {dataset}")
            continue
        
        # Get list of splits
        split_dirs = [d for d in os.listdir(ref_cv_base_dir) if d.startswith('split_')]
        if splits is not None: 
            split_dirs = [d for d in split_dirs if d in splits]

        if not split_dirs:
            print(f"No splits found for {reference_model_name} on {dataset}")
            continue
        
        # Load the target model to get its configuration
        target_model_path = os.path.join(results_dir, "models", f"{dataset}_{target_model_name}")
        if not os.path.exists(target_model_path):
            print(f"No original model found for {target_model_name} on {dataset}")
            continue
            
        try:
            original_model_dict = fm.load_model(MODEL_CLASSES[target_model_name], path=target_model_path)
            target_model = original_model_dict['model']
            #target_model.learning_rate = target_model.learning_rate * 10 ** -2
            #target_model.scheduler_cls = CosineAnnealingLR
            #target_model.scheduler_kwargs = {"T_max": target_model.epochs}

        except Exception as e:
            print(f"Error loading original model: {str(e)}")
            continue
        
        # Process each split
        for split_dir in split_dirs:
            print(f"Processing {split_dir}...")
            try:
                # Load the reference split data
                ref_result_dict = fm.load_model(
                    MODEL_CLASSES[reference_model_name], 
                    path=os.path.join(ref_cv_base_dir, split_dir)
                )
                
                # Get the train-test split from reference model
                X_train = ref_result_dict['X_train'][:700]
                y_train = ref_result_dict['y_train'][:700]

                ### Apply inverse scaling
                if inverse_scaling:
                    y_train = 1/y_train

                X_test = ref_result_dict['X_test']
                y_test = ref_result_dict['y_test']

                # Train and evaluate target model
                print(target_model.optimizer_kwargs)
                target_model.fit(X_train, y_train)
                mean, lower, upper = target_model.predict(X_test)

                ### Apply inverse scaling for Cd
                if inverse_scaling: 
                    mean = 1 / mean 
                    lower = 1 / upper 
                    upper = 1 / lower 

                metrics = compute_all_metrics(mean, lower, upper, y_test, alpha=target_model.alpha)
                
                # Save results
                cv_save_dir = os.path.join(results_dir, 'CV_models', dataset, target_model_name, split_dir + "_" + save_number)
                os.makedirs(cv_save_dir, exist_ok=True)
                
                fm.save_model(
                    model=target_model,
                    path=cv_save_dir,
                    metrics=metrics,
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    y_test=y_test
                )
                print(f"Saved results for {split_dir} to {cv_save_dir}")
                print(f"Metrics: {metrics}")
                
            except Exception as e:
                print(f"Error processing split {split_dir}: {str(e)}")

def perform_scaled_comparison(results_dir, alpha=0.05, save_dir=None, figsize=(10, 6)): 
    fm = FileManager(results_dir)
    metric_results = defaultdict(lambda: defaultdict(list))
    os.makedirs(save_dir, exist_ok=True)
    for dataset in DATASETS: 
        for model_name in FINAL_COMP_MODELS: 
            cv_base_dir = os.path.join(results_dir, 'CV_models', dataset, model_name)
            if not os.path.exists(cv_base_dir):
                print(f"No CV results found for {model_name} on {dataset}")
                continue

            print(f"Evaluating {model_name}")
            split_dirs = [d for d in os.listdir(cv_base_dir) if d.startswith('split_')]
            for split_dir in split_dirs: 
                try:
                    # Load original model to get hyperparameters
                    model_path = os.path.join(cv_base_dir, split_dir)
                    result_dict = fm.load_model(MODEL_CLASSES[model_name], path=model_path)
                    model = result_dict['model']
                    scaler = StandardScaler()
                    y_train = result_dict['y_train']
                    scaler.fit(y_train.reshape(-1, 1))
                    X_test = result_dict['X_test']
                    y_test = result_dict['y_test']
                    y_test = scaler.transform(y_test.reshape(-1, 1))
                    # Train and evaluate
                    mean, lower, upper = model.predict(X_test)
                    mean = scaler.transform(mean.reshape(-1, 1))
                    lower = scaler.transform(lower.reshape(-1, 1))
                    upper = scaler.transform(upper.reshape(-1, 1))
                    metrics = compute_all_metrics(mean, lower, upper, y_test, alpha=model.alpha)
                    
                    # Store results
                    for metric_name, value in metrics.items():
                        metric_results[(dataset, metric_name)][model_name].append(value)

                except Exception as e: 
                    print(f"Error loading split {split_dir} for {model_name}: {str(e)}")
    
    # Plot results for this dataset
        better_direction = {
            "rmse": "lower is better",
            "nll_gaussian": "lower is better",
            "interval_score": "lower is better",
            "coverage": f"closer to {0.90} is better",
            "average_interval_width": "lower is better",
            "error_width_corr": "higher is better",
            "RMSCD": "lower is better",
            "RMSCD_under": "lower is better",
            "lowest_group_coverage": "higher is better"
        }
        
        for metric_name in metrics.keys():
            plt.figure(figsize=figsize)
            
            # Prepare data for plotting
            models = FINAL_COMP_MODELS
            results = metric_results[(dataset, metric_name)]
            df = pd.DataFrame(results)

            # Create and save a summary plot
            plt.figure(figsize=figsize)
            
            # Create vertical stripplot with jittered points
            sns.swarmplot(data=df, orient='v', size=10, alpha=0.6, 
                         palette='deep')  # using 'deep' palette for distinct colors
            

            # Add mean points with 95% confidence intervals using t-distribution
            means = df.mean()
            n = len(df)  # number of samples
            dof = n - 1  # degrees of freedom
            
            # Calculate confidence intervals using t-distribution
            conf_intervals = []
            for col in df.columns:
                std_err = df[col].std() / np.sqrt(n)
                t_val = stats.t.ppf(0.975, dof)  # 97.5th percentile for 95% CI
                ci = t_val * std_err
                conf_intervals.append(ci)
            
            plt.errorbar(x=range(len(means)), y=means, yerr=conf_intervals,
                        fmt='o', color='black', capsize=5, 
                        markersize=10, markerfacecolor='white',
                        label='Mean ± 95% CI of the mean (t-dist)')
            
            if metric_name == "coverage":
                plt.axhline(0.90, color='red', linestyle='--', label='Nominal')
                plt.legend()

            plt.xticks(range(len(df.columns)), df.columns, rotation=45, ha='right')
            plt.ylabel(f'{metric_name} ({better_direction[metric_name]})')
            plt.title(f'Distribution of scaled {metric_name} Across CV Splits\n{dataset}')
            plt.legend()
            
            # Add grid for better readability
            plt.grid(True, axis='y', alpha=0.3)
            
            plt.tight_layout()
            os.makedirs(os.path.join(save_dir, "scaled_results"), exist_ok=True)
            plt.savefig(os.path.join(save_dir, "scaled_results", f"{dataset}_{metric_name}_comparison.png"))
            plt.close()
            if metric_name == 'interval_score': 
                ranked_df = df.rank(axis=0, method="min")
                average_rank_per_row = ranked_df.mean(axis=1)
                result_df = pd.DataFrame({
                    "AverageRank": average_rank_per_row
                })
                print (result_df)
                print (result_df.std(axis=0))
                plt.figure(figsize=figsize)
                df_long = df.reset_index().melt(id_vars="index", var_name="Method", value_name="RMSE")

                plt.figure(figsize=figsize)
                sns.lineplot(data=df_long, x="Method", y="RMSE", hue="index", legend=False)
                plt.xticks(rotation=45)
                plt.title(f"{metric_name} distribution, where each line is a split: {dataset}")
                plt.tight_layout()
                plt.show()

def pairwise_comparison(results_dir, order, normalizing_factor="pct_imp", report = "SE"):    
    fm = FileManager(results_dir)

    # For each dataset
    dataset_avg_diffs = np.zeros((len(DATASETS), len(order) - 1))
    stds = np.zeros((len(DATASETS), len(order) - 1))
    df_total = pd.DataFrame()
    for i, dataset in enumerate(DATASETS):
        print(f"\nAnalyzing results for dataset: {dataset}")
        
        # Collect interval scores for each model and split
        scores = defaultdict(list)
        n_splits = None
        # Load results from each split
        for model_name in FINAL_COMP_MODELS:
            cv_base_dir = os.path.join(results_dir, 'CV_models', dataset, model_name)
            if not os.path.exists(cv_base_dir):
                print(f"No CV results found for {model_name} on {dataset}")
                continue
                
            split_dirs = [d for d in os.listdir(cv_base_dir) if d.startswith('split_')]
            n_splits = len(split_dirs)
            
            for split_dir in split_dirs:
                try:
                    result_dict = fm.load_model(MODEL_CLASSES[model_name], 
                                              path=os.path.join(cv_base_dir, split_dir))
                    scores[model_name].append(result_dict['metrics']['interval_score'])
                except Exception as e:
                    print(f"Error loading split {split_dir} for {model_name}: {str(e)}")
        
        if not scores:
            print("No results found for analysis")
            continue
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(scores)
        df['row_avg'] = df.sum(axis=1) / len(FINAL_COMP_MODELS)
        for j in range(dataset_avg_diffs.shape[1]): 
            worse = df[order[j]]
            best = df[order[j+1]]
            if normalizing_factor == "row_avg": 
                avg_diffs = ((worse - best) / (df_total['row_avg']))
            elif normalizing_factor == "pct_imp": 
                avg_diffs = ((worse - best) / worse)
            elif normalizing_factor == "avg_imp": 
                avg_diffs = (2 * (worse - best) / (worse + best))
            dataset_avg_diffs[i, j] = np.mean(avg_diffs)
            if report == "SE": 
                report_factor = np.sqrt(len(avg_diffs))
            elif report == "SD": 
                report_factor = 1
            stds[i, j] = np.std(avg_diffs) / report_factor
        df_total = pd.concat([df_total, df], ignore_index=True)

    print("Dataset Averages: ")
    print(dict(zip(DATASETS, dataset_avg_diffs)))
    
    print("stds")
    print(dict(zip(DATASETS, stds)))
    print("------------------")

    
    total_diffs = np.zeros(dataset_avg_diffs.shape[1])
    total_stds = np.zeros(dataset_avg_diffs.shape[1])
    for j in range(len(total_diffs)): 
        worse = df_total[order[j]]
        best = df_total[order[j+1]]
        if normalizing_factor == "row_avg": 
            avg_diffs = ((worse - best) / (df_total['row_avg']))
        elif normalizing_factor == "pct_imp": 
            avg_diffs = ((worse - best) / worse)
        elif normalizing_factor == "avg_imp": 
            avg_diffs = (2 * (worse - best) / (worse + best))
        total_diffs[j] = np.mean(avg_diffs)

        if report == "SE": 
            report_factor = np.sqrt(len(avg_diffs))
        elif report == "SD": 
            report_factor = 1
        total_stds[j] = np.std(avg_diffs) / report_factor

    print("Total Averages: ")
    print(total_diffs)

    print("stds")
    print(total_stds)
    print("----------------")

def evaluate_conditional_coverage_metrics(results_dir, target_model_name, save_number="", splits = None): 
    fm = FileManager(results_dir)
    
    # For each dataset
    for dataset in DATASETS:
        print(f"\nEvaluating {target_model_name} on {dataset}")
        
        # Get the directory containing the reference model's CV splits
        ref_cv_base_dir = os.path.join(results_dir, 'CV_models', dataset, target_model_name)
        if not os.path.exists(ref_cv_base_dir):
            print(f"No CV results found for reference model {target_model_name} on {dataset}")
            continue
        
        # Get list of splits
        split_dirs = [d for d in os.listdir(ref_cv_base_dir) if d.startswith('split_')]
        if splits is not None: 
            split_dirs = [d for d in split_dirs if d in splits]

        if not split_dirs:
            print(f"No splits found for {reference_model_name} on {dataset}")
            continue
        
        # Load the target model to get its configuration
        target_model_path = os.path.join(results_dir, "models", f"{dataset}_{target_model_name}")
        if not os.path.exists(target_model_path):
            print(f"No original model found for {target_model_name} on {dataset}")
            continue
            
        try:
            original_model_dict = fm.load_model(MODEL_CLASSES[target_model_name], path=target_model_path)
            target_model = original_model_dict['model']
            #target_model.learning_rate = target_model.learning_rate * 10 ** -2
            #target_model.scheduler_cls = CosineAnnealingLR
            #target_model.scheduler_kwargs = {"T_max": target_model.epochs}

        except Exception as e:
            print(f"Error loading original model: {str(e)}")
            continue
        
        # Process each split
        for split_dir in split_dirs:
            print(f"Processing {split_dir}...")
            try:
                # Load the reference split data
                ref_result_dict = fm.load_model(
                    MODEL_CLASSES[reference_model_name], 
                    path=os.path.join(ref_cv_base_dir, split_dir)
                )
                
                # Get the train-test split from reference model
                X_train = ref_result_dict['X_train'][:700]
                y_train = ref_result_dict['y_train'][:700]

                ### Apply inverse scaling
                if inverse_scaling:
                    y_train = 1/y_train

                X_test = ref_result_dict['X_test']
                y_test = ref_result_dict['y_test']

                # Train and evaluate target model
                print(target_model.optimizer_kwargs)
                target_model.fit(X_train, y_train)
                mean, lower, upper = target_model.predict(X_test)

                ### Apply inverse scaling for Cd
                if inverse_scaling: 
                    mean = 1 / mean 
                    lower = 1 / upper 
                    upper = 1 / lower 

                metrics = compute_all_metrics(mean, lower, upper, y_test, alpha=target_model.alpha)
                
                # Save results
                cv_save_dir = os.path.join(results_dir, 'CV_models', dataset, target_model_name, split_dir + "_" + save_number)
                os.makedirs(cv_save_dir, exist_ok=True)
                
                fm.save_model(
                    model=target_model,
                    path=cv_save_dir,
                    metrics=metrics,
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    y_test=y_test
                )
                print(f"Saved results for {split_dir} to {cv_save_dir}")
                print(f"Metrics: {metrics}")
                
            except Exception as e:
                print(f"Error processing split {split_dir}: {str(e)}")

if __name__ == "__main__":
    results_dir = 'results/airfoil_results_lr_sched/p_1000_runs'
    #perform_scaled_comparison(results_dir, save_dir= os.path.join(results_dir, "cv_comparisons"))
    #evaluate_single_model_cv(results_dir, "TempScaleDeepEns", reference_model_name="BBMM_GPR", save_number="")
    #evaluate_single_model_cv(results_dir, "BBMM_GPR", reference_model_name="KFoldQuantileRegression", save_number="") #, save_number="1", splits=['split_23'])
    #load_and_compare_results(results_dir=results_dir, 
    #                         generate_calibration=False, 
    #                         generate_comparisons=True, 
    #                         generate_bootstrap_comparisons=False,
    #                         generate_pred_vs_true=False, 
    #                         generate_tuning_curves=False)

    # First run cross-validation if needed
    #evaluate_final_models(
    #    results_dir=results_dir,
    #    n_splits=30,
    #    test_size=0.3,
    #)
    
    #plot_cv_results(results_dir, save_dir='results/airfoil_results_lr_sched/p_1000_runs/comp_plots')
    # Then analyze the results
    #analyze_cv_results(results_dir)
    #plot_cv_results(results_dir, save_dir = os.path.join(results_dir, "comp_plots", "subset"))
   # order_1000 = ["BBMM_GPR", "MonteCarloDropout", "ConformalQuantileRegression", "KFoldQuantileRegression", "NormalizedConformalEnsembles", "TempScaleDeepEns"]
    #order_250 = ["ConformalQuantileRegression", "MonteCarloDropout", "NormalizedConformalEnsembles", "BBMM_GPR", "KFoldQuantileRegression", "TempScaleDeepEns"]
    #order_cqr = ["ConformalQuantileRegression", "KFoldQuantileRegression"]
    #order_temp_scale = ["NormalizedConformalEnsembles", "ConformalQuantileRegression"]
    #pairwise_comparison(results_dir, order_1000, normalizing_factor="avg_imp", report="SE")
    # Generate prediction vs true plots for CV splits of BBMM_GPR
    #import torch
    #print(torch.load(os.path.join(results_dir, "Cm_nf_4D_TempScaleDeepEns", "temperature.pt")))
    #generate_cv_pred_vs_true_plots(results_dir, "TempScaleDeepEns_scale")
