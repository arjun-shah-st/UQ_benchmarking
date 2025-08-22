import numpy as np
from scipy.stats import norm
import torch

def validate_inputs(mean, lower, upper, y_true, alpha=0.5):
    """
    Ensure inputs are converted to 1D numpy arrays and alpha is a float in (0, 1) for use in computing metrics.
    
    Args: 
        mean (Union[torch.Tensor, np.ndarray]): The mean predictions to compute metrics for, should be able to be flattened to one dimension.
        lower (Union[torch.Tensor, np.ndarray]): The lower bound predictions to compute metrics for, should be the same shape as mean. 
        upper (Union[torch.Tensor, np.ndarray]): The upper bound predictions to compute metrics for, should be the same shape as mean. 
        y_true (Union[torch.Tensor, np.ndarray]): The targets to compute metrics with, should be the same shape as mean.
        alpha (float): The desired confidence level, if relevannt, should be a float between 0 and 1. 
    """

    def to_1d_numpy(x):
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        x = np.asarray(x)
        if x.ndim != 1:
            x = x.flatten()
        return x

    mean = to_1d_numpy(mean).astype(np.float64)
    lower = to_1d_numpy(lower).astype(np.float64)
    upper = to_1d_numpy(upper).astype(np.float64)
    y_true = to_1d_numpy(y_true).astype(np.float64)

    if not (0 < float(alpha) < 1):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    length = len(mean)
    if not (len(lower) == len(upper) == len(y_true) == length):
        raise ValueError("All input arrays must be of the same length.")

    return mean, lower, upper, y_true, float(alpha)


def rmse(mean, y_true, **kwargs):
    """
    Computes the root mean square error of the predictions compared to the targets.

    Args: 
        mean (Union[np.ndarray, torch.Tensor]): The mean predictions made by the model, should be able to be flattened to 1 dimension.
        y_true (Union[np.ndarray, torch.Tensor]): The targets to compare against, should be the same shape as mean. 
    
    Returns: 
        (float): Scalar root mean squared error.
    """
    mean, _, _, y_true, _ = validate_inputs(mean, mean, mean, y_true)
    return np.sqrt(np.mean((mean - y_true) ** 2))


def coverage(lower, upper, y_true, **kwargs):
    """
    Computes the coverage as a float between 0 and 1. 

    Args:
        lower (Union[np.ndarray, torch.Tensor]): The lower bound predictions made by the model, should be able to be flattened to 1 dimension.
        upper (Union[np.ndarray, torch.Tensor]): The upper bound predictions made by the model, should be the same shape as lower.
        y_true (Union[np.ndarray, torch.Tensor]): The targets to compare against, should be the same shape as lower.

    Returns: 
        (float): Coverage as a scalar between 0.0 and 1.0.
    """
    _, lower, upper, y_true, _ = validate_inputs(lower, lower, upper, y_true)
    covered = (y_true >= lower) & (y_true <= upper)
    return np.mean(covered)


def average_interval_width(lower, upper, **kwargs):
    """
    Computes the average interval width (distance between the predicted upper and lower bounds). 

    Args:
        lower (Union[np.ndarray, torch.Tensor]): The lower bound predictions made by the model, should be able to be flattened to 1 dimension.
        upper (Union[np.ndarray, torch.Tensor]): The upper bound predictions made by the model, should be the same shape as lower.

    Returns: 
        (float): Average distance between the upper and lower bound.
    """
    _, lower, upper, _, _ = validate_inputs(lower, lower, upper, lower)
    return np.mean(upper - lower)


def interval_score(lower, upper, y_true, alpha, **kwargs):
    """
    Computes the interval score as given in [Gneiting and Raftery, 2007](https://sites.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf).

    Args: 
        lower (Union[np.ndarray, torch.Tensor]): The lower bound predictions made by the model, should be able to be flattened to 1 dimension.
        upper (Union[np.ndarray, torch.Tensor]): The upper bound predictions made by the model, should be the same shape as lower.
        y_true (Union[np.ndarray, torch.Tensor]): The targets to compare against, should be the same shape as lower.
        alpha (float): 1 - confidence, should be a float between 0 and 1. 

    Returns: 
        (float): Interval score.
    """
    _, lower, upper, y_true, alpha = validate_inputs(lower, lower, upper, y_true)
    width = upper - lower
    penalty_lower = (2 / alpha) * (lower - y_true) * (y_true < lower)
    penalty_upper = (2 / alpha) * (y_true - upper) * (y_true > upper)
    return np.mean(width + penalty_lower + penalty_upper)


def nll_gaussian(mean, lower, upper, y_true, alpha, **kwargs):
    """
    Computes the average negative log likelihood of the data given the predictions and assuming a Gaussian distribution of predictions.

    Args: 
        lower (Union[np.ndarray, torch.Tensor]): The lower bound predictions made by the model, should be able to be flattened to 1 dimension.
        upper (Union[np.ndarray, torch.Tensor]): The upper bound predictions made by the model, should be the same shape as lower.
        y_true (Union[np.ndarray, torch.Tensor]): The targets to compare against, should be the same shape as lower.
        alpha (float): 1 - confidence, should be a float between 0 and 1. 

    Returns: 
        (float): Average negative log likelihood of the data given the predictions.
    """
    mean, lower, upper, y_true, alpha = validate_inputs(mean, lower, upper, y_true, alpha)
    z = norm.ppf(1 - alpha / 2)
    std = (upper - lower) / (2 * z)
    std = np.clip(std, 1e-6, None)

    log_likelihoods = -0.5 * np.log(2 * np.pi * std**2) - 0.5 * ((y_true - mean) / std) ** 2
    return -np.mean(log_likelihoods)

def error_width_corr(mean, lower, upper, y_true, **kwargs): 
    """
    Computes the Pearson correlation coefficient between true errors and the predicted interval width.

    Args: 
        lower (Union[np.ndarray, torch.Tensor]): The lower bound predictions made by the model, should be able to be flattened to 1 dimension.
        upper (Union[np.ndarray, torch.Tensor]): The upper bound predictions made by the model, should be the same shape as lower.
        y_true (Union[np.ndarray, torch.Tensor]): The targets to compare against, should be the same shape as lower.

    Returns: 
        (float): Correlation coefficient between residuals and predicted interval width, bounded in [-1, 1].
    """
    mean, lower, upper, y_true, _ = validate_inputs(mean, lower, upper, y_true)
    width = upper - lower 
    res = np.abs(mean - y_true)
    corr = np.corrcoef(width, res)[0, 1]
    return corr

def group_conditional_coverage(lower, upper, y_true, n_bins = 10): 
    """
    Divides the outputs into approximately equal bins, and computes the coverage in each bin. Returns a dictionary containing the mean of the 
    output in each bin, and the coverage in each bin. 

    Args: 
        lower (Union[np.ndarray, torch.Tensor]): The lower bound predictions made by the model, should be able to be flattened to 1 dimension.
        upper (Union[np.ndarray, torch.Tensor]): The upper bound predictions made by the model, should be the same shape as lower.
        y_true (Union[np.ndarray, torch.Tensor]): The targets to compare against, should be the same shape as lower.
        n_bins (int): The number of bins to compute conditional coverage for.

    Returns: 
        (dict): dictionary containing the following keys: 

            y_true_bin_means (np.ndarray): One dimensional array of the mean of the outputs within each bin.

            bin_coverages (np.ndarray): One dimensional array of the coverage of the predictions within each bin.
    """
    _, lower, upper, y_true, alpha = validate_inputs(lower, lower, upper, y_true)
    coverage_mask = (y_true > lower) & (y_true < upper)
    sort_ind = np.argsort(y_true)
    y_true_sort = y_true[sort_ind]
    coverage_mask_sort = coverage_mask[sort_ind]
    split_y_true = np.array_split(y_true_sort, n_bins)
    split_coverage_mask = np.array_split(coverage_mask_sort, n_bins)
    bin_means = [np.mean(bin) for bin in split_y_true]
    bin_coverages = [np.mean(bin) for bin in split_coverage_mask]
    return {"y_true_bin_means": np.array(bin_means), 
            "bin_coverages": np.array(bin_coverages)}

def RMSCD(lower, upper, y_true, alpha, n_bins=10): 
    """
    Computes the Root Mean Square Coverage Deviation (RMSCD) evaluated over a given number of bins (see group_conditional_coverage).
    
    Args: 
        lower (Union[np.ndarray, torch.Tensor]): The lower bound predictions made by the model, should be able to be flattened to 1 dimension.
        upper (Union[np.ndarray, torch.Tensor]): The upper bound predictions made by the model, should be the same shape as lower.
        y_true (Union[np.ndarray, torch.Tensor]): The targets to compare against, should be the same shape as lower.
        alpha (float): 1 - confidence, should be a float between 0 and 1. 
        n_bins (int): The number of bins to divide the outputs into.

    Returns: 
        (float): The root mean square coverage deviation from alpha.
    """
    _, lower, upper, y_true, alpha = validate_inputs(lower, lower, upper, y_true, alpha)
    gcc = group_conditional_coverage(lower, upper, y_true, n_bins)
    return np.sqrt(np.mean((gcc["bin_coverages"] - (1-alpha)) ** 2))

def RMSCD_under(lower, upper, y_true, alpha, n_bins=10):
    """
    Computes the Root Mean Square Coverage Deviation (RMSCD) evaluated only over bins which do not meet nominal coverage (see RMSCD, group_conditional_coverage).

    Args: 
        lower (Union[np.ndarray, torch.Tensor]): The lower bound predictions made by the model, should be able to be flattened to 1 dimension.
        upper (Union[np.ndarray, torch.Tensor]): The upper bound predictions made by the model, should be the same shape as lower.
        y_true (Union[np.ndarray, torch.Tensor]): The targets to compare against, should be the same shape as lower.
        alpha (float): 1 - confidence, should be a float between 0 and 1. 
        n_bins (int): The number of bins to divide the outputs into.

    Returns: 
        (float): The root mean square coverage deviation from alpha over bins which do not meet nominal coverage.
    """
    _, lower, upper, y_true, alpha = validate_inputs(lower, lower, upper, y_true, alpha)
    gcc = group_conditional_coverage(lower, upper, y_true, n_bins)
    miscovered_bins = gcc["bin_coverages"][gcc["bin_coverages"] < (1-alpha)]
    if len(miscovered_bins) == 0: 
        rmscd = 0.0
    else: 
        rmscd = np.sqrt(np.mean((miscovered_bins - (1-alpha)) ** 2))
    return rmscd

def lowest_group_coverage(lower, upper, y_true, n_bins=10): 
    """
    Computes the coverage of the bin with lowest coverage when the outputs are divided into several bins and coverage is evaluated conditional on each bin. 

    Args: 
        lower (Union[np.ndarray, torch.Tensor]): The lower bound predictions made by the model, should be able to be flattened to 1 dimension.
        upper (Union[np.ndarray, torch.Tensor]): The upper bound predictions made by the model, should be the same shape as lower.
        y_true (Union[np.ndarray, torch.Tensor]): The targets to compare against, should be the same shape as lower.
        n_bins (int): The number of bins to divide the outputs into.

    Returns: 
        (float): The coverage of the least covered bin of outputs, float between 0 and 1. 
    """
    _, lower, upper, y_true, alpha = validate_inputs(lower, lower, upper, y_true)
    gcc = group_conditional_coverage(lower, upper, y_true, n_bins)
    return np.min(gcc["bin_coverages"])

def compute_all_metrics(mean, lower, upper, y_true, alpha, n_bins=10, excluded_metrics=["group_conditional_coverage"]):
    """
    Compute all standard uncertainty quantification metrics and return as a dictionary.
    Computes the Root Mean Square Coverage Deviation (RMSCD) evaluated over a given number of bins. 

    Args: 
        mean (Union[torch.Tensor, np.ndarray]): The mean predictions to compute metrics for, should be able to be flattened to one dimension.
        lower (Union[np.ndarray, torch.Tensor]): The lower bound predictions made by the model, should be able to be flattened to 1 dimension. 
        upper (Union[np.ndarray, torch.Tensor]): The upper bound predictions made by the model, should be the same shape as mean.
        y_true (Union[np.ndarray, torch.Tensor]): The targets to compare against, should be the same shape as mean.
        alpha (float): 1 - confidence, should be a float between 0 and 1. 
        n_bins (int): The number of bins to divide the outputs into for conditional coverage metrics. 
        excluded_metrics (list): The key of any metrics to exclude from being returned.

    Returns: 
        (dict): dictionary containing the following metrics, except those named in excluded_metrics.

            rmse (float): Root Mean Square Error. 
        
            coverage (float): Marginal coverage. 

            average interval width (float): Average distance between upper and lower bound predictions.

            interval_score (float): Interval score between predictions and data. 

            nll_gaussian (float): Average Negative Log Likelihood of data given predictions under Gaussian assumption.

            error_width_corr (float): Pearson correlation coefficient between true errors and predicted interval width. 

            group_conditional_coverage (dict): Dictionary containing the mean and coverage of each bin when the outputs are split between several bins.

            RMSCD (float): Root mean square coverage deviation between the coverage conditional on output bin and the nominal coverage.

            RMSCD_under (float): Root mean square coverage deviation for all bins which undercover compared to nominal coverage.

            lowest_group_coverage (float): The lowest coverage of any bin into which the outputs were binned. 
    """

    mean, lower, upper, y_true, alpha = validate_inputs(mean, lower, upper, y_true, alpha)

    metrics_dict = {
        "rmse": rmse(mean, y_true, alpha=alpha),
        "coverage": coverage(lower, upper, y_true, alpha=alpha),
        "average_interval_width": average_interval_width(lower, upper, alpha=alpha),
        "interval_score": interval_score(lower, upper, y_true, alpha),
        "nll_gaussian": nll_gaussian(mean, lower, upper, y_true, alpha),
        "error_width_corr": error_width_corr(mean, lower, upper, y_true), 
        "group_conditional_coverage": group_conditional_coverage(lower, upper, y_true, n_bins),
        "RMSCD": RMSCD(lower, upper, y_true, alpha, n_bins),
        "RMSCD_under": RMSCD_under(lower, upper, y_true, alpha, n_bins),
        "lowest_group_coverage": lowest_group_coverage(lower, upper, y_true, n_bins)
    }

    return_dict = {}
    for metric, value in metrics_dict.items(): 
        if metric not in excluded_metrics: 
            return_dict[metric] = value 

    return return_dict