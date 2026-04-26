"""
evaluation.py
=============
A collection of methods used to evaluate surrogate models
"""
from dataclasses import dataclass
import numpy as np
from mapie.metrics import regression_coverage_score


@dataclass
class EvalObject:
    """
    Dataclass to store evaluation metrics

    Attributes
    ----------
    mse : Union(float, NoneType) default=None
        The mean squared error between the predictions and true values
        None if the true y-values were not provided during evaluation

    mean_relative_error : Union(float, NoneType) default=None
        The average of the absolute error divided by the true function value
        Meaningless if any of the true function values are very close to zero

    max_absolute_error : Union(float, NoneType) default=None
        The maximum absolute error between the predictions and true values
        None if the true y-values were not provided during evaluation

    mean_width : Union(float, NoneType) default=None
        The average distance between the upper and lower bounds of the predicted values
        Defaults to None, but should be updated during evaluation

    max_width : Union(float, NoneType) default=None
        The maximum distance between the upper and lower bounds of the predicted values
        Defaults to None, but should be updated during evaluation

    coverage : Union(float, NoneType) default=None
        The proportion of values, between 0 and 1 where the true values lie within the predicted bounds
        None if the true values are not provided during evaluation
    """
    mse: float = None
    mean_relative_error: float = None
    max_absolute_error: float = None
    mean_width: float = None
    max_width: float = None
    coverage: float = None

def evaluate_performance(y_preds, y_bounds, y_true=None, include_mre=False) -> EvalObject:
    """
    Evaluation of model performance. If y_true is included, width, error, and coverage metrics are calculated.
    Otherwise, only width metrics are calculated.

    Parameters
    ----------
    y_preds : NDArray
        The mean value predictions of the surrogate model
        Should be a 1D array with length (n_evaluation_points)

    y_bounds : NDArray
        The lower and upper bound predictions of the surrogate model
        Should be a 2D array with the same zeroth axis length as y_preds.
        The first element should be the lower bounds, and the second element should be the upper bounds

    y_true : Union(NDArray, NoneType) default=None
        The true values from a validation set not seen during training
        Should be a 1D array with the same length as y_preds

    include_mre: Bool default=False
        If true, the mean relative error will be calculated.
        Defaults to false in order to avoid zero division in cases where the true function values are close to zero

    Returns
    -------
    EvalObject
        Contains the evaluation metrics for the given dataset

    Raises
    ------
    ValueError
        If the shape of y_bounds is not (n_evaluation_points, 2)

    ValueError
        If the shape of y_true is not equal to the shape of y_preds
    """
    if y_bounds.shape != (y_preds.shape[0], 2):
        raise ValueError(f"""Shape of y_bounds was {y_bounds.shape}, while the length of y_preds is 
                        {y_preds.shape[0]}. The shape of y_bounds should be: (n_eval_points, 2)""")

    mean_width = np.float64(np.mean(y_bounds[:, 1] - y_bounds[:, 0]))
    max_width = np.float64(np.max(y_bounds[:, 1] - y_bounds[:, 0]))

    if y_true is None:
        return EvalObject(mean_width=mean_width, max_width=max_width)

    else:
        if y_true.shape != y_preds.shape:
            raise ValueError(f"""The shapes of y_preds and y_true should be equal. The shape of y_preds is 
                            {y_preds.shape}, and the shape of y_true is {y_true.shape}""")

        mse = np.float64(np.mean((y_true - y_preds) ** 2))
        if include_mre:
            mean_relative_error = np.float64(np.mean(np.abs((y_true - y_preds) / y_true)))
        else:
            mean_relative_error = None
        max_absolute_error = np.float64(np.max(np.abs(y_true-y_preds)))
        coverage = np.float64(regression_coverage_score(y_true, y_bounds[:, 0], y_bounds[:, 1]))

        return EvalObject(mean_width=mean_width, mean_relative_error=mean_relative_error, max_width=max_width, mse=mse,
                          max_absolute_error=max_absolute_error, coverage=coverage)
