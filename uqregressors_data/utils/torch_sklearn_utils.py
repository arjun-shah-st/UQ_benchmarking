"""
torch_sklearn_utils
-------------------
A collection of sklearn utility functions refactored to work with pytorch tensors. 

The key functions are: 
    - TorchStandardScaler (class)
    - TorchKFold (class)
    - train_test_split (function)

!!! warning 
    TorchKFold returns the indices of each K-Fold, while train_test_split returns the values in each split.
"""

import torch
import numpy as np 

class TorchStandardScaler:
    """
    Standardized scaling to 0 mean values with unit variance.

    Attributes: 
        mean_ (float): The mean of the data, subtracted from the data during scaling. 
        std_ (float): The standard deviation of the data, by which the data is divided during scaling.
    """
    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X):
        """
        Fits the standard scaler. 
        
        Args: 
            X (torch.Tensor): data to be scaled of shape (n_samples, n_features).

        Returns: 
            (TorchStandardScaler): the scaler with updated mean_ and std_ attributes. 
        """
        self.mean_ = X.mean(dim=0, keepdim=True)
        self.std_ = X.std(dim=0, unbiased=False, keepdim=True)
        # Avoid division by zero
        self.std_[self.std_ < 1e-8] = 1.0
        return self

    def transform(self, X):
        """
        Transforms the standard scaler based on the attributes obtained with the fit method. 

        Args: 
            X (torch.Tensor): data to be scaled of shape (n_samples, n_features).

        Returns: 
            (torch.Tensor): The scaled data
        """
        return (X - self.mean_) / self.std_
    
    def fit_transform(self, X): 
        """
        Performs the fit and transforms the data. A combination of the fit and transform methods.

        Args: 
            X (torch.Tensor): data to be scaled of shape (n_samples, n_features).

        Returns: 
            (torch.Tensor): The scaled data
        """
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X_scaled):
        """
        Transforms scaled data back to the original scale. 

        Args: 
            X_scaled (torch.Tensor): scaled data of shape (n_samples, n_features).

        Returns: 
            (torch.Tensor): The unscaled data. 
        """
        return X_scaled * self.std_ + self.mean_
    
def train_test_split(X, y, test_size=0.2, device="cpu", random_state=None, shuffle=True):
    """
    Split arrays or tensors into training and test sets.
    
    Args:
        X (array-like or torch.Tensor): Features to be split. 
        y (array-like or torch.Tensor): Targets to be split. 
        test_size (float): Proportion of the dataset to include in the test split (between 0 and 1).
        random_state (int or None): Controls the shuffling for reproducibility.
        shuffle (bool): Whether or not to shuffle the data before splitting.
    
    Returns:
        (Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]): X_train, X_test, y_train, y_test; same type as inputs
    """
    # Convert to numpy for easy indexing
    if isinstance(X, torch.Tensor):
        X_np = X.cpu().numpy()
        is_tensor = True
    else:
        X_np = np.asarray(X)
        is_tensor = False

    if isinstance(y, torch.Tensor):
        y_np = y.cpu().numpy()
    else:
        y_np = np.asarray(y)

    # Check dimensions
    if X_np.shape[0] != y_np.shape[0]:
        raise ValueError(f"X and y must have the same number of samples. Got {X_np.shape[0]} and {y_np.shape[0]}.")

    n_samples = X_np.shape[0]
    n_test = int(n_samples * test_size)

    if random_state is not None:
        rng = np.random.default_rng(random_state)
    else:
        rng = np.random.default_rng()

    indices = np.arange(n_samples)
    if shuffle:
        rng.shuffle(indices)

    test_indices = indices[:n_test]
    train_indices = indices[n_test:]

    X_train, X_test = X_np[train_indices], X_np[test_indices]
    y_train, y_test = y_np[train_indices], y_np[test_indices]

    if is_tensor:
        X_train = torch.tensor(X_train, dtype=X.dtype, device=device)
        X_test = torch.tensor(X_test, dtype=X.dtype, device=device)
        y_train = torch.tensor(y_train, dtype=y.dtype, device=device)
        y_test = torch.tensor(y_test, dtype=y.dtype, device=device)

    return X_train, X_test, y_train, y_test

class TorchKFold:
    """
    A class meant to split the data into K-folds for conformalization or cross validation. 

    Args: 
        n_splits (int): The number of folds for data splitting.
        shuffle (bool): Whether to shuffle the data before splitting. 
        random_state (int or None): Controls shuffling for reproducibility.
    """
    def __init__(self, n_splits=5, shuffle=False, random_state=None):
        if n_splits < 2:
            raise ValueError("n_splits must be at least 2.")
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X):
        """
        Yield train/test indices for each fold.

        Args:
            X (torch.Tensor, np.ndarray, or list): Input data with shape (n_samples, ...)

        Yields:
            (tuple[torch.LongTensor, torch.LongTensor]): train_idx, val_idx; the indices of the training and validation sets for each of the splits. 
        """
        if isinstance(X, torch.Tensor):
            n_samples = X.shape[0]
        else:
            X = np.asarray(X)
            n_samples = len(X)

        indices = np.arange(n_samples)

        if self.shuffle:
            rng = np.random.default_rng(self.random_state)
            rng.shuffle(indices)

        fold_sizes = np.full(self.n_splits, n_samples // self.n_splits, dtype=int)
        fold_sizes[:n_samples % self.n_splits] += 1
        current = 0

        for fold_size in fold_sizes:
            val_idx = indices[current:current + fold_size]
            train_idx = np.concatenate([indices[:current], indices[current + fold_size:]])
            current += fold_size

            yield (
                torch.from_numpy(train_idx).long(),
                torch.from_numpy(val_idx).long()
            )