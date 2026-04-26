"""
This is a module to be used as a reference for building other modules
"""
import numpy as np
import quantnn as q
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, _fit_context
from sklearn.metrics import euclidean_distances
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted


class QuantNNRegressor(BaseEstimator):
    """A template estimator to be used as a reference implementation.

    For more information regarding how to build your own estimator, read more
    in the :ref:`User Guide <user_guide>`.

    Parameters
    ----------
    demo_param : str, default='demo_param'
        A parameter used for demonstration of how to pass and store parameters.

    Attributes
    ----------
    is_fitted_ : bool
        A boolean indicating whether the estimator has been fitted.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

    Examples
    --------
    >>> from skltemplate import TemplateEstimator
    >>> import numpy as np
    >>> X = np.arange(100).reshape(100, 1)
    >>> y = np.zeros((100, ))
    >>> estimator = TemplateEstimator()
    >>> estimator.fit(X, y)
    TemplateEstimator()
    """

    # This is a dictionary allowing to define the type of parameters.
    # It used to validate parameter within the `_fit_context` decorator.
    _parameter_constraints = {
        "demo_param": [str],
    }

    def __init__(self, quantiles=[0.05, 0.95], layers=4, neurons=128, activation="relu", n_epochs=15, batch_size=32):
        self.quantiles = quantiles
        self.layers = layers
        self.neurons = neurons
        self.activation = activation
        self.n_epochs = n_epochs
        self.batch_size = batch_size

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y, **fit_params):
        """A reference implementation of a fitting function.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.

        y : array-like, shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in
            regression).

        Returns
        -------
        self : object
            Returns self.
        """

        # `_validate_data` is defined in the `BaseEstimator` class.
        # It allows to:
        # - run different checks on the input data;
        # - define some attributes associated to the input data: `n_features_in_` and
        #   `feature_names_in_`.
        n_epochs = fit_params.pop("n_epochs", None)
        if n_epochs is not None:
            self.n_epochs = n_epochs
        batch_size = fit_params.pop("batch_size", None)
        if batch_size is not None:
            self.batch_size = batch_size

        X, y = self._validate_data(X, y, accept_sparse=True)
        model = (self.layers, self.neurons, self.activation)
        self.qrnn = q.QRNN(self.quantiles, n_inputs=X.shape[1], model=model)
        self.qrnn.is_fitted = True
        results = self.qrnn.train((X, y), n_epochs=self.n_epochs, batch_size=self.batch_size, **fit_params)
        self.is_fitted_ = True
        # `fit` should always return `self`
        return self

    def predict(self, X):
        """A reference implementation of a predicting function.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.

        Returns
        -------
        y : ndarray, shape (n_samples,)
            Returns an array of ones.
        """
        # Check if fit had been called
        check_is_fitted(self)
        # We need to set reset=False because we don't want to overwrite `n_features_in_`
        # `feature_names_in_` but only check that the shape is consistent.
        y_pred = self.qrnn.predict(X)
        return y_pred

    def score(self, X, y_true):
        y_preds = self.predict(X)
        y_val = np.mean(y_preds, axis=1)
        return -np.linalg.norm(y_val - y_true) ** 2 / y_val.shape[0] * 10 ** 5
