"""
base_sampler.py
==================
Parent class for all samplers: functions which sample from data generators, typically for the purpose of constructing
a surrogate model
"""
from typing import Tuple, Iterable
from numpy.typing import NDArray
import numpy as np
from rich.progress import track
from UBAS.generators.input_generator import InputGenerator
from UBAS.utils.evaluation import evaluate_performance
import os
import pandas as pd
from dataclasses import asdict
import pickle
from matplotlib import pyplot as plt

class BaseSampler:
    """Base class for data samplers that samples uniformly within the given bounds"""
    def __init__(self, directory, dimension, surrogate, generator, bounds, n_iterations, n_batch_points,
                 initial_inputs, initial_targets, test_inputs=None, test_targets=None, scaler=None,
                 intermediate_training=False, plotter=None, save_interval=5, mean_relative_error=False,
                 adaptive_batch_size=False):
        """
        Sampler Initialization

        Parameters
        ----------
        directory : str
            Path to save files in
        dimension : int
            Dimension of the problem
        surrogate : Object
            A surrogate model which predicts confidence intervals with the scikit-learn 'fit' and 'predict' API
        generator : BaseGenerator
            A generator model that generates target values from the inputs
        bounds : NDArray
            The bounds within which to sample. Should be an NDArray of shape (2, n_dimensions) where
            the first element of the 0th axis are the lower bounds and the second element are the upper
            bounds for each dimension.
        n_iterations : int
            Number of sampling iterations to perform
        n_batch_points : int
            Number of points to add to the training set per sampling iteration
        initial_inputs : NDArray
            Starting inputs, should be of shape (n_initial_samples, dimension)
        initial_targets : NDArray
            Initial targets, should be of shape (n_initial_samples, )
        test_inputs : NDArray default=None
            Inputs of a validation set to test the surrogate model against. If not provided, no evaluation will occur
            Should be of shape (n_test_samples, dimension)
        test_targets : NDArray default=None
            Targets of a validation set to test the surrogate model against. If not provided, no evaluation will occur
            Should be of shape (n_test_samples, )
        scaler : MinMaxScaler default=None
            If None, data will not be reverse scaled before being sent to the generator. If a scaler is passed, the
            new_x coordinates will first be unscaled before being sent to the generator. In this case, the scaler should
            encode the transformation from the bounds meant to be passed to the generator to the bounds passed to the
            sampler.
        intermediate_training : Bool default=False
            If True, the surrogate model will be retrained after every sampling iteration. If False, the surrogate
            model gets trained once at the end of sampling
        plotter : Plotter default=None
            Not yet implemented
        save_interval : int default=5
            Not yet implemented
        mean_relative_error : Bool, default=False
            If True, mean relative error metrics will be computed (as long as test_inputs and test_targets are
            provided). If false, the mean relative error will be none in the error evaluation objects.
        adaptive_batch_size : Bool, default=False
            If True, the batch size will be automatically set to a third of the number of training points used
        """
        self.directory = directory
        self.dimension = dimension
        self.surrogate = surrogate
        self.generator = generator
        self.bounds = bounds
        self.n_iterations = n_iterations
        self.n_batch_points = n_batch_points
        self.initial_inputs = initial_inputs
        self.initial_targets = initial_targets
        self.n_initial_points = self.initial_inputs.shape[0]
        self.test_inputs = test_inputs
        self.test_targets = test_targets
        self.scaler = scaler
        self.intermediate_training = intermediate_training
        self.plotter = plotter
        self.plotter.filename = self.directory
        self.save_interval = save_interval
        self.mean_relative_error = mean_relative_error
        self.adaptive_batch_size = adaptive_batch_size

        os.makedirs(self.directory, exist_ok=True)

        self.SAMPLER_DATA_PATH = os.path.join(self.directory, "sampler_data.pkl")
        self.PERF_DATA_PATH = os.path.join(self.directory, "performance_data.json")
        self.TRACK_DATA_PATH = os.path.join(self.directory, "tracked_values.json")

        # Initialize sampler
        self.sampler = InputGenerator(bounds, dimension)

        # Allocate arrays
        x_exact = np.zeros((self.n_iterations * self.n_batch_points + self.n_initial_points, self.dimension))
        y_exact = np.zeros(x_exact.shape[0])

        if intermediate_training is True:
            self.model_performance = np.empty(self.n_iterations + 1, dtype=object)
        else:
            self.model_performance = np.empty(1, dtype=object)

        # Populate arrays with initial values
        x_exact[:self.n_initial_points] = self.initial_inputs
        y_exact[:self.n_initial_points] = self.initial_targets

        # Create instance variables with the exact arrays:
        self.x_exact = x_exact
        self.y_exact = y_exact

        # Perform checks on test_inputs and test_targets and set evaluation flag accordingly
        if test_inputs is not None:
            self.evaluate_surrogates = True
        else:
            self.evaluate_surrogates = False

        # Set iteration counter
        self._iteration = 0

        self.fit_kwargs = None
        self.plot_kwargs = None
        self.predict_kwargs = None

    def sample(self, track_values=["mean_width"], fit_kwargs=None, plot_kwargs=None, predict_kwargs=None):
        """
        Method to perform iterations of sample_step while handling re-training and evaluation the surrogate. Usually
        does not need to be overriden in subclasses.

        Parameters
        ----------
        track_values : list
            A list of error metrics to be plotted as the sampler selects new points. Currently supported values are:
            mse, mean_relative_error, mean_width, max_width, max_absolute_error, coverage
        fit_kwargs : dict
            Extra keyword parameters to be passed to the fit method of the surrogate
        plot_kwargs : dict
            Extra keyword parameters to be passed to the plotting method
        predict_kwargs : dict
            Extra keyword parameters to be passed to the predictor
        """
        n_iterations = self.n_iterations
        n_initial_points = self.n_initial_points
        n_batch_points = self.n_batch_points

        if track_values is not None and track_values is not []:
            dynamic_plotting = True
            plt.ion()

            fig, ax_tup = plt.subplots(len(track_values), 1, sharex=True, figsize=(10, 10))
            plt.tight_layout()
            if not isinstance(ax_tup, Iterable):
                ax_tup = [ax_tup]
            ax_tup[len(ax_tup)-1].set_xlabel("Number of Samples")
            ax_tup[0].set_title("History: " + self.directory)
            graph_objects = []
            tracked_variables = {"n_samples": []}

            for i, track_value in enumerate(track_values):
                if track_value in ["mse", "mean_relative_error", "max_absolute_error"]:
                    ax_tup[i].semilogy()
                ax_tup[i].set_ylabel(track_value)
                graph_objects.append(ax_tup[i].plot([0], [0])[0])
                tracked_variables[track_value] = []
            plt.ioff()
        else:
            dynamic_plotting = False
            tracked_variables = None

        if fit_kwargs is None:
            fit_kwargs = {}

        if plot_kwargs is None:
            plot_kwargs = {}

        if predict_kwargs is None:
            predict_kwargs = {}

        self.fit_kwargs = fit_kwargs
        self.plot_kwargs = plot_kwargs
        self.predict_kwargs = predict_kwargs

        # Run sampling loop:
        for i in track(range(n_iterations), description=f"Running Main Sampling Loop: {self.directory}"):
            start_index = n_initial_points + i * n_batch_points

            if self.adaptive_batch_size is True:
                fit_kwargs["batch_size"] = int(start_index / 3)
                self.fit_kwargs = fit_kwargs

            self._iteration = i

            # Re-train surrogate
            if self.intermediate_training is True:
                print("Training model...")
                self.surrogate.fit(self.x_exact[:start_index], self.y_exact[:start_index], **fit_kwargs)

                # if test data is available, evaluate the surrogate
                if self.evaluate_surrogates is True:
                    print("Evaluating model...")
                    y_preds, y_bounds = self.predict(self.test_inputs, **predict_kwargs)
                    eval_obj = evaluate_performance(y_preds, y_bounds, self.test_targets, self.mean_relative_error)
                    self.model_performance[i] = eval_obj

                    if dynamic_plotting is True:
                        tracked_variables["n_samples"].append(start_index)
                        plt.ion()
                        for j, track_value in enumerate(track_values):
                            if track_value in list(asdict(self.model_performance[i]).keys()):
                                tracked_variables[track_value].append(asdict(self.model_performance[i])[track_value])
                            elif track_value in list(self.__dict__.keys()):
                                tracked_variables[track_value].append(self.__dict__[track_value])
                            else:
                                raise UserWarning(f"Could not find requested tracking variable: {track_value}")

                            graph_objects[j].set_xdata(tracked_variables["n_samples"])
                            graph_objects[j].set_ydata(tracked_variables[track_value])

                            ax_tup[j].relim()
                            ax_tup[j].autoscale_view()
                            plt.tight_layout()

                            fig.canvas.draw()
                            fig.canvas.flush_events()
                            plt.ioff()
                    if (i + 1) % self.save_interval == 0:
                        self.save_model_performance(self.model_performance[:i+1], tracked_variables)
            # Sample new points with sampling_step
            print("Generating new inputs...")
            new_x = self.sampling_step(n_batch_points)

            if self.scaler is None:
                new_x, new_y = self.generator.generate(new_x)
            else:
                unscaled_x = self.scaler.inverse_transform(new_x)
                unscaled_x, new_y = self.generator.generate(unscaled_x)
                new_x = self.scaler.transform(unscaled_x)

            self.x_exact[start_index:start_index+n_batch_points] = new_x
            self.y_exact[start_index:start_index+n_batch_points] = new_y

            # Plot data
            if self.plotter is not None:
                print("Plotting Data...")
                if (i + 1) % self.plotter.plotting_interval == 0:
                    y_preds, y_bounds = self.predict(self.x_exact[:start_index+n_batch_points], **predict_kwargs)
                    if self.scaler is None:
                        self.plotter.generate_plots(i+1, self.x_exact[:start_index+n_batch_points], new_x, y_preds,
                                                    y_bounds, self.y_exact[:start_index+n_batch_points], new_y,
                                                    **plot_kwargs)
                    else:
                        self.plotter.generate_plots(i + 1, self.scaler.inverse_transform(self.x_exact[:start_index +
                                                                                                       n_batch_points]),
                                                    self.scaler.inverse_transform(new_x), y_preds,
                                                    y_bounds, self.y_exact[:start_index + n_batch_points], new_y,
                                                    **plot_kwargs)
            # Save Sampler State
            if (i + 1) % self.save_interval == 0:
                self.save_sampler()

        print("Re-training surrogate on full training data...")

        # Re-train surrogate on full training data
        self.surrogate.fit(self.x_exact, self.y_exact, **fit_kwargs)

        print("Evaluating surrogate on full training data...")
        # Re-evaluate_surrogate on full training data
        if self.evaluate_surrogates is True:
            y_preds, y_bounds = self.predict(self.test_inputs, **predict_kwargs)
            eval_obj = evaluate_performance(y_preds, y_bounds, self.test_targets, self.mean_relative_error)
            if self.intermediate_training is True:
                self.model_performance[n_iterations] = eval_obj
            else:
                self.model_performance[0] = eval_obj

            if dynamic_plotting is True:
                tracked_variables["n_samples"].append(n_initial_points + n_iterations * n_batch_points)
                for j, track_value in enumerate(track_values):
                    if track_value in list(asdict(self.model_performance[n_iterations]).keys()):
                        tracked_variables[track_value].append(asdict(self.model_performance[n_iterations])[track_value])
                    elif track_value in list(self.__dict__.keys()):
                        tracked_variables[track_value].append(self.__dict__[track_value])
                    else:
                        raise UserWarning(f"Could not find requested tracking variable: {track_value}")
        # Save data:
        print("Saving data...")
        self.save_model_performance(self.model_performance, tracked_variables)
        self.save_sampler()
        print("Sampling has finished")

    def sampling_step(self, n_batch_points) -> NDArray:
        """
        One step of uniform sampling. This is usually overriden by subclasses

        Parameters
        ----------
        n_batch_points
            The number of points to sample with one call to sampling_step

        Returns
        -------
        NDArray
            A set of new training points of shape (n_batch_points, dimension)
        """
        new_x = self.sampler.uniformly_sample(n_batch_points)
        return new_x

    def predict(self, inputs, **kwargs):
        """
        A helper method to transform different types of predictions arising from surrogate models into a usable form.

        Parameters
        ----------
        inputs : NDArray
            Inputs for which to predict values

        **kwargs
            Keyword arguments to the predict method of the surrogate

        Returns
        -------
        Y_preds : NDArray
            Array of shape (length of inputs, )
            The mean predictions of the surrogate model

        Y_bounds : NDArray
            Array of shape (length of inputs, 2)
            The lower and upper bound predictions of the surrogate model.
        """
        pred_results = self.surrogate.predict(inputs, **kwargs)
        if isinstance(pred_results, Tuple):
            y_preds, y_bounds = pred_results
        else:
            y_bounds = pred_results
            y_preds = np.mean(pred_results, axis=1)
        return y_preds, y_bounds

    def save_model_performance(self, model_performance_list, tracked_values=None):
        """
        Method to save the model performance outputs in an easily accessible file every several iterations

        Parameters
        ----------
        model_performance_list : list(EvalObject)
            A list of eval objects describing the model performance
        tracked_values : dict
            A dictionary containing the tracked values that are plotted during runtime
        """
        dict_to_save = {}
        for i, perf in enumerate(model_performance_list):
            if i == 0:
                for key, value in asdict(perf).items():
                    dict_to_save[key] = [value]
            else:
                for key, value in asdict(perf).items():
                    dict_to_save[key].append(value)

        if tracked_values is not None:
            for key, value in tracked_values.items():
                if key not in list(dict_to_save.keys()):
                    dict_to_save[key] = value
        else:
            dict_to_save["n_samples"] = list(self.n_initial_points + np.arange(model_performance_list.shape[0]) *
                                    self.n_batch_points)

        df = pd.DataFrame(dict_to_save)
        json = df.to_json(orient='split', compression='infer')

        with open(self.PERF_DATA_PATH, 'w') as f:
            f.write(json)

    def save_sampler(self):
        """
        Serializes and stores the state of the sampler which can then be re-loaded
        """

        with open(self.SAMPLER_DATA_PATH, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load_sampler(filename):
        """
        Loads and returns a previously stored sampler

        Parameters
        ----------
        filename : str
            The full path to the sampler to be loaded

        Returns
        -------
        BaseSampler
            The Sampler to be loaded
        """
        with open(filename, 'rb') as f:
            sampler = pickle.load(f)

        return sampler
