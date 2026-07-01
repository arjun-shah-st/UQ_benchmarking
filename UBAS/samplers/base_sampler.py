from pathlib import Path
import os
import numpy as np 
import matplotlib.pyplot as plt 
from typing import Iterable
from rich.progress import track
from uqregressors.metrics.metrics import compute_all_metrics
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pickle 
import json

class BaseSampler: 
    """
    Base class for sampling datasets
    """
    def __init__(self, directory, regressor, generator, bounds, n_iterations, n_batch_points, 
                 initial_inputs, initial_targets, test_inputs=None, test_targets=None, plotter=None, 
                 save_interval=1, random_seed = 42): 
        """
        Initialization of the base class
        """
        self.directory = Path(directory)
        self.regressor = regressor 
        self.generator = generator 
        self.bounds = bounds 
        self.n_iterations = n_iterations 
        self.n_batch_points = n_batch_points 
        self.initial_inputs = initial_inputs 
        self.initial_targets = initial_targets
        self.n_initial_points = len(initial_inputs)
        self.dimension = self.initial_inputs.shape[1]
        self.test_inputs = test_inputs 
        self.scaled_test_inputs = None 
        self.test_outputs = test_targets 
        self.input_scaler = None 
        self.output_scaler = None
        self.plotter = plotter
        self.save_interval = save_interval
        self.random_seed = random_seed

        os.makedirs(self.directory, exist_ok=True) 

        self.PERF_DATA_PATH = os.path.join(self.directory, "performance_data.json")
        self.PRED_DIR = os.path.join(self.directory, "predictions")
        os.makedirs(self.PRED_DIR, exist_ok=True)

        x_exact = np.zeros((self.n_iterations * self.n_batch_points + self.n_initial_points, self.dimension))
        y_exact = np.zeros(x_exact.shape[0])

        self.model_performance = np.empty(self.n_iterations+1, dtype=object)

        x_exact[:self.n_initial_points] = self.initial_inputs 
        y_exact[:self.n_initial_points] = self.initial_targets 

        self.x_exact = x_exact 
        self.y_exact = y_exact 

        self.input_scaler = MinMaxScaler()
        self.input_scaler.fit(bounds)
        self.output_scaler = StandardScaler()
        self.output_scaler.fit(self.initial_targets.reshape(-1, 1))

        if test_inputs is not None: 
            self.evaluate = True 
            self.scaled_test_inputs = self.input_scaler.transform(self.test_inputs)
        else: 
            self.evaluate = False 

        self._iteration = 0 
        self.plot_kwargs = None 
        self.rng = np.random.default_rng(self.random_seed)

    def sample(self, track_values=["interval_score"]): 
        """
        Runs a sampling loop
        """
        n_iterations = self.n_iterations
        n_initial_points = self.n_initial_points 
        n_batch_points = self.n_batch_points

        if track_values is not None: 
            dynamic_plotting=True 
            plt.ion() 


            fig, ax_tup = plt.subplots(len(track_values), 1, sharex=True, figsize=(10, 10))
            plt.tight_layout() 
            if not isinstance(ax_tup, Iterable): 
                ax_tup = [ax_tup]

            ax_tup[len(ax_tup)-1].set_xlabel("Number of Samples")
            ax_tup[0].set_title("History: " + str(self.directory))

            graph_objects = [] 
            tracked_variables={"n_samples": []}

            for i, track_value in enumerate(track_values): 
                ax_tup[i].set_ylabel(track_value)
                graph_objects.append(ax_tup[i].plot([0], [0])[0])
                tracked_variables[track_value] = []
            plt.ioff() 
        
        else:
            dynamic_plotting = False 
            tracked_variables = None 

        for i in track(range(n_iterations), description=f"Running Main Sampling Loop: {self.directory}"): 
            start_index = n_initial_points + i * n_batch_points 

            self._iteration = i 
            self.regressor.fit(self.input_scaler.transform(self.x_exact[:start_index]), self.output_scaler.transform(self.y_exact[:start_index].reshape(-1, 1))) 

            if self.evaluate is True: 
                print("Evaluating Model: ")
                mean, lower, upper = self.regressor.predict(self.scaled_test_inputs)
                mean = self.output_scaler.inverse_transform(mean.reshape(-1, 1)).ravel()
                lower = self.output_scaler.inverse_transform(lower.reshape(-1, 1)).ravel()
                upper = self.output_scaler.inverse_transform(upper.reshape(-1, 1)).ravel() 
                metrics = compute_all_metrics(mean, lower, upper, self.test_outputs, alpha=self.regressor.alpha)
                metrics["n_samples"] = start_index
                self.model_performance[i] = metrics
                self.save_iteration_predictions(i, mean, lower, upper) 

                if dynamic_plotting is True: 
                    tracked_variables["n_samples"].append(start_index)
                    plt.ion() 
                    for j, track_value in enumerate(track_values):
                        if track_value in list(self.model_performance[i].keys()): 
                            tracked_variables[track_value].append(self.model_performance[i][track_value])
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
                    self.save_model_performance(self.model_performance[:i+1])

            print("Generating new inputs: ") 
            new_x = self.sample_step()

            unscaled_x = self.input_scaler.inverse_transform(new_x)
            new_x, new_y = self.generator.generate(unscaled_x)

            self.x_exact[start_index:start_index+n_batch_points] = new_x 
            self.y_exact[start_index:start_index+n_batch_points] = new_y 

            if self.plotter is not None: 
                print("Plotting Data: ")
                if (i + 1) % self.plotter.plotting_interval == 0: 
                    mean, lower, upper = self.regressor.predict(self.input_scaler.transform(self.x_exact[:start_index+n_batch_points]))
                    mean = self.output_scaler.inverse_transform(mean.reshape(-1, 1)).ravel()
                    lower = self.output_scaler.inverse_transform(lower.reshape(-1, 1)).ravel()
                    upper = self.output_scaler.inverse_transform(upper.reshape(-1, 1)).ravel()
                    self.plotter.generate_plots(i+1, self.x_exact[:start_index + n_batch_points], 
                                                new_x, mean, lower, upper, self.y_exact[:start_index+n_batch_points], new_y)
                    
            if (i + 1) % self.save_interval == 0: 
                print("Saving Sampler: Overriding previous state")
                self.save_sampler()

        print("Re-training regressor on full training data: ")
        
        self.regressor.fit(self.input_scaler.transform(self.x_exact), self.output_scaler.transform(self.y_exact.reshape(-1, 1)))

        if self.evaluate: 
            print ("Evaluating regressor: ")
            mean, lower, upper = self.regressor.predict(self.scaled_test_inputs)
            mean = self.output_scaler.inverse_transform(mean.reshape(-1, 1)).ravel() 
            lower = self.output_scaler.inverse_transform(lower.reshape(-1, 1)).ravel() 
            upper = self.output_scaler.inverse_transform(upper.reshape(-1, 1)).ravel() 
            metrics = compute_all_metrics(mean, lower, upper, self.test_outputs, alpha=self.regressor.alpha)
            metrics["n_samples"] = n_iterations * n_batch_points + n_initial_points 
            self.model_performance[n_iterations] = metrics
            self.save_iteration_predictions(n_iterations, mean, lower, upper) 
            if dynamic_plotting is True: 
                    tracked_variables["n_samples"].append(start_index)
                    for j, track_value in enumerate(track_values):
                        if track_value in list(self.model_performance[i].keys()): 
                            tracked_variables[track_value].append(self.model_performance[i][track_value])
                        elif track_value in list(self.__dict__.keys()): 
                            tracked_variables[track_value].append(self.__dict__[track_value]) 
                        else: 
                            raise UserWarning(f"Could not find requested tracking variable: {track_value}")

            print("Saving data: ")
            self.save_model_performance(self.model_performance, tracked_variables)
            self.save_sampler()
            print("Sampling has finished successfully")

    def sample_step(self): 
        """
        A single sampling step
        """
        
        scaled_x = self.rng.random((self.n_batch_points, len(self.bounds[1])))
        return scaled_x

    def save_iteration_predictions(self, iteration, mean, lower, upper):
        """
        Save predictions for a specific iteration to a separate file
        """
        pred_data = {
            "mean": mean.astype(float),
            "lower": lower.astype(float),
            "upper": upper.astype(float)
        }
        pred_file = os.path.join(self.PRED_DIR, f"iteration_{iteration}_predictions.pkl")
        with open(pred_file, 'wb') as f:
            pickle.dump(pred_data, f)

    def save_model_performance(self, model_performance_list, tracked_values=None): 
        dict_to_save = {} 
        for i, perf in enumerate(model_performance_list): 
            if i==0: 
                for key, value in perf.items(): 
                    dict_to_save[key] = [value]
            else: 
                for key, value in perf.items(): 
                    dict_to_save[key].append(value)

        with open(self.PERF_DATA_PATH, 'w') as f: 
            json.dump(dict_to_save, f, indent=4)

    def save_sampler(self): 
        
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super().default(obj)
        path = self.directory
        path.mkdir(parents=True, exist_ok=True)

        # Save serializable attributes (excluding non-picklable ones)
        config = {
            "bounds": self.bounds,
            "n_iterations": self.n_iterations,
            "n_batch_points": self.n_batch_points,
            "n_initial_points": self.n_initial_points,
            "dimension": self.dimension,
            "evaluate": self.evaluate,
            "random_seed": self.random_seed,
            "save_interval": self.save_interval,
            "plot_kwargs": self.plot_kwargs,
            "_iteration": self._iteration,
        }

        # Save input/output data
        with open(path / "sampler_arrays.pkl", "wb") as f:
            pickle.dump({
                "x_exact": self.x_exact,
                "y_exact": self.y_exact,
                "initial_inputs": self.initial_inputs,
                "initial_targets": self.initial_targets,
                "test_inputs": self.test_inputs,
                "test_outputs": self.test_outputs,
                "scaled_test_inputs": self.scaled_test_inputs,
            }, f)

        # Save the config
        with open(path / "sampler_config.json", "w") as f:
            json.dump(config, f, indent=4, cls=NumpyEncoder)

        # Save the regressor (using its own save method)
        regressor_path = path / "regressor"
        self.regressor.save(regressor_path)

        # Save generator if picklable (optional)
        try:
            with open(path / "generator.pkl", "wb") as f:
                pickle.dump(self.generator, f)
        except Exception:
            print("Warning: Generator could not be pickled and was skipped.")

    # Save scalers (optional)
        with open(path / "scalers.pkl", "wb") as f:
            pickle.dump({
                "input_scaler": self.input_scaler,
                "output_scaler": self.output_scaler
            }, f)


    @classmethod
    def load_sampler(cls, path, regressor_cls): 
        path = Path(path)

        # Load config
        with open(path / "sampler_config.json", "r") as f:
            config = json.load(f)

        # Load data arrays
        with open(path / "sampler_arrays.pkl", "rb") as f:
            arrays = pickle.load(f)

        # Load performance data
        with open(path / "performance_data.json", "r") as f:
            perf_data = json.load(f)

        # Load the regressor
        regressor = regressor_cls.load(path / "regressor")

        # Load the generator
        try:
            with open(path / "generator.pkl", "rb") as f:
                generator = pickle.load(f)
        except Exception:
            generator = None
            print("Warning: Generator could not be loaded.")

        # Load scalers
        with open(path / "scalers.pkl", "rb") as f:
            scalers = pickle.load(f)

        # Create the object
        obj = cls(
            directory=path,
            regressor=regressor,
            generator=generator,
            bounds=config["bounds"],
            n_iterations=config["n_iterations"],
            n_batch_points=config["n_batch_points"],
            initial_inputs=arrays["initial_inputs"],
            initial_targets=arrays["initial_targets"],
            test_inputs=arrays["test_inputs"],
            test_targets=arrays["test_outputs"],
            random_seed=config["random_seed"],
            plotter=None  # Optional: skip or reassign externally
        )

        obj.x_exact = arrays["x_exact"]
        obj.y_exact = arrays["y_exact"]
        obj.scaled_test_inputs = arrays["scaled_test_inputs"]
        obj.input_scaler = scalers["input_scaler"]
        obj.output_scaler = scalers["output_scaler"]
        obj.evaluate = config["evaluate"]
        obj.plot_kwargs = config["plot_kwargs"]
        obj._iteration = config["_iteration"]
        obj.model_performance = perf_data

        return obj
