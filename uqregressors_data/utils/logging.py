import logging
from pathlib import Path
import os

LOGGING_CONFIG = {
    "print": True
}

def set_logging_config(print=True): 
    """
    Sets global logging printing configuration. 

    Args: 
        print (bool): If False, disables printing to the terminal for all future Logger instances
    """
    LOGGING_CONFIG["print"] = print

try:
    import wandb

    _wandb_available = True
except ImportError:
    _wandb_available = False


class Logger:
    """
    Base Logging class.

    Args: 
        use_wandb (bool): Whether to use weights and biases for logging (Experimental feature, not validated yet).
        project_name (str): The logger project name.
        run_name (str): The logger run name for a given training run. 
        config (dict): Dictionary of relevant training parameters, only used if weights and biases is used.
        name (str): Name of the logger.
    """
    def __init__(self, use_wandb=False, project_name=None, run_name=None, config=None, name=None):
        self.use_wandb = use_wandb and _wandb_available
        self.logs = []

        if self.use_wandb:
            wandb.init(
                project=project_name or "default_project",
                name=run_name,
                config=config or {},
            )
            self.wandb = wandb
        else:
            self.logger = logging.getLogger(name or f"Logger-{os.getpid()}")
            self.logger.setLevel(logging.INFO)

            if LOGGING_CONFIG["print"] and not self.logger.handlers:
                ch = logging.StreamHandler()
                formatter = logging.Formatter("[%(name)s] %(message)s")
                ch.setFormatter(formatter)
                self.logger.addHandler(ch)

    def log(self, data: dict):
        """
        Writes a dictionary to a stored internal log.
        """
        if self.use_wandb:
            self.wandb.log(data)
        else:
            msg = ", ".join(f"{k}={v}" for k, v in data.items())
            self.logs.append(msg)
            self.logger.info(msg)

    def save_to_file(self, path, subdir="logs", idx=0, name=""): 
        """
        Saves logs to the logs subdirectory when model.save is called.
        """
        log_dir = Path(path) / subdir 
        log_dir.mkdir(parents=True, exist_ok=True)
        with open(log_dir / f"{name}_{str(idx)}.log", "w", encoding="utf-8") as f: 
            f.write("\n".join(self.logs))


    def finish(self):
        """
        Finish method for weights and biases logging.
        """
        if self.use_wandb:
            self.wandb.finish()