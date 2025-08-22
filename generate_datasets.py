import os
import json
import numpy as np
from datetime import datetime
from pathlib import Path
from synthetic_dataset_generator import generate_synthetic_dataset  # Replace with actual filename
import pickle

# --- CONFIGURABLE PARAMETERS ---
NUM_DATASETS = 200
OUTPUT_DIR = "generated_datasets"

# Parameter ranges
PARAM_RANGES = {
    "n_samples": (250, 1000),
    "input_dim": (2, 20),
    "complexity_level": (1, 5),
    "noise_scale": (0.01, 0.3),
    "heteroscedasticity": (0.0, 1.0),
    "noise_type": ["gaussian", "laplace", "uniform"],
}
# --------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)

def sample_params(seed=None):
    rng = np.random.default_rng(seed)
    input_dim = rng.integers(*PARAM_RANGES["input_dim"])
    return {
        "n_samples": rng.integers(*PARAM_RANGES["n_samples"]),
        "input_dim": input_dim,
        "intrinsic_dim": input_dim,
        "complexity_level": rng.integers(*PARAM_RANGES["complexity_level"]),
        "noise_type": rng.choice(PARAM_RANGES["noise_type"]),
        "noise_scale": rng.uniform(*PARAM_RANGES["noise_scale"]),
        "heteroscedasticity": rng.uniform(*PARAM_RANGES["heteroscedasticity"]),
        "random_state": rng.integers(1e6)
    }

def save_dataset(index, X_train, X_test, y_train, y_test, meta):
    folder = Path(OUTPUT_DIR) / f"dataset_{index:03d}"
    folder.mkdir(parents=True, exist_ok=True)

    with open(folder / "data.pkl", "wb") as f: 
        pickle.dump([X_train, X_test, y_train, y_test], f)

    with open(folder / "meta.json", "w") as f:
        json.dump(convert_json_serializable(meta), f, indent=4)

    print(f"Saved dataset {index} to {folder}")

def convert_json_serializable(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    elif isinstance(obj, (list, tuple)):
        return [convert_json_serializable(i) for i in obj]
    elif isinstance(obj, dict):
        return {k: convert_json_serializable(v) for k, v in obj.items()}
    else:
        return obj


def main():
    for i in range(NUM_DATASETS):
        params = sample_params(seed=i)
        try:
            X_train, X_test, y_train, y_test, meta = generate_synthetic_dataset(**params)
            meta.update(params)  # Add generation parameters to metadata
            save_dataset(i, X_train, X_test, y_train, y_test, meta)
        except Exception as e:
            print(f"Error on dataset {i}: {e}")

if __name__ == "__main__":
    main()