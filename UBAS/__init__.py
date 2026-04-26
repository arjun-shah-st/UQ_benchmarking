import os
import warnings

warnings.filterwarnings('ignore', message="Environment variable TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD detected")
warnings.filterwarnings('ignore', message="WARNING: The predictions are ill-sorted.")
os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'
