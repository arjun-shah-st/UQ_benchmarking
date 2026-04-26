import pickle
import os
import numpy as np

with open(os.path.join("D:", os.sep, "UBAS", "projects", "2D_central_peak", "Adaptive", "sampler_data.pkl"), 'rb') as f:
    model = pickle.load(f)

print(model.surrogate.predict(np.array([[0.5, 0.5], [0.2, 0.2], [0.3, 0.3]])))
