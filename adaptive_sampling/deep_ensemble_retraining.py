from UBAS.samplers.badge_sampler import BADGESampler
from uqregressors.bayesian.deep_ens import DeepEnsembleRegressor
from uqregressors.metrics.metrics import compute_all_metrics

path = "adaptive_sampling/SHIFTWING_CD/DE_RAND"
sampler = BADGESampler.load_sampler(path, DeepEnsembleRegressor)

test_inputs = sampler.test_inputs
test_outputs = sampler.test_outputs 

iteration = 8

train_inputs = sampler.x_exact[:sampler.n_initial_points + sampler.n_batch_points * iteration]
train_outputs = sampler.y_exact[:sampler.n_initial_points + sampler.n_batch_points * iteration]

input_scaler = sampler.input_scaler 
output_scaler = sampler.output_scaler 

scaled_train_inputs = input_scaler.transform(train_inputs)
scaled_train_outputs = output_scaler.transform(train_outputs.reshape(-1, 1))

sampler.regressor.fit(scaled_train_inputs, scaled_train_outputs)

scaled_test_inputs = input_scaler.transform(test_inputs)
mean, lower, upper = sampler.regressor.predict(scaled_test_inputs)

mean = output_scaler.inverse_transform(mean.reshape(-1, 1)).ravel()
lower = output_scaler.inverse_transform(lower.reshape(-1, 1)).ravel()
upper = output_scaler.inverse_transform(upper.reshape(-1, 1)).ravel()

metrics = compute_all_metrics(mean, lower, upper, test_outputs, alpha=sampler.regressor.alpha)
print(metrics)