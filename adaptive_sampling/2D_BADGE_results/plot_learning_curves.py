import matplotlib.pyplot as plt
import numpy as np
import scienceplots 

plt.style.use('science')  # Use scienceplots styling

# Dummy example data: replace with your actual RMSE data
x = np.arange(10, 31, 5)  # Number of labeled points (e.g., after 0, 5, 10, ... iterations)

rmse_grubs = [0.22, 0.09, 0.07, 0.0474, 0.0388]
rmse_random = [0.22, 0.19, 0.152, 0.154, 0.0677]
rmse_graddiv = [0.185, 0.119, 0.0949, 0.0814, 0.0703]
rmse_uncertainty = [0.200, 0.115, 0.0502, 0.0585, 0.0357]

# Plot
plt.figure(figsize=(6, 4))


plt.plot(x, rmse_random, label='Random', marker='s', linestyle = '--', alpha=0.75)
plt.plot(x, rmse_graddiv, label='Gradient Diversity Only', marker='^', linestyle='--', alpha=0.75)
plt.plot(x, rmse_uncertainty, label='Sequential Max Uncertainty', marker='D', linestyle='--', alpha=0.75)
plt.plot(x, rmse_grubs, label='GRUBS', marker='o')

plt.xlabel('Number of Samples')
plt.ylabel('RMSE')
plt.title('Learning Curves on 2D Central Peak Problem')
plt.legend()
plt.xticks(np.arange(10, 31, 5))
plt.grid(True)
plt.tight_layout()
plt.savefig('adaptive_sampling/2D_BADGE_results/learning_curves_grubs.png')  # Save as vector graphic
plt.show()