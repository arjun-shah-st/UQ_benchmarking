   results = []

    for d in dims:
            print(f"\n=== Running experiments for dimension d={d} ===")

            # Generate fixed test set for fairness
            X_test = rng.uniform(bounds[0], bounds[1], size=(n_test, d))
            theta_true = rng.uniform(-1, 1, size=(d, 1))
            y_test = X_test @ theta_true + rng.normal(0, 0.01, size=(n_test, 1))

            n_init = n_init_factor * d ** dim_scaling_factor 
            b = b_factor * d ** dim_scaling_factor

            for sigma_th in sigmas:
                print(f"--- Noise level = {sigma_th} ---")

                for trial in range(n_trials):
                    # Generate new initial design for each trial
                    X_init = rng.uniform(bounds[0], bounds[1], size=(n_init, d))

                    # ==== BASELINE: RANDOM SAMPLING ====
                    X_random = X_init.copy()
                    # add b random points
                    X_random_add = rng.uniform(bounds[0], bounds[1], size=(b, d))
                    X_random = np.vstack([X_random, X_random_add])

                    # generate outputs
                    epsilon = rng.normal(0, (sigma_th * np.linalg.norm(theta_true)) ** 2, size=(n_init + b, 1))
                    y_random = X_random @ theta_true + epsilon

                    # fit LS
                    theta_hat_random = np.linalg.lstsq(X_random, y_random, rcond=None)[0]
                    y_pred_random = X_test @ theta_hat_random
                    mse_random = np.mean((y_pred_random - y_test)**2)

                    results.append({
                        "dimension": d,
                        "noise": sigma_th,
                        "criterion": "Random",
                        "trial": trial,
                        "mse": mse_random
                    })

                    # ==== DESIGN CRITERIA ====
                    for crit in criteria:

                        out = select_optimal_design(
                            X_init=X_init,
                            X_test=X_test,
                            y_test=y_test,
                            theta=theta_true,
                            bounds=bounds,
                            criterion=crit,
                            sigma_threshold=sigma_th,
                            d=d,
                            b=b,
                            sobol_factor=sobol_factor,
                            refine_steps=refine_steps,
                            random_seed=rng.integers(1e9)
                        )
                         
                        results.append({
                            "dimension": d,
                            "noise": sigma_th,
                            "criterion": crit,
                            "trial": trial,
                            "mse": out["mse_test"]
                        })
                        print(f"Criterion={crit}, trial={trial}, mse={out['mse_test']}")

    df_results = pd.DataFrame(results)
    print("\n===== EXPERIMENTS COMPLETE =====\n")
    return df_results