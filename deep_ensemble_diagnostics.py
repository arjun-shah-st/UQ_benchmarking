import os 
from collections import defaultdict 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
from uqregressors.bayesian.deep_ens import DeepEnsembleRegressor 
from uqregressors.conformal.k_fold_cqr import KFoldCQR
from scipy import stats as st
import pandas as pd 
from uqregressors.utils.file_manager import FileManager
from uqregressors.metrics.metrics import compute_all_metrics
from uqregressors.utils.data_loader import validate_X_input, validate_and_prepare_inputs
from torch.utils.data import TensorDataset, DataLoader 
from torch.optim.lr_scheduler import CosineAnnealingLR

DATASETS = ['Cd_nf_4D']

def extract_model(results_dir, dataset, split, model="DeepEnsemble", model_cls = DeepEnsembleRegressor): 
    """
    Loads model and training dataset from save pipeline in compare_model_results
    """
    fm = FileManager(results_dir)
    model_path = os.path.join(results_dir, "CV_models", f"{dataset}", model, split)
    result_dict = fm.load_model(model_cls, path=model_path) 

    model = result_dict['model']
    X_train = result_dict['X_train']
    X_test = result_dict['X_test']
    y_train = result_dict['y_train']
    y_test = result_dict['y_test']
    metrics = result_dict['metrics']

    print(f"loaded {split} from {dataset}, metrics: ")
    print(metrics)

    return model, X_train, X_test, y_train, y_test, metrics

def generate_pred_vs_true(y_train, train_preds, y_test, test_preds): 
    mean_train, lower_train, upper_train = train_preds 
    mean_test, lower_test, upper_test = test_preds 

    all_y = np.concatenate([y_train, y_test])
    all_mean = np.concatenate([mean_train, mean_test])
    y_min, y_max = min(all_y.min(), all_mean.min()), max(all_y.max(), all_mean.max())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))
    ax1.plot([y_min, y_max], [y_min, y_max], 'k--', alpha=0.5, label='Perfect Prediction')
    ax2.plot([y_min, y_max], [y_min, y_max], 'k--', alpha=0.5, label='Perfect Prediction')

    ax1.scatter(y_train, mean_train, c='blue', alpha=0.2, s=20, label='Training Points')
    
    # Plot test data
    ax2.scatter(y_test, mean_test, c='red', alpha=0.8, s=30, label='Test Points')
    
    # Plot confidence intervals for test points
    for i in range(len(y_test)):
        ax2.plot([y_test[i], y_test[i]], [lower_test[i], upper_test[i]], 
                 color='red', alpha=0.3, linewidth=1)
    
    for i in range(len(y_train)): 
        ax1.plot([y_train[i], y_train[i]], [lower_train[i], upper_train[i]], 
                 color='blue', alpha=0.3, linewidth=1)
    ax1.set_xlabel('True Values')
    ax1.set_ylabel('Predicted Values')
    ax2.set_xlabel('True Values')
    ax1.set_title('Training Data')
    ax2.set_title('Test Data')

    return fig

def evaluate_model(m, X, y, input_scaler, output_scaler, alpha): 
    X_tensor = validate_X_input(X, X.shape[1], requires_grad=False)
    X_s = input_scaler.transform(X_tensor)
    m.eval()
    preds = m.forward(X_s)
    mean = preds[:, 0]
    variances = preds[:, 1]
    std = variances.sqrt()
    std_mult = st.norm.ppf(1 - alpha / 2)
    lower = mean - std * std_mult 
    upper = mean  + std * std_mult

    mean = output_scaler.inverse_transform(mean.view(-1, 1)).squeeze().detach().cpu().numpy()
    lower = output_scaler.inverse_transform(lower.view(-1, 1)).squeeze().detach().cpu().numpy()
    upper = output_scaler.inverse_transform(upper.view(-1, 1)).squeeze().detach().cpu().numpy()

    print(y[:10])
    print(lower[:10])
    print(upper[:10])
    metrics = compute_all_metrics(mean, lower, upper, y, alpha=alpha)

    return mean, lower, upper, metrics

def continue_training(m, X_train, y_train, extra_epochs, input_scaler, output_scaler, batch_size, loss_fn, optimizer_cls, learning_rate): 
    X_tensor, y_tensor = validate_and_prepare_inputs(X_train, y_train)
    X_tensor = input_scaler.transform(X_tensor)
    y_tensor = output_scaler.transform(y_tensor)
    
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    #optimizer = optimizer_cls(m.parameters(), lr=learning_rate, weight_decay=0.005199)
    optimizer = optimizer_cls(m.parameters(), lr=learning_rate, weight_decay=0)

    for epoch in range(extra_epochs): 
        m.train() 
        epoch_loss = 0.0 
        for xb, yb in dataloader: 
            optimizer.zero_grad()
            preds = m(xb)
            loss = loss_fn(preds, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() 

        #print({"epoch": epoch, "train_loss": epoch_loss})

    return m

def retrain_members_extra_epochs(results_dir, dataset, split, extra_epochs=200, model = "DeepEnsemble", model_cls=DeepEnsembleRegressor): 
    model, X_train, X_test, y_train, y_test, prev_metrics = extract_model(results_dir, dataset, split, model, model_cls)
    print (model.name)
    epochs = model.epochs 
    total_epochs = epochs + extra_epochs
    model.epochs = total_epochs
    model.scheduler_cls = CosineAnnealingLR
    model.scheduler_kwargs = {"T_max": total_epochs}
    model.fit(X_train, y_train)
    mean, lower, upper = model.predict(X_test) 
    new_metrics = compute_all_metrics(mean, lower, upper, y_test, alpha=model.alpha)
    print("Previous Metrics: ")
    for k, v in prev_metrics.items(): print(k, v)
    print ("-" * 10)
    print("New Metrics: ")
    for k, v in new_metrics.items(): print(k, v)

def analyze_ensemble_members(results_dir, dataset, split, plot=False): 
    model, X_train, X_test, y_train, y_test, prev_metrics = extract_model(results_dir, dataset, split)
    input_scaler = model.input_scaler 
    output_scaler = model.output_scaler 
    alpha = model.alpha
    batch_size = model.batch_size 
    loss_fn = model.loss_fn 
    optimizer_cls = model.optimizer_cls 
    learning_rate = model.learning_rate
    extra_epochs = 200

    for i, m in enumerate(model.models):
        print(f"evaluating model {i} on test data") 
        test_mean, test_lower, test_upper, test_metrics = evaluate_model(m, X_test, y_test, input_scaler, output_scaler, alpha)
        print(test_metrics)
        print(f"evaluating model {i} on training data")
        train_mean, train_lower, train_upper, train_metrics = evaluate_model(m, X_train, y_train, input_scaler, output_scaler, alpha)
        print(train_metrics)
        generate_pred_vs_true(y_train, (train_mean, train_lower, train_upper), y_test, (test_mean, test_lower, test_upper))

        m_updated = continue_training(m, X_train, y_train, extra_epochs, input_scaler, output_scaler, batch_size, loss_fn, optimizer_cls, learning_rate)
        print(f"evaluating model {i} on test data") 
        test_mean, test_lower, test_upper, test_metrics = evaluate_model(m_updated, X_test, y_test, input_scaler, output_scaler, alpha)
        print(test_metrics)
        print(f"evaluating model {i} on training data")
        train_mean, train_lower, train_upper, train_metrics = evaluate_model(m_updated, X_train, y_train, input_scaler, output_scaler, alpha)
        print(train_metrics)
        if plot: 
            generate_pred_vs_true(y_train, (train_mean, train_lower, train_upper), y_test, (test_mean, test_lower, test_upper))
            #plt.title("Extra training")
            plt.show()
        model.models[i] = m
    
    ens_mean, ens_lower, ens_upper = model.predict(X_test)
    new_metrics = compute_all_metrics(ens_mean, ens_lower, ens_upper, y_test, alpha=model.alpha)
    print ("-" * 10)
    print("Previous Metrics: ")
    for k, v in prev_metrics.items(): print(k, v)
    print ("-" * 10)
    print("New Metrics: ")
    for k, v in new_metrics.items(): print(k, v)

def perform_weighted_prediction(): 
    pass

if __name__ == "__main__": 
    #for i in range(30): 
    #    extract_model("results/airfoil_results_lr_sched/p_250_runs", dataset = "Cl_nf_4D", split = f"split_{i}")
    #extract_model("results/airfoil_results_lr_sched/p_250_runs", dataset="Cl_nf_4D", split="split_6")
    analyze_ensemble_members("results/airfoil_results_lr_sched/p_250_runs", dataset="Cl_nf_4D", split="split_6", plot=True)
    #for i in range(6): 
    #    retrain_members_extra_epochs("results/airfoil_results_new/p_250_runs", dataset="Cl_nf_4D", split=f"split_{i}", model="KFoldQuantileRegression", model_cls=KFoldCQR, extra_epochs=600)