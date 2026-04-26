import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

class FeedForwardRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=3, activation='relu'):
        super().__init__()
        act_fn = {'relu': nn.ReLU, 'tanh': nn.Tanh, 'gelu': nn.GELU}.get(activation.lower(), nn.ReLU)

        layers = [nn.Linear(input_dim, hidden_dim), act_fn()]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), act_fn()]
        layers += [nn.Linear(hidden_dim, 1)]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x).squeeze(1)  # Output shape: (batch,)

class TorchMeanRegressor:
    def __init__(self, hidden_dim=64, num_layers=3, activation='relu', epochs=100,
                 batch_size='auto', learning_rate=1e-3, weight_decay=1e-4, lr_decay=0.95):
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.activation = activation
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.lr_decay = lr_decay

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None

    def fit(self, X, y, n_epochs=None, batch_size=None):
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y).float()

        X, y = X.to(self.device), y.to(self.device)

        input_dim = X.shape[1]
        self.model = FeedForwardRegressor(
            input_dim=input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            activation=self.activation
        ).to(self.device)

        dataset = TensorDataset(X, y)
        batch_size = len(X) // 3 if self.batch_size == 'None' else batch_size
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=self.lr_decay)
        loss_fn = nn.MSELoss()

        for epoch in range(n_epochs or self.epochs):
            self.model.train()
            for xb, yb in loader:
                optimizer.zero_grad()
                pred = self.model(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                optimizer.step()
            scheduler.step()
            if epoch % 100 == 0:
                print(f"epoch: {epoch}")

    def predict(self, X):
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()
        if X.dim() == 1:
            X = X.unsqueeze(0)
        X = X.to(self.device)

        self.model.eval()
        with torch.no_grad():
            return self.model(X)
