"""GNN model definition and utilities."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
import numpy as np


class GNNYield(nn.Module):
    """GNN model for yield prediction."""

    def __init__(self, node_in_dim, hidden_dim=32, out_dim=1, dropout_rate=0.2, num_conv_layers=1):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.num_conv_layers = num_conv_layers

        # Graph convolutional layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        # First conv layer: node_in_dim -> hidden_dim
        self.convs.append(GCNConv(node_in_dim, hidden_dim))
        self.bns.append(nn.BatchNorm1d(hidden_dim))

        # Additional conv layers: hidden_dim -> hidden_dim
        for _ in range(num_conv_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        # Prediction head / MLP (2 layers)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.bn_fc1 = nn.BatchNorm1d(hidden_dim)

        self.fc2 = nn.Linear(hidden_dim, out_dim)

        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Graph convolutions
        for conv, bn in zip(self.convs, self.bns):
            x = F.relu(bn(conv(x, edge_index)))

        # Global pooling
        x = global_mean_pool(x, batch)

        # MLP 1
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout(x)

        # MLP 2
        x = self.fc2(x)

        return x.squeeze(-1)


def train_epoch(model, loader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        # Forward pass
        output = model(batch)

        # Loss calculation
        loss = criterion(output, batch.y)

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch.num_graphs

    return total_loss / len(loader.dataset)


def evaluate(model, loader, criterion, device):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    predictions = []
    targets = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            output = model(batch)

            loss = criterion(output, batch.y)
            total_loss += loss.item() * batch.num_graphs

            # Convert to probability with sigmoid
            probs = torch.sigmoid(output)
            predictions.extend(probs.cpu().numpy())
            targets.extend(batch.y.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    predictions = np.array(predictions)
    targets = np.array(targets)

    # Calculate MAE and RMSE
    mae = np.mean(np.abs(predictions - targets))
    rmse = np.sqrt(np.mean((predictions - targets) ** 2))

    return avg_loss, mae, rmse


def enable_dropout(model):
    """Enable dropout layers during inference."""
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()


def mc_dropout_predict(model, data_list, n_samples=10, device='cpu'):
    """
    MC Dropout inference for uncertainty estimation.

    Args:
        model: Trained GNN model
        data_list: List of PyG Data objects
        n_samples: Number of MC Dropout samples
        device: Device to use

    Returns:
        means: Predicted mean values
        variances: Predicted variances
    """
    model.eval()
    enable_dropout(model)  # Enable dropout only

    all_predictions = []

    # Run n_samples forward passes
    with torch.no_grad():
        for _ in range(n_samples):
            batch_preds = []
            for data in data_list:
                data = data.to(device)
                output = model(data)
                # Convert to probability with sigmoid
                prob = torch.sigmoid(output).item()
                batch_preds.append(prob)
            all_predictions.append(batch_preds)

    # Calculate statistics
    all_predictions = np.array(all_predictions)  # shape: (n_samples, n_data)
    means = np.mean(all_predictions, axis=0)
    variances = np.var(all_predictions, axis=0)

    return means, variances
