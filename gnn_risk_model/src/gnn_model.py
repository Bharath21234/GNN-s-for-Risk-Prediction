"""GraphSAGE model for node-level CVaR regression."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
import numpy as np


class GraphSAGECVaR(nn.Module):
    """
    2-layer GraphSAGE for node-level CVaR prediction.
    Each node (stock) outputs its predicted 5-day CVaR.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()

        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        self.head = nn.Sequential(
            nn.Linear(hidden_channels, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Softplus(),   # ensure non-negative CVaR output
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.head(x).squeeze(-1)  # (N,)


class EarlyStopper:
    def __init__(self, patience: int = 20, min_delta: float = 1e-6) -> None:
        self.patience   = patience
        self.min_delta  = min_delta
        self.best_loss  = float("inf")
        self.counter    = 0
        self.best_state = None

    def step(self, val_loss: float, model: nn.Module) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss  = val_loss
            self.counter    = 0
            self.best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
        return self.counter >= self.patience

    def restore(self, model: nn.Module) -> None:
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


def train_one_epoch(
    model: GraphSAGECVaR,
    features: np.ndarray,
    targets: np.ndarray,
    edge_snapshots: list,
    indices: np.ndarray,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0

    # Shuffle time steps
    perm = np.random.permutation(indices)

    for t in perm:
        x   = torch.tensor(features[t], dtype=torch.float32, device=device)
        y   = torch.tensor(targets[t],  dtype=torch.float32, device=device)
        ei, _ = edge_snapshots[t]
        edge_index = torch.tensor(ei, dtype=torch.long, device=device)

        optimizer.zero_grad()
        pred = model(x, edge_index)
        loss = F.mse_loss(pred, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(perm)


@torch.no_grad()
def evaluate(
    model: GraphSAGECVaR,
    features: np.ndarray,
    targets: np.ndarray,
    edge_snapshots: list,
    indices: np.ndarray,
    device: torch.device,
) -> tuple[float, np.ndarray]:
    model.eval()
    all_preds = []
    total_loss = 0.0

    for t in indices:
        x   = torch.tensor(features[t], dtype=torch.float32, device=device)
        y   = torch.tensor(targets[t],  dtype=torch.float32, device=device)
        ei, _ = edge_snapshots[t]
        edge_index = torch.tensor(ei, dtype=torch.long, device=device)

        pred = model(x, edge_index)
        total_loss += F.mse_loss(pred, y).item()
        all_preds.append(pred.cpu().numpy())

    return total_loss / len(indices), np.stack(all_preds)  # (T, N)


def train_model(
    model: GraphSAGECVaR,
    features: np.ndarray,
    targets: np.ndarray,
    edge_snapshots: list,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    num_epochs: int = 150,
    patience: int = 20,
    device: torch.device | None = None,
) -> dict:
    if device is None:
        device = torch.device("cpu")

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    stopper   = EarlyStopper(patience=patience)

    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, num_epochs + 1):
        train_loss = train_one_epoch(
            model, features, targets, edge_snapshots, train_indices, optimizer, device
        )
        val_loss, _ = evaluate(
            model, features, targets, edge_snapshots, val_indices, device
        )
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:>4d} | train MSE={train_loss:.6f} | val MSE={val_loss:.6f}")

        if stopper.step(val_loss, model):
            print(f"  Early stopping at epoch {epoch}.")
            break

    stopper.restore(model)
    print(f"  Best val MSE: {stopper.best_loss:.6f}")
    return history
