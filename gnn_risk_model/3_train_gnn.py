"""
Step 3 — Train the GraphSAGE CVaR model.

Loads pre-computed features, targets, and graph snapshots, trains the GNN with
early stopping on the validation set, saves the trained model and predictions
to models/ and results/.

Runtime: ~10–30 minutes on CPU.
"""

import os
import sys
import pickle
import json
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
import config
from src.gnn_model import GraphSAGECVaR, train_model, evaluate

os.makedirs(config.MODEL_DIR,   exist_ok=True)
os.makedirs(config.RESULTS_DIR, exist_ok=True)


def main():
    device = torch.device("cpu")
    print(f"Using device: {device}\n")

    # ── Load data ──────────────────────────────────────────────────────────────
    print("Loading processed data...")
    features = np.load(config.DATA_DIR + "processed/features.npy")
    targets  = np.load(config.DATA_DIR + "processed/targets.npy")
    train_idx = np.load(config.DATA_DIR + "processed/train_indices.npy")
    val_idx   = np.load(config.DATA_DIR + "processed/val_indices.npy")
    test_idx  = np.load(config.DATA_DIR + "processed/test_indices.npy")
    dates     = pd.read_parquet(config.DATA_DIR + "processed/dates.parquet")["date"].values
    tickers   = pd.read_parquet(config.DATA_DIR + "processed/tickers.parquet")["ticker"].tolist()

    with open(config.DATA_DIR + "processed/edge_snapshots.pkl", "rb") as f:
        edge_snapshots = pickle.load(f)

    T, N, F = features.shape
    print(f"  Shape: {T} time steps × {N} stocks × {F} features")
    print(f"  Train/Val/Test: {len(train_idx)}/{len(val_idx)}/{len(test_idx)}\n")

    # ── Build model ────────────────────────────────────────────────────────────
    model = GraphSAGECVaR(
        in_channels     = F,
        hidden_channels = config.GNN_HIDDEN_DIM,
        num_layers      = config.GNN_NUM_LAYERS,
        dropout         = config.GNN_DROPOUT,
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"GraphSAGE | parameters: {n_params:,}")
    print(f"  in={F}, hidden={config.GNN_HIDDEN_DIM}, "
          f"layers={config.GNN_NUM_LAYERS}, dropout={config.GNN_DROPOUT}\n")

    # ── Train ──────────────────────────────────────────────────────────────────
    print("Training...")
    history = train_model(
        model          = model,
        features       = features,
        targets        = targets,
        edge_snapshots = edge_snapshots,
        train_indices  = train_idx,
        val_indices    = val_idx,
        lr             = config.LEARNING_RATE,
        weight_decay   = config.WEIGHT_DECAY,
        num_epochs     = config.NUM_EPOCHS,
        patience       = config.PATIENCE,
        device         = device,
    )

    # ── Save model ─────────────────────────────────────────────────────────────
    torch.save(model.state_dict(), config.MODEL_DIR + "gnn_model.pt")
    print(f"\nSaved model → {config.MODEL_DIR}gnn_model.pt")

    # ── Generate predictions for val + test ────────────────────────────────────
    print("\nGenerating predictions...")

    val_loss,  val_preds  = evaluate(model, features, targets, edge_snapshots, val_idx,  device)
    test_loss, test_preds = evaluate(model, features, targets, edge_snapshots, test_idx, device)

    print(f"  Val  MSE: {val_loss:.6f}")
    print(f"  Test MSE: {test_loss:.6f}")

    # Save predictions as DataFrames (T_split × N)
    val_dates  = pd.DatetimeIndex(dates[val_idx])
    test_dates = pd.DatetimeIndex(dates[test_idx])

    pd.DataFrame(val_preds,  index=val_dates,  columns=tickers).to_parquet(
        config.RESULTS_DIR + "gnn_val_predictions.parquet"
    )
    pd.DataFrame(test_preds, index=test_dates, columns=tickers).to_parquet(
        config.RESULTS_DIR + "gnn_test_predictions.parquet"
    )

    # Also save targets for reference
    val_targets  = targets[val_idx]
    test_targets = targets[test_idx]
    pd.DataFrame(val_targets,  index=val_dates,  columns=tickers).to_parquet(
        config.RESULTS_DIR + "targets_val.parquet"
    )
    pd.DataFrame(test_targets, index=test_dates, columns=tickers).to_parquet(
        config.RESULTS_DIR + "targets_test.parquet"
    )

    # ── Plot training curve ─────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(history["train_loss"], label="Train MSE", linewidth=1.5)
    ax.plot(history["val_loss"],   label="Val MSE",   linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")
    ax.set_title("GraphSAGE — Training Curve")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(config.RESULTS_DIR + "gnn_training_curve.png", dpi=150)
    plt.close(fig)
    print(f"  Training curve → {config.RESULTS_DIR}gnn_training_curve.png")

    # ── Save history ───────────────────────────────────────────────────────────
    with open(config.RESULTS_DIR + "gnn_training_history.json", "w") as f:
        json.dump(history, f)

    print("\nStep 3 complete.\n")


if __name__ == "__main__":
    main()
