"""
Step 3 — Train the GraphSAGE (VaR, ES) model and the no-graph MLP ablation.

For each seed in config.SEEDS we train:
  * GNN  — GraphSAGE over the correlation+sector graph (use_graph=True)
  * MLP  — identical architecture, empty edge_index (use_graph=False), which
           isolates the value of the graph structure (P1-4 ablation).

Both are trained with the strictly consistent Fissler-Ziegel (FZ0) joint
(VaR, ES) loss against realised forward 5-day returns.  Per-seed VaR/ES
predictions for the val and test windows are persisted so that Step 5 can
report mean±std, Diebold-Mariano significance, and correct VaR/ES backtests.

Runtime: ~10-30 min per model per seed on CPU (≈ a few hours for 5 seeds × 2).
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
from src.gnn_model import GraphSAGERisk, train_model, evaluate

os.makedirs(config.MODEL_DIR,   exist_ok=True)
os.makedirs(config.RESULTS_DIR, exist_ok=True)


def _train_one(features, fwd_returns, edge_snapshots, train_idx, val_idx,
               test_idx, F, use_graph, seed, device):
    """Train a single model (GNN or MLP) for one seed; return preds + history."""
    model = GraphSAGERisk(
        in_channels     = F,
        hidden_channels = config.GNN_HIDDEN_DIM,
        num_layers      = config.GNN_NUM_LAYERS,
        dropout         = config.GNN_DROPOUT,
        conv_type       = config.GNN_CONV_TYPE,
        attn_dropout    = config.GNN_ATTN_DROPOUT if use_graph else 0.0,
    )
    history = train_model(
        model          = model,
        features       = features,
        fwd_returns    = fwd_returns,
        edge_snapshots = edge_snapshots,
        train_indices  = train_idx,
        val_indices    = val_idx,
        lr             = config.LEARNING_RATE,
        weight_decay   = config.WEIGHT_DECAY,
        num_epochs     = config.NUM_EPOCHS,
        patience       = config.PATIENCE,
        alpha          = config.TAIL_ALPHA,
        use_graph      = use_graph,
        seed           = seed,
        device         = device,
        verbose        = True,
    )
    _, val_pred  = evaluate(model, features, fwd_returns, edge_snapshots,
                            val_idx,  device, alpha=config.TAIL_ALPHA, use_graph=use_graph)
    _, test_pred = evaluate(model, features, fwd_returns, edge_snapshots,
                            test_idx, device, alpha=config.TAIL_ALPHA, use_graph=use_graph)
    return model, history, val_pred, test_pred   # preds: (T, N, 2) = [VaR, ES]


def main():
    # Thermal throttle: cap CPU threads so training does not pin all cores.
    n_threads = int(os.environ.get("TORCH_NUM_THREADS", "2"))
    torch.set_num_threads(n_threads)
    device = torch.device("cpu")
    print(f"Using device: {device} | torch threads: {n_threads}\n")

    # ── Load data ──────────────────────────────────────────────────────────────
    print("Loading processed data...")
    features    = np.load(config.DATA_DIR + "processed/features.npy")
    fwd_returns = np.load(config.DATA_DIR + "processed/fwd_returns.npy")
    targets     = np.load(config.DATA_DIR + "processed/targets.npy")
    train_idx   = np.load(config.DATA_DIR + "processed/train_indices.npy")
    val_idx     = np.load(config.DATA_DIR + "processed/val_indices.npy")
    test_idx    = np.load(config.DATA_DIR + "processed/test_indices.npy")
    dates       = pd.read_parquet(config.DATA_DIR + "processed/dates.parquet")["date"].values
    tickers     = pd.read_parquet(config.DATA_DIR + "processed/tickers.parquet")["ticker"].tolist()

    with open(config.DATA_DIR + "processed/edge_snapshots.pkl", "rb") as f:
        edge_snapshots = pickle.load(f)

    T, N, F = features.shape
    print(f"  Shape: {T} time steps × {N} stocks × {F} features")
    print(f"  Train/Val/Test: {len(train_idx)}/{len(val_idx)}/{len(test_idx)}")
    print(f"  Seeds: {config.SEEDS}\n")

    val_dates  = pd.DatetimeIndex(dates[val_idx])
    test_dates = pd.DatetimeIndex(dates[test_idx])

    # Persist the labels needed by Step 5
    pd.DataFrame(targets[val_idx],  index=val_dates,  columns=tickers).to_parquet(config.RESULTS_DIR + "targets_val.parquet")
    pd.DataFrame(targets[test_idx], index=test_dates, columns=tickers).to_parquet(config.RESULTS_DIR + "targets_test.parquet")
    pd.DataFrame(fwd_returns[val_idx],  index=val_dates,  columns=tickers).to_parquet(config.RESULTS_DIR + "fwd_val.parquet")
    pd.DataFrame(fwd_returns[test_idx], index=test_dates, columns=tickers).to_parquet(config.RESULTS_DIR + "fwd_test.parquet")

    # ── Seed loop, both models ─────────────────────────────────────────────────
    store = {m: {s: {"val": None, "test": None}
                 for s in ["var", "es"]} for m in ["gnn", "mlp"]}
    # collect per-seed prediction stacks
    stacks = {f"{m}_{split}_{q}": []
              for m in ["gnn", "mlp"] for split in ["val", "test"] for q in ["var", "es"]}
    seed_summary = {"gnn": [], "mlp": []}
    first_history = None
    best = {"gnn": (np.inf, None), "mlp": (np.inf, None)}

    for seed in config.SEEDS:
        for name, use_graph in [("gnn", True), ("mlp", False)]:
            print(f"── Training {name.upper()} | seed {seed} | use_graph={use_graph} "
                  f"{'─'*20}")
            model, hist, val_pred, test_pred = _train_one(
                features, fwd_returns, edge_snapshots,
                train_idx, val_idx, test_idx, F, use_graph, seed, device,
            )
            stacks[f"{name}_val_var"].append(val_pred[:, :, 0])
            stacks[f"{name}_val_es"].append(val_pred[:, :, 1])
            stacks[f"{name}_test_var"].append(test_pred[:, :, 0])
            stacks[f"{name}_test_es"].append(test_pred[:, :, 1])
            seed_summary[name].append(
                {"seed": seed, "best_val_fz0": hist["best_val_loss"]}
            )
            if hist["best_val_loss"] < best[name][0]:
                best[name] = (hist["best_val_loss"], model.state_dict())
            if name == "gnn" and first_history is None:
                first_history = hist
            print()

    # ── Save prediction stacks (S, T, N) ───────────────────────────────────────
    print("Saving per-seed predictions...")
    for key, lst in stacks.items():
        np.save(config.RESULTS_DIR + f"{key}_seeds.npy", np.stack(lst))  # (S, T, N)

    # Save best-seed model states
    torch.save(best["gnn"][1], config.MODEL_DIR + "gnn_model.pt")
    torch.save(best["mlp"][1], config.MODEL_DIR + "mlp_model.pt")

    # Save seed summary + history
    with open(config.RESULTS_DIR + "train_summary.json", "w") as f:
        json.dump({
            "seeds": config.SEEDS,
            "gnn": seed_summary["gnn"],
            "mlp": seed_summary["mlp"],
        }, f, indent=2)
    with open(config.RESULTS_DIR + "gnn_training_history.json", "w") as f:
        json.dump({k: v for k, v in first_history.items() if k != "best_val_loss"}, f)

    # ── Training curve (first GNN seed) ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(first_history["train_loss"], label="Train FZ0", linewidth=1.5)
    ax.plot(first_history["val_loss"],   label="Val FZ0",   linewidth=1.5)
    ax.set_xlabel("Epoch"); ax.set_ylabel("FZ0 loss")
    ax.set_title(f"GraphSAGE (VaR, ES) — Training Curve (seed {config.SEEDS[0]})")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(config.RESULTS_DIR + "gnn_training_curve.png", dpi=150)
    plt.close(fig)

    # ── Console summary ────────────────────────────────────────────────────────
    for name in ["gnn", "mlp"]:
        vals = [s["best_val_fz0"] for s in seed_summary[name]]
        print(f"  {name.upper()} best val FZ0: {np.mean(vals):.5f} ± {np.std(vals):.5f} "
              f"(n={len(vals)})")
    print("\nStep 3 complete.\n")


if __name__ == "__main__":
    main()
