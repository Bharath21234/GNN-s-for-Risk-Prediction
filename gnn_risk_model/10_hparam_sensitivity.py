"""
Step 10 — Graph-hyperparameter sensitivity.

Reviewers rightly ask whether a graph result is an artefact of one arbitrary
graph.  Here we rebuild the graph across a grid of the two construction knobs —
the correlation threshold for cross-sector edges and the graph update frequency
— and retrain the GraphSAGE model for each cell (over SENSITIVITY_SEEDS),
reporting the forward FZ0 loss.  A result that is stable across the grid is far
more convincing than a single tuned setting.

The default configuration (config.CORR_THRESHOLD, config.GRAPH_UPDATE_FREQ) is
included in the grid and flagged, so the table shows how far the reported number
moves as the graph is rewired.  Output: results/sensitivity.csv.
"""

import os
import sys
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(__file__))
import config
from src.graph import precompute_edge_indices
from src.gnn_model import GraphSAGERisk, train_model, evaluate

R = config.RESULTS_DIR
os.makedirs(R, exist_ok=True)


def main():
    n_threads = int(os.environ.get("TORCH_NUM_THREADS", "2"))
    torch.set_num_threads(n_threads)
    device = torch.device("cpu")
    print(f"Using device: {device} | torch threads: {n_threads}\n")

    D = config.DATA_DIR + "processed/"
    features = np.load(D + "features.npy")
    fwd      = np.load(D + "fwd_returns.npy")
    tr = np.load(D + "train_indices.npy")
    va = np.load(D + "val_indices.npy")
    te = np.load(D + "test_indices.npy")
    dates   = pd.read_parquet(D + "dates.parquet")["date"]
    valid_dates = pd.DatetimeIndex(dates.values)
    tickers = pd.read_parquet(D + "tickers.parquet")["ticker"].tolist()
    T, N, Fdim = features.shape

    # Full raw returns, restricted to the model universe in the saved order.
    returns = pd.read_parquet(config.DATA_DIR + "raw/returns.parquet")
    returns = returns[[t for t in tickers if t in returns.columns]]
    sectors = {t: config.STOCKS[t] for t in tickers}

    alpha = config.TAIL_ALPHA
    def_thr, def_freq = config.CORR_THRESHOLD, config.GRAPH_UPDATE_FREQ

    rows = []
    grid = [(thr, freq) for thr in config.SENSITIVITY_CORR_THRESH
                        for freq in config.SENSITIVITY_UPDATE_FREQ]
    print(f"Sensitivity grid: {len(grid)} cells × {len(config.SENSITIVITY_SEEDS)} seeds\n")

    for thr, freq in grid:
        is_default = (thr == def_thr and freq == def_freq)
        tag = f"thr={thr} freq={freq}" + ("  [default]" if is_default else "")
        print(f"── Rebuilding graph: {tag} ─────────────────────────────")
        edges = precompute_edge_indices(
            returns=returns, tickers=tickers, sectors=sectors,
            valid_dates=valid_dates, corr_window=config.CORR_WINDOW,
            corr_threshold=thr, update_freq=freq, graph_variant="combined",
        )
        n_edges = int(edges[-1][0].shape[1])

        fz_seeds = []
        for seed in config.SENSITIVITY_SEEDS:
            model = GraphSAGERisk(
                in_channels=Fdim, hidden_channels=config.GNN_HIDDEN_DIM,
                num_layers=config.GNN_NUM_LAYERS, dropout=config.GNN_DROPOUT,
                conv_type=config.GNN_CONV_TYPE, attn_dropout=config.GNN_ATTN_DROPOUT,
            )
            train_model(
                model=model, features=features, fwd_returns=fwd, edge_snapshots=edges,
                train_indices=tr, val_indices=va, lr=config.LEARNING_RATE,
                weight_decay=config.WEIGHT_DECAY, num_epochs=config.NUM_EPOCHS,
                patience=config.PATIENCE, alpha=alpha, use_graph=True,
                seed=seed, device=device, verbose=False,
            )
            test_fz, _ = evaluate(model, features, fwd, edges, te, device,
                                  alpha=alpha, use_graph=True)
            fz_seeds.append(test_fz)
        fz_seeds = np.array(fz_seeds)
        rows.append({
            "corr_threshold": thr, "update_freq": freq, "n_edges": n_edges,
            "test_FZ0_mean": float(fz_seeds.mean()),
            "test_FZ0_std":  float(fz_seeds.std()),
            "is_default":    is_default,
        })
        print(f"    edges={n_edges}  test FZ0 = {fz_seeds.mean():.5f} ± {fz_seeds.std():.5f}\n")

    df = pd.DataFrame(rows)
    df.to_csv(R + "sensitivity.csv", index=False)
    spread = df["test_FZ0_mean"].max() - df["test_FZ0_mean"].min()
    print(df.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print(f"\n  FZ0 spread across the whole grid: {spread:.5f}")
    print("  saved sensitivity.csv\nStep 10 complete.\n")


if __name__ == "__main__":
    main()
