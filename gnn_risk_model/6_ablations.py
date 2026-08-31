"""
Step 6 — Graph-construction and edge-weight ablations (P2-8 + edge weights).

Shows the main result is not an artefact of one arbitrary graph, and tests
whether correlation edge *weights* help. Each configuration is trained for
config.ABLATION_SEEDS seeds with the same FZ0 objective and evaluated on the
test window; we report mean±std FZ0 loss.

Configurations:
  sector       (SAGE)  — intra-sector edges only
  correlation  (SAGE)  — cross-sector correlation edges only
  combined     (SAGE)  — both edge types (matches the main model)
  combined     (GraphConv, |corr|-weighted) — edge-weight ablation

Runtime: 4 configs x ABLATION_SEEDS trainings (bounded seeds to limit compute).
"""

import os, sys, pickle, json
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(__file__))
import config
from src.graph import precompute_edge_indices
from src.gnn_model import GraphSAGERisk, train_model, evaluate
from src.metrics import fz0_loss, pinball_loss

torch.set_num_threads(int(os.environ.get("TORCH_NUM_THREADS", "2")))


def _snapshots(returns, tickers, sectors, dates, variant):
    return precompute_edge_indices(
        returns=returns, tickers=tickers, sectors=sectors, valid_dates=dates,
        corr_window=config.CORR_WINDOW, corr_threshold=config.CORR_THRESHOLD,
        update_freq=config.GRAPH_UPDATE_FREQ, graph_variant=variant,
    )


def main():
    device = torch.device("cpu")
    features    = np.load(config.DATA_DIR + "processed/features.npy")
    fwd_returns = np.load(config.DATA_DIR + "processed/fwd_returns.npy")
    train_idx   = np.load(config.DATA_DIR + "processed/train_indices.npy")
    val_idx     = np.load(config.DATA_DIR + "processed/val_indices.npy")
    test_idx    = np.load(config.DATA_DIR + "processed/test_indices.npy")
    dates       = pd.read_parquet(config.DATA_DIR + "processed/dates.parquet")["date"]
    tickers     = pd.read_parquet(config.DATA_DIR + "processed/tickers.parquet")["ticker"].tolist()
    returns     = pd.read_parquet(config.DATA_DIR + "raw/returns.parquet")[tickers]
    sectors     = {t: config.STOCKS[t] for t in tickers}
    F = features.shape[2]
    Rtest = fwd_returns[test_idx]

    dates_idx = pd.DatetimeIndex(dates.values)
    configs = [
        ("sector",      "sage",  "Sector-only (SAGE)"),
        ("correlation", "sage",  "Correlation-only (SAGE)"),
        ("combined",    "sage",  "Combined (SAGE)"),
        ("combined",    "graph", "Combined (GraphConv, |corr|-weighted)"),
        ("combined",    "gat",   "Combined (GAT, primary model)"),
    ]

    # cache snapshots per variant
    snap_cache = {}
    rows = []
    for variant, conv, label in configs:
        if variant not in snap_cache:
            print(f"  building '{variant}' snapshots...")
            snap_cache[variant] = _snapshots(returns, tickers, sectors, dates_idx, variant)
        snaps = snap_cache[variant]
        n_edges = snaps[test_idx[0]][0].shape[1]

        fzs, pbs = [], []
        for seed in config.ABLATION_SEEDS:
            print(f"── {label} | seed {seed} (edges≈{n_edges}) ──")
            model = GraphSAGERisk(F, config.GNN_HIDDEN_DIM, config.GNN_NUM_LAYERS,
                                  config.GNN_DROPOUT, conv_type=conv,
                                  attn_dropout=config.GNN_ATTN_DROPOUT)
            train_model(model, features, fwd_returns, snaps, train_idx, val_idx,
                        lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY,
                        num_epochs=config.NUM_EPOCHS, patience=config.PATIENCE,
                        alpha=config.TAIL_ALPHA, use_graph=True, seed=seed,
                        device=device, verbose=False)
            _, pred = evaluate(model, features, fwd_returns, snaps, test_idx,
                               device, alpha=config.TAIL_ALPHA, use_graph=True)
            fzs.append(fz0_loss(Rtest, pred[:, :, 0], pred[:, :, 1], config.TAIL_ALPHA))
            pbs.append(pinball_loss(Rtest, pred[:, :, 0], config.TAIL_ALPHA))
        rows.append({"config": label, "n_edges": int(n_edges),
                     "FZ0_mean": float(np.mean(fzs)), "FZ0_std": float(np.std(fzs)),
                     "pinball_mean": float(np.mean(pbs)), "n_seeds": len(fzs)})
        print(f"   → FZ0 {np.mean(fzs):.5f} ± {np.std(fzs):.5f}")

    df = pd.DataFrame(rows).set_index("config")
    print("\n── Graph-construction / edge-weight ablation (test FZ0) ──")
    print(df.to_string(float_format=lambda x: f"{x:.5f}"))
    df.to_csv(config.RESULTS_DIR + "ablation_graph.csv")
    with open(config.RESULTS_DIR + "ablation_graph.json", "w") as f:
        json.dump(rows, f, indent=2)
    print("\nStep 6 complete.\n")


if __name__ == "__main__":
    main()
