"""
Step 7 — Rolling-origin / walk-forward evaluation (P2-7).

Instead of one fixed 2023-24 test window, we refit on an expanding history and
evaluate out-of-sample on each subsequent year (config.WALKFORWARD_FOLDS). For
every fold we retrain the GNN and the no-graph MLP (config.WALKFORWARD_SEEDS
seeds each) with the FZ0 objective and score forward-looking FZ0 on the fold's
test year; GARCH ES is reused from step 4. This strengthens generalisation
claims across regimes (2021 recovery, 2022 drawdown, 2023-24).

Runtime: len(folds) x 2 models x WALKFORWARD_SEEDS trainings (bounded seeds).
"""

import os, sys, pickle, json
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(__file__))
import config
from src.gnn_model import GraphSAGERisk, train_model, evaluate
from src.metrics import fz0_loss, diebold_mariano

torch.set_num_threads(int(os.environ.get("TORCH_NUM_THREADS", "2")))


def _trim(idx, h):
    return idx[:-(h - 1)] if h > 1 and len(idx) > (h - 1) else idx


def main():
    device = torch.device("cpu")
    features    = np.load(config.DATA_DIR + "processed/features.npy")
    fwd_returns = np.load(config.DATA_DIR + "processed/fwd_returns.npy")
    dates       = pd.to_datetime(pd.read_parquet(config.DATA_DIR + "processed/dates.parquet")["date"])
    tickers     = pd.read_parquet(config.DATA_DIR + "processed/tickers.parquet")["ticker"].tolist()
    with open(config.DATA_DIR + "processed/edge_snapshots.pkl", "rb") as f:
        snaps = pickle.load(f)
    F = features.shape[2]
    d = dates.values
    h = config.HORIZON

    # Continuous GARCH ES over 2021-2024 (val 2021-22 + test 2023-24), reused per fold
    g_es = pd.concat([pd.read_parquet(config.RESULTS_DIR + "garch_val_es.parquet"),
                      pd.read_parquet(config.RESULTS_DIR + "garch_test_es.parquet")])
    g_es = g_es[~g_es.index.duplicated()].reindex(columns=tickers)
    g_var = pd.concat([pd.read_parquet(config.RESULTS_DIR + "garch_val_var.parquet"),
                       pd.read_parquet(config.RESULTS_DIR + "garch_test_var.parquet")])
    g_var = g_var[~g_var.index.duplicated()].reindex(columns=tickers)

    fold_rows, agg = [], {"gnn": [], "mlp": [], "garch": []}
    for (tr_end, va_end, te_end) in config.WALKFORWARD_FOLDS:
        tr = _trim(np.where(d <= np.datetime64(tr_end))[0], h)
        va = _trim(np.where((d > np.datetime64(tr_end)) & (d <= np.datetime64(va_end)))[0], h)
        te = _trim(np.where((d > np.datetime64(va_end)) & (d <= np.datetime64(te_end)))[0], h)
        if len(te) == 0 or len(tr) == 0:
            continue
        te_dates = pd.DatetimeIndex(d[te])
        Rte = fwd_returns[te]
        year = te_end[:4]
        print(f"── Fold test {year}: train={len(tr)} val={len(va)} test={len(te)} ──")

        fold = {"fold": year, "n_test": len(te)}
        for name, ug in [("gnn", True), ("mlp", False)]:
            fzs = []
            for seed in config.WALKFORWARD_SEEDS:
                model = GraphSAGERisk(F, config.GNN_HIDDEN_DIM, config.GNN_NUM_LAYERS,
                                      config.GNN_DROPOUT, conv_type=config.GNN_CONV_TYPE,
                                      attn_dropout=config.GNN_ATTN_DROPOUT if ug else 0.0)
                train_model(model, features, fwd_returns, snaps, tr, va,
                            lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY,
                            num_epochs=config.NUM_EPOCHS, patience=config.PATIENCE,
                            alpha=config.TAIL_ALPHA, use_graph=ug, seed=seed,
                            device=device, verbose=False)
                _, pred = evaluate(model, features, fwd_returns, snaps, te,
                                   device, alpha=config.TAIL_ALPHA, use_graph=ug)
                fzs.append(fz0_loss(Rte, pred[:, :, 0], pred[:, :, 1], config.TAIL_ALPHA))
            fold[f"{name}_FZ0"] = float(np.mean(fzs))
            fold[f"{name}_FZ0_std"] = float(np.std(fzs))
            agg[name].append(np.mean(fzs))

        # GARCH on this fold's test dates
        gv = g_var.reindex(index=te_dates, columns=tickers).values
        ge = g_es.reindex(index=te_dates, columns=tickers).values
        m = np.isfinite(ge).all(axis=1)
        fold["garch_FZ0"] = float(fz0_loss(Rte[m], gv[m], ge[m], config.TAIL_ALPHA))
        agg["garch"].append(fold["garch_FZ0"])
        print(f"   GNN={fold['gnn_FZ0']:.5f}  MLP={fold['mlp_FZ0']:.5f}  GARCH={fold['garch_FZ0']:.5f}")
        fold_rows.append(fold)

    df = pd.DataFrame(fold_rows).set_index("fold")
    print("\n── Walk-forward FZ0 by fold ──")
    print(df.to_string(float_format=lambda x: f"{x:.5f}"))
    df.to_csv(config.RESULTS_DIR + "walkforward.csv")
    summary = {"mean_FZ0": {k: float(np.mean(v)) for k, v in agg.items() if v},
               "folds": fold_rows}
    with open(config.RESULTS_DIR + "walkforward.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Aggregate mean FZ0: " +
          "  ".join(f"{k}={np.mean(v):.5f}" for k, v in agg.items() if v))
    print("\nStep 7 complete.\n")


if __name__ == "__main__":
    main()
