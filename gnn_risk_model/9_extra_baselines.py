"""
Step 9 — Extra baselines and a second risk level (reviewer-value additions).

Adds, at the primary 95% level:
  * GCN            — plain (transductive, symmetric-normalised) graph conv, so the
                     "why GraphSAGE and not GCN?" question is answered empirically.
                     Saved as gcn_{val,test}_{var,es}_seeds.npy → auto-detected by
                     Step 5 and folded into every headline table.
  * Historical Sim — non-parametric rolling empirical VaR/ES, the standard tail
                     benchmark.  Saved as histsim_{val,test}_{var,es}.parquet.

And a robustness study at each extra tail level in config.EXTRA_TAIL_ALPHAS
(default 0.025 = 97.5% ES, the Basel/FRTB regulatory level): GNN, MLP and GCN
are retrained at that level; GARCH/GJR are re-derived by the exact normal
quantile rescaling of their 95% forecasts; historical sim is recomputed; and a
compact FZ0 / pinball / Diebold-Mariano / VaR-coverage table is written to
robustness_a{level}.csv.
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import torch
from scipy.stats import norm

sys.path.insert(0, os.path.dirname(__file__))
import config
from src.gnn_model import GraphSAGERisk, train_model, evaluate
from src.histsim import run_histsim
from src.metrics import fz0_loss, pinball_loss, diebold_mariano, backtest_model

R = config.RESULTS_DIR
os.makedirs(R, exist_ok=True)


def _alpha_tag(a: float) -> str:
    return f"a{round(a * 1000):03d}"          # 0.025 -> "a025"


def _norm_scales(a: float) -> tuple[float, float]:
    """(VaR, ES) multipliers of sigma for a normal tail at level a."""
    z = norm.ppf(a)
    return float(-z), float(norm.pdf(z) / a)


def _train_seed_stack(data, conv_type, use_graph, seeds, alpha, device):
    """Train `conv_type` over `seeds` at tail level `alpha`; return prediction stacks."""
    features, fwd, edges, tr, va, te, F = data
    out = {f"{sp}_{q}": [] for sp in ["val", "test"] for q in ["var", "es"]}
    best_val = []
    for seed in seeds:
        model = GraphSAGERisk(
            in_channels=F, hidden_channels=config.GNN_HIDDEN_DIM,
            num_layers=config.GNN_NUM_LAYERS, dropout=config.GNN_DROPOUT,
            conv_type=conv_type,
            attn_dropout=config.GNN_ATTN_DROPOUT if use_graph else 0.0,
        )
        hist = train_model(
            model=model, features=features, fwd_returns=fwd, edge_snapshots=edges,
            train_indices=tr, val_indices=va, lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY, num_epochs=config.NUM_EPOCHS,
            patience=config.PATIENCE, alpha=alpha, use_graph=use_graph,
            seed=seed, device=device, verbose=False,
        )
        _, vp = evaluate(model, features, fwd, edges, va, device, alpha=alpha, use_graph=use_graph)
        _, tp = evaluate(model, features, fwd, edges, te, device, alpha=alpha, use_graph=use_graph)
        out["val_var"].append(vp[:, :, 0]);  out["val_es"].append(vp[:, :, 1])
        out["test_var"].append(tp[:, :, 0]); out["test_es"].append(tp[:, :, 1])
        best_val.append(hist["best_val_loss"])
        print(f"    seed {seed}: best val FZ0 = {hist['best_val_loss']:.5f}")
    return {k: np.stack(v) for k, v in out.items()}, best_val


def _save_seed_stack(name, stacks):
    for sp in ["val", "test"]:
        for q in ["var", "es"]:
            np.save(R + f"{name}_{sp}_{q}_seeds.npy", stacks[f"{sp}_{q}"])


def main():
    n_threads = int(os.environ.get("TORCH_NUM_THREADS", "2"))
    torch.set_num_threads(n_threads)
    device = torch.device("cpu")
    print(f"Using device: {device} | torch threads: {n_threads}\n")

    # ── Load processed data (same layout as Step 3) ────────────────────────────
    D = config.DATA_DIR + "processed/"
    features = np.load(D + "features.npy")
    fwd      = np.load(D + "fwd_returns.npy")
    tr = np.load(D + "train_indices.npy")
    va = np.load(D + "val_indices.npy")
    te = np.load(D + "test_indices.npy")
    dates   = pd.read_parquet(D + "dates.parquet")["date"].values
    tickers = pd.read_parquet(D + "tickers.parquet")["ticker"].tolist()
    with open(D + "edge_snapshots.pkl", "rb") as f:
        edges = pickle.load(f)
    T, N, F = features.shape
    data = (features, fwd, edges, tr, va, te, F)

    val_dates  = pd.DatetimeIndex(dates[va])
    test_dates = pd.DatetimeIndex(dates[te])
    returns_raw = pd.read_parquet(config.DATA_DIR + "raw/returns.parquet")

    print(f"  {T} steps × {N} stocks × {F} feats | "
          f"val {len(va)} / test {len(te)}\n")

    # ══ 1. Primary-level (95%) extra baselines ════════════════════════════════
    a0 = config.TAIL_ALPHA

    print("── GCN baseline (plain graph conv) @ 95% ────────────────────────────")
    gcn_stacks, _ = _train_seed_stack(data, "gcn", True, config.GCN_SEEDS, a0, device)
    _save_seed_stack("gcn", gcn_stacks)
    print("  saved gcn_*_seeds.npy\n")

    print("── Historical-simulation baseline @ 95% ─────────────────────────────")
    for split, pdates in [("val", val_dates), ("test", test_dates)]:
        v, e = run_histsim(returns_raw, pdates, tickers, alpha=a0,
                           horizon=config.HORIZON, window=config.HISTSIM_WINDOW)
        v.to_parquet(R + f"histsim_{split}_var.parquet")
        e.to_parquet(R + f"histsim_{split}_es.parquet")
    print("  saved histsim_*_{var,es}.parquet\n")

    # ══ 2. Robustness at each extra tail level ════════════════════════════════
    for a in config.EXTRA_TAIL_ALPHAS:
        tag = _alpha_tag(a)
        pct = (1 - a) * 100
        print(f"══ Extra risk level: {pct:.1f}% (alpha={a}, tag={tag}) ══════════════")

        # Retrain the learned models at this level
        stacks = {}
        for name, conv, ug in [("gnn", config.GNN_CONV_TYPE, True),
                               ("mlp", config.GNN_CONV_TYPE, False),
                               ("gcn", "gcn", True)]:
            print(f"── {name.upper()} @ {pct:.1f}% ───────────────────────────────")
            st, _ = _train_seed_stack(data, conv, ug, config.GCN_SEEDS, a, device)
            _save_seed_stack(f"{name}_{tag}", st)
            stacks[name] = st

        # Historical sim at this level
        for split, pdates in [("val", val_dates), ("test", test_dates)]:
            v, e = run_histsim(returns_raw, pdates, tickers, alpha=a,
                               horizon=config.HORIZON, window=config.HISTSIM_WINDOW)
            v.to_parquet(R + f"histsim_{tag}_{split}_var.parquet")
            e.to_parquet(R + f"histsim_{tag}_{split}_es.parquet")

        # GARCH / GJR: exact normal rescaling from the 95% forecasts
        vs0, cs0 = _norm_scales(a0)
        vs,  cs  = _norm_scales(a)
        for base in ["garch", "gjr"]:
            for split in ["val", "test"]:
                vp = pd.read_parquet(R + f"{base}_{split}_var.parquet") * (vs / vs0)
                ep = pd.read_parquet(R + f"{base}_{split}_es.parquet")  * (cs / cs0)
                vp.to_parquet(R + f"{base}_{tag}_{split}_var.parquet")
                ep.to_parquet(R + f"{base}_{tag}_{split}_es.parquet")

        # ── Compact robustness table on the test set ──────────────────────────
        _robustness_table(a, tag, stacks, val_dates, test_dates, dates, te,
                          tickers, fwd, returns_raw)

    print("Step 9 complete.\n")


def _robustness_table(a, tag, seed_stacks, val_dates, test_dates, dates, te,
                      tickers, fwd, returns_raw):
    """FZ0 / pinball / DM-vs-GNN / VaR coverage at level `a`, written to CSV."""
    Rmat = fwd[te]                                            # (T, N) realised
    LABEL = {"gnn": "GNN", "mlp": "MLP (no graph)", "gcn": "GCN",
             "garch": "GARCH(1,1)", "gjr": "GJR-GARCH", "histsim": "Historical Sim"}

    # Assemble ensemble (mean-over-seed) VaR/ES per model on the test grid.
    ens = {}
    for name, st in seed_stacks.items():                     # gnn, mlp, gcn
        ens[name] = (st["test_var"].mean(0), st["test_es"].mean(0))
    def _reidx(fn):
        return pd.read_parquet(R + fn).reindex(index=test_dates, columns=tickers).values
    for base in ["garch", "gjr"]:
        ens[base] = (_reidx(f"{base}_{tag}_test_var.parquet"),
                     _reidx(f"{base}_{tag}_test_es.parquet"))
    ens["histsim"] = (_reidx(f"histsim_{tag}_test_var.parquet"),
                      _reidx(f"histsim_{tag}_test_es.parquet"))

    order = ["gnn", "mlp", "gcn", "garch", "gjr", "histsim"]
    la = fz0_loss(Rmat, *ens["gnn"], a, reduce=False).reshape(len(test_dates), -1).mean(1)
    rows = []
    for name in order:
        v, e = ens[name]
        fz = fz0_loss(Rmat, v, e, a)
        pb = pinball_loss(Rmat, v, a)
        lb = fz0_loss(Rmat, v, e, a, reduce=False).reshape(len(test_dates), -1).mean(1)
        dm = diebold_mariano(la, lb, horizon=config.HORIZON) if name != "gnn" else {"dm_stat": np.nan, "p_value": np.nan}
        bt = backtest_model(pd.DataFrame(v, index=test_dates, columns=tickers),
                            returns_raw, horizon=config.HORIZON, alpha=a)["summary"]
        rows.append({"model": LABEL[name], "FZ0": fz, "pinball": pb,
                     "DM_vs_GNN": dm["dm_stat"], "p_vs_GNN": dm["p_value"],
                     "viol_rate_%": bt["mean_violation_rate"] * 100,
                     "kupiec_pass_%": bt["kupiec_pass_pct"],
                     "target_viol_%": a * 100})
    df = pd.DataFrame(rows).set_index("model")
    print(df.to_string(float_format=lambda x: f"{x:.4f}"))
    df.to_csv(R + f"robustness_{tag}.csv")
    print(f"  saved robustness_{tag}.csv\n")


if __name__ == "__main__":
    main()
