"""
Step 12 — Post-hoc scalar recalibration (further robustness / discussion
analysis, honestly reported both ways).

Observation: every learned model over-breaches VaR by roughly 2x the 5%
target (Table 1's Kupiec column) while its raw FZ0 is worse than GARCH's.
Both symptoms are consistent with a single, simple cause: the raw (VaR, ES)
magnitudes are systematically too small. We test this directly with the
cheapest possible fix — a single scalar k, applied identically to VaR and ES
(so ES >= VaR is preserved exactly), selected by grid search minimising FZ0
on the *validation* set only, confirmed exactly once on test. Same discipline
as Step 11's ensemble weight.

Reported honestly in both directions:
  (+) it closes/reverses the primary-level gap to GARCH for the MLP and the
      GNN+MLP ensemble, and ties it for the GNN;
  (-) it *widens* the GNN-vs-MLP gap (more decisively, not less) — so the
      scale bias is not what was making the graph model worse than the
      no-graph one; correcting it does not rescue the graph's case on its
      own, it just puts every model on a fairer footing against GARCH.
"""

import os
import sys
import json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import config
from src.metrics import fz0_loss, diebold_mariano, backtest_model

R = config.RESULTS_DIR
ALPHA = config.TAIL_ALPHA
K_GRID = np.arange(0.5, 3.01, 0.05)


def _load_seed_mean(name, split):
    v = np.load(f"{R}{name}_{split}_var_seeds.npy").mean(0)
    e = np.load(f"{R}{name}_{split}_es_seeds.npy").mean(0)
    return v, e


def _load_single(name, split, index=None, columns=None):
    v = pd.read_parquet(f"{R}{name}_{split}_var.parquet")
    e = pd.read_parquet(f"{R}{name}_{split}_es.parquet")
    if index is not None:
        v = v.reindex(index=index, columns=columns)
        e = e.reindex(index=index, columns=columns)
    return v.values, e.values


def _best_k(v_val, e_val, fwd_val):
    best_k, best_l = 1.0, np.inf
    for k in K_GRID:
        l = fz0_loss(fwd_val, k * v_val, k * e_val, alpha=ALPHA, reduce=True)
        if l < best_l:
            best_l, best_k = l, k
    return float(best_k), float(best_l)


def main():
    fwd_val = pd.read_parquet(f"{R}fwd_val.parquet")
    fwd_test = pd.read_parquet(f"{R}fwd_test.parquet")
    dates_test, tickers = fwd_test.index, list(fwd_test.columns)
    returns_raw = pd.read_parquet(config.DATA_DIR + "raw/returns.parquet")

    preds_val, preds_test = {}, {}
    for name in ["gnn", "mlp", "gcn"]:
        preds_val[name] = _load_seed_mean(name, "val")
        preds_test[name] = _load_seed_mean(name, "test")
    preds_val["blend"] = _load_single("blend", "val")
    preds_test["blend"] = _load_single("blend", "test")

    garch_v, garch_e = _load_single("garch", "test", index=dates_test, columns=tickers)
    garch_fz = fz0_loss(fwd_test.values, garch_v, garch_e, ALPHA)
    l_garch = fz0_loss(fwd_test.values, garch_v, garch_e, ALPHA, reduce=False) \
        .reshape(len(fwd_test), -1).mean(1)

    results = {"garch_test_FZ0": garch_fz}
    scaled_losses = {}
    for name in ["gnn", "mlp", "gcn", "blend"]:
        v_val, e_val = preds_val[name]
        v_test, e_test = preds_test[name]
        k, val_fz = _best_k(v_val, e_val, fwd_val.values)
        vs, es = k * v_test, k * e_test

        raw_fz = fz0_loss(fwd_test.values, v_test, e_test, ALPHA)
        scaled_fz = fz0_loss(fwd_test.values, vs, es, ALPHA)
        l_scaled = fz0_loss(fwd_test.values, vs, es, ALPHA, reduce=False) \
            .reshape(len(fwd_test), -1).mean(1)
        scaled_losses[name] = l_scaled

        dm_vs_garch = diebold_mariano(l_scaled, l_garch, horizon=config.HORIZON)
        bt = backtest_model(pd.DataFrame(vs, index=dates_test, columns=tickers),
                             returns_raw, horizon=config.HORIZON, alpha=ALPHA)["summary"]

        results[name] = {
            "k": k, "val_FZ0": val_fz,
            "raw_test_FZ0": raw_fz, "scaled_test_FZ0": scaled_fz,
            "dm_vs_garch_stat": dm_vs_garch["dm_stat"], "dm_vs_garch_p": dm_vs_garch["p_value"],
            "viol_rate": bt["mean_violation_rate"], "kupiec_pass_pct": bt["kupiec_pass_pct"],
        }
        print(f"{name}: k*={k:.2f} (val FZ0={val_fz:.4f}) | "
              f"TEST FZ0 raw={raw_fz:.4f} scaled={scaled_fz:.4f} (GARCH={garch_fz:.4f}) | "
              f"DM vs GARCH p={dm_vs_garch['p_value']:.4f} | "
              f"viol={bt['mean_violation_rate']*100:.2f}% kupiec_pass={bt['kupiec_pass_pct']:.1f}%")

    dm_gnn_vs_mlp = diebold_mariano(scaled_losses["gnn"], scaled_losses["mlp"], horizon=config.HORIZON)
    results["dm_gnn_scaled_vs_mlp_scaled_stat"] = dm_gnn_vs_mlp["dm_stat"]
    results["dm_gnn_scaled_vs_mlp_scaled_p"] = dm_gnn_vs_mlp["p_value"]
    print(f"\nGNN-scaled vs MLP-scaled DM: stat={dm_gnn_vs_mlp['dm_stat']:.3f} "
          f"p={dm_gnn_vs_mlp['p_value']:.4f} (for reference, raw GNN vs MLP DM p=0.032)")

    with open(f"{R}recalibration.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nSaved {R}recalibration.json")


if __name__ == "__main__":
    main()
