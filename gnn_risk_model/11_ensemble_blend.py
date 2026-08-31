"""
Step 11 — Post-hoc ensemble of the GNN and MLP forecasts (reviewer-value addition).

Motivation: Sec. 5's DM tests show the GNN is *noisier* than the MLP but not
redundant with it — a natural next question is whether their forecasts are
*complementary*. A simple convex blend, var/es = w*GNN + (1-w)*MLP, tests this
directly with no additional training.

Discipline: the blend weight w is chosen by a grid search over [0, 1] that
minimises FZ0 on the *validation* set only, independently at each tail level.
The test set is touched exactly once, to report the already-committed
(w, split) configuration's forward loss, DM significance and calibration —
the same protocol the sensitivity study in Step 10 deliberately avoids using
for model selection, applied correctly here because the selection itself
never sees the test set.

Blend predictions are written in the exact single-model parquet layout that
Step 5 / Step 9 already auto-detect ({name}_{split}_{var,es}.parquet and the
{tag} variants for extra tail levels), so registering "blend" as a
SINGLE_CANDIDATE in 5_evaluate.py folds it into every headline table with no
duplicated statistics code.
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


def _alpha_tag(a: float) -> str:
    return "" if abs(a - config.TAIL_ALPHA) < 1e-12 else f"_a{round(a * 1000):03d}"


def _load_seed_mean(name: str, split: str, tag: str):
    v = np.load(f"{R}{name}{tag}_{split}_var_seeds.npy").mean(0)
    e = np.load(f"{R}{name}{tag}_{split}_es_seeds.npy").mean(0)
    return v, e


def _select_weight(gv, ge, mv, me, fwd_val, alpha):
    best_w, best_loss = 0.0, np.inf
    for w in np.arange(0.0, 1.001, 0.05):
        v = w * gv + (1 - w) * mv
        e = w * ge + (1 - w) * me
        l = fz0_loss(fwd_val, v, e, alpha=alpha, reduce=True)
        if l < best_loss:
            best_loss, best_w = l, w
    return float(best_w), float(best_loss)


def run_level(alpha: float):
    tag = _alpha_tag(alpha)
    fwd_val = pd.read_parquet(f"{R}fwd_val.parquet")
    fwd_test = pd.read_parquet(f"{R}fwd_test.parquet")
    dates_val, dates_test = fwd_val.index, fwd_test.index
    tickers = list(fwd_test.columns)

    gv_val, ge_val = _load_seed_mean("gnn", "val", tag)
    mv_val, me_val = _load_seed_mean("mlp", "val", tag)
    gv_test, ge_test = _load_seed_mean("gnn", "test", tag)
    mv_test, me_test = _load_seed_mean("mlp", "test", tag)

    w, val_fz = _select_weight(gv_val, ge_val, mv_val, me_val, fwd_val.values, alpha)

    bv_val = w * gv_val + (1 - w) * mv_val
    be_val = w * ge_val + (1 - w) * me_val
    bv_test = w * gv_test + (1 - w) * mv_test
    be_test = w * ge_test + (1 - w) * me_test

    pd.DataFrame(bv_val, index=dates_val, columns=tickers).to_parquet(f"{R}blend{tag}_val_var.parquet")
    pd.DataFrame(be_val, index=dates_val, columns=tickers).to_parquet(f"{R}blend{tag}_val_es.parquet")
    pd.DataFrame(bv_test, index=dates_test, columns=tickers).to_parquet(f"{R}blend{tag}_test_var.parquet")
    pd.DataFrame(be_test, index=dates_test, columns=tickers).to_parquet(f"{R}blend{tag}_test_es.parquet")

    # ── Confirmatory stats on TEST, computed once, reported as-is ──────────────
    returns_raw = pd.read_parquet(config.DATA_DIR + "raw/returns.parquet")
    gnn_fz = fz0_loss(fwd_test.values, gv_test, ge_test, alpha)
    mlp_fz = fz0_loss(fwd_test.values, mv_test, me_test, alpha)
    blend_fz = fz0_loss(fwd_test.values, bv_test, be_test, alpha)

    la = fz0_loss(fwd_test.values, bv_test, be_test, alpha, reduce=False).reshape(len(fwd_test), -1).mean(1)
    lb = fz0_loss(fwd_test.values, mv_test, me_test, alpha, reduce=False).reshape(len(fwd_test), -1).mean(1)
    lg = fz0_loss(fwd_test.values, gv_test, ge_test, alpha, reduce=False).reshape(len(fwd_test), -1).mean(1)
    dm_vs_mlp = diebold_mariano(la, lb, horizon=config.HORIZON)
    dm_vs_gnn = diebold_mariano(la, lg, horizon=config.HORIZON)

    bt = backtest_model(pd.DataFrame(bv_test, index=dates_test, columns=tickers),
                         returns_raw, horizon=config.HORIZON, alpha=alpha)["summary"]

    result = {
        "alpha": alpha, "w": w, "val_FZ0": val_fz,
        "gnn_test_FZ0": gnn_fz, "mlp_test_FZ0": mlp_fz, "blend_test_FZ0": blend_fz,
        "dm_vs_mlp_stat": dm_vs_mlp["dm_stat"], "dm_vs_mlp_p": dm_vs_mlp["p_value"],
        "dm_vs_gnn_stat": dm_vs_gnn["dm_stat"], "dm_vs_gnn_p": dm_vs_gnn["p_value"],
        "viol_rate": bt["mean_violation_rate"], "kupiec_pass_pct": bt["kupiec_pass_pct"],
        "cc_pass_pct": bt["cc_pass_pct"],
    }

    # DM vs each single-value baseline present on disk (garch/gjr/histsim), so
    # the "does the blend close the gap to the strong parametric baselines?"
    # question has a direct, honestly-computed answer too.
    for base in ["garch", "gjr", "histsim"]:
        bpath = f"{R}{base}{tag}_test_var.parquet"
        if not os.path.exists(bpath):
            continue
        bv = pd.read_parquet(bpath).reindex(index=dates_test, columns=tickers).values
        be = pd.read_parquet(f"{R}{base}{tag}_test_es.parquet").reindex(index=dates_test, columns=tickers).values
        base_fz = fz0_loss(fwd_test.values, bv, be, alpha)
        lbase = fz0_loss(fwd_test.values, bv, be, alpha, reduce=False).reshape(len(fwd_test), -1).mean(1)
        dm_base = diebold_mariano(la, lbase, horizon=config.HORIZON)
        result[f"{base}_test_FZ0"] = base_fz
        result[f"dm_vs_{base}_stat"] = dm_base["dm_stat"]
        result[f"dm_vs_{base}_p"] = dm_base["p_value"]

    print(f"alpha={alpha}: w*={w:.2f} (val FZ0={val_fz:.5f}) | "
          f"TEST FZ0  gnn={gnn_fz:.5f}  mlp={mlp_fz:.5f}  blend={blend_fz:.5f} | "
          f"DM vs MLP p={dm_vs_mlp['p_value']:.4f} | DM vs GNN p={dm_vs_gnn['p_value']:.4f} | "
          f"DM vs GARCH p={result.get('dm_vs_garch_p', float('nan')):.4f} | "
          f"viol={bt['mean_violation_rate']*100:.2f}% kupiec_pass={bt['kupiec_pass_pct']:.1f}%")
    return result, tag


def _append_robustness_row(tag: str, alpha: float, res: dict):
    path = f"{R}robustness{tag}.csv"
    if not os.path.exists(path):
        return
    df = pd.read_csv(path, index_col=0)
    df.loc["GNN+MLP Ensemble"] = {
        "FZ0": res["blend_test_FZ0"], "pinball": np.nan,
        "DM_vs_GNN": res["dm_vs_gnn_stat"], "p_vs_GNN": res["dm_vs_gnn_p"],
        "viol_rate_%": res["viol_rate"] * 100, "kupiec_pass_%": res["kupiec_pass_pct"],
        "target_viol_%": alpha * 100,
    }
    df.to_csv(path)


def main():
    results = {}
    res0, _ = run_level(config.TAIL_ALPHA)
    results["primary"] = res0
    for a in getattr(config, "EXTRA_TAIL_ALPHAS", []):
        res, tag = run_level(a)
        results[f"a{round(a * 1000):03d}"] = res
        _append_robustness_row(tag, a, res)

    with open(f"{R}ensemble_blend.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nSaved blend{{,_a025}}_{{val,test}}_{{var,es}}.parquet and {R}ensemble_blend.json")


if __name__ == "__main__":
    main()
