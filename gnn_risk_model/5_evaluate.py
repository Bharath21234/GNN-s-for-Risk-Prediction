"""
Step 5 — Evaluate and compare all models (BNAIC evaluation).

Primary (forward-looking, decision-relevant):
  * FZ0 joint (VaR, ES) loss and pinball VaR loss vs *realised* forward returns
  * Diebold-Mariano significance (HAC + HLN correction) for GNN vs GARCH / GJR / MLP
  * multi-seed mean±std and fraction of seeds beating each baseline
Calibration:
  * VaR: Kupiec POF + Christoffersen CC (now correctly specified at alpha=5%)
  * ES : Acerbi-Szekely Test 2 + exceedance-residual t-test
  * non-overlapping (every-5th-day) VaR backtest as a robustness check
Ablation:
  * GNN vs MLP (identical model, no graph) — isolates the value of graph structure
Secondary (reported, not headline):
  * RMSE of predicted ES vs the backward-looking rolling-CVaR label (reconstruction)
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(__file__))
import config
from src.metrics import (
    fz0_loss, pinball_loss, diebold_mariano, backtest_model,
    acerbi_szekely_z2, es_residual_test, rmse, moving_block_bootstrap_ci,
)

R = config.RESULTS_DIR
os.makedirs(R + "plots/", exist_ok=True)
sns.set_theme(style="whitegrid", palette="tab10")
ALPHA   = config.TAIL_ALPHA
HORIZON = config.HORIZON
COLORS  = {"GNN": "#2196F3", "MLP (no graph)": "#9C27B0",
           "GCN": "#00BCD4", "GARCH(1,1)": "#F44336",
           "GJR-GARCH": "#FF9800", "Historical Sim": "#4CAF50",
           "GNN+MLP Ensemble": "#3F51B5"}

# Model registry. SEED models ship per-seed prediction stacks ({m}_test_var_seeds.npy);
# SINGLE models ship one VaR/ES parquet ({m}_test_var.parquet). Any that are present
# on disk are auto-detected and folded into every table — so adding the GCN or
# historical-simulation baselines needs no change here.
SEED_CANDIDATES   = ["gnn", "mlp", "gcn"]
SINGLE_CANDIDATES = ["garch", "gjr", "histsim", "blend"]
LABEL = {"gnn": "GNN", "mlp": "MLP (no graph)", "gcn": "GCN",
         "garch": "GARCH(1,1)", "gjr": "GJR-GARCH", "histsim": "Historical Sim",
         "blend": "GNN+MLP Ensemble"}


# ── Loading ─────────────────────────────────────────────────────────────────────

def _load_seed_stack(name, split, q):
    return np.load(R + f"{name}_{split}_{q}_seeds.npy")          # (S, T, N)


def main():
    print("Loading predictions and labels...")
    fwd     = pd.read_parquet(R + "fwd_test.parquet")            # realised fwd returns
    targets = pd.read_parquet(R + "targets_test.parquet")        # rolling CVaR (reconstruction)
    returns_raw = pd.read_parquet(config.DATA_DIR + "raw/returns.parquet")
    dates, tickers = fwd.index, list(fwd.columns)
    Rmat = fwd.values                                            # (T, N) realised

    # Detect which baselines are present on disk (gnn is always required).
    seed_models   = [m for m in SEED_CANDIDATES
                     if os.path.exists(R + f"{m}_test_var_seeds.npy")]
    single_models = [m for m in SINGLE_CANDIDATES
                     if os.path.exists(R + f"{m}_test_var.parquet")]
    MODELS = seed_models + single_models          # gnn first (reference for DM)
    print(f"  Models detected: {', '.join(LABEL[m] for m in MODELS)}")

    preds = {}
    for name in seed_models:
        preds[name] = {
            "var": _load_seed_stack(name, "test", "var"),        # (S, T, N)
            "es":  _load_seed_stack(name, "test", "es"),
        }
    # Single VaR/ES baselines, reindexed to the evaluation grid (S=1 axis)
    def _reidx(fn):
        return pd.read_parquet(R + fn).reindex(index=dates, columns=tickers).values
    for name in single_models:
        preds[name] = {"var": _reidx(f"{name}_test_var.parquet")[None],
                       "es":  _reidx(f"{name}_test_es.parquet")[None]}

    # ── Per-seed scalar losses + ensemble predictions ──────────────────────────
    def seed_losses(name):
        v, e = preds[name]["var"], preds[name]["es"]
        S = v.shape[0]
        fz = [fz0_loss(Rmat, v[s], e[s], ALPHA) for s in range(S)]
        pb = [pinball_loss(Rmat, v[s], ALPHA)   for s in range(S)]
        return np.array(fz), np.array(pb)

    def ensemble(name):
        return preds[name]["var"].mean(0), preds[name]["es"].mean(0)   # (T,N),(T,N)

    # ── 1. Primary forward-looking loss table ──────────────────────────────────
    print("\n── Forward-looking losses (vs realised returns) ─────────────────────")
    rows = []
    seed_fz = {}
    for name in MODELS:
        fz, pb = seed_losses(name)
        seed_fz[name] = fz
        rows.append({
            "model":         LABEL[name],
            "FZ0_mean":      float(fz.mean()),  "FZ0_std": float(fz.std()),
            "pinball_mean":  float(pb.mean()),  "pinball_std": float(pb.std()),
            "n_seeds":       len(fz),
        })
    loss_df = pd.DataFrame(rows).set_index("model")
    # fraction of GNN seeds beating each single-value baseline
    for base in [m for m in MODELS if m != "gnn"]:
        thr = seed_fz[base].mean()
        loss_df.loc["GNN", f"GNN seeds < {LABEL[base]}"] = \
            f"{np.mean(seed_fz['gnn'] < thr)*100:.0f}%"
    # moving-block-bootstrap 95% CI on the mean FZ0 (ensemble per-date loss series).
    # Also record the ensemble point estimate itself: FZ0_mean above averages each
    # seed's *independent* loss, but the DM tests and this CI both score the
    # *ensembled* (seed-averaged) prediction, which is what you'd actually deploy
    # and is a strictly less noisy estimator -- report both rather than let the
    # headline number quietly understate what the significance tests already use.
    for name in MODELS:
        ev, ee = ensemble(name)
        series = fz0_loss(Rmat, ev, ee, ALPHA, reduce=False).reshape(len(fwd), -1).mean(1)
        ci = moving_block_bootstrap_ci(series, block=HORIZON, n_boot=2000)
        loss_df.loc[LABEL[name], "FZ0_ensemble"] = float(series.mean())
        loss_df.loc[LABEL[name], "FZ0_ci_lo"] = ci["lo"]
        loss_df.loc[LABEL[name], "FZ0_ci_hi"] = ci["hi"]
    print(loss_df.to_string(float_format=lambda x: f"{x:.5f}"))
    loss_df.to_csv(R + "forward_losses.csv")

    # ── 2. Diebold-Mariano significance (ensemble, per-date loss series) ────────
    print("\n── Diebold-Mariano (FZ0 loss; neg stat ⇒ GNN better) ────────────────")
    gnn_v, gnn_e = ensemble("gnn")
    dm_rows = []
    for base in [m for m in MODELS if m != "gnn"]:
        bv, be = ensemble(base)
        la = fz0_loss(Rmat, gnn_v, gnn_e, ALPHA, reduce=False).reshape(len(dates), -1).mean(1)
        lb = fz0_loss(Rmat, bv,    be,    ALPHA, reduce=False).reshape(len(dates), -1).mean(1)
        dm = diebold_mariano(la, lb, horizon=HORIZON)
        dm_rows.append({"comparison": f"GNN vs {LABEL[base]}", **dm})
        print(f"  GNN vs {LABEL[base]:16s}  DM={dm['dm_stat']:+.3f}  p={dm['p_value']:.4g}  "
              f"mean_diff={dm['mean_diff']:+.5f}")
    pd.DataFrame(dm_rows).to_csv(R + "dm_tests.csv", index=False)

    # ── 3. VaR backtests (Kupiec + CC, correctly specified at 5%) ──────────────
    print("\n── VaR backtests (Kupiec POF + Christoffersen CC; target 5%) ────────")
    var_bt, es_bt = {}, {}
    for name in MODELS:
        v, e = ensemble(name)
        vdf = pd.DataFrame(v, index=dates, columns=tickers)
        bt  = backtest_model(vdf, returns_raw, horizon=HORIZON, alpha=ALPHA)
        var_bt[name] = bt["summary"]
        # ES calibration on all (t,n)
        z2 = acerbi_szekely_z2(Rmat.flatten(), v.flatten(), e.flatten(), ALPHA)
        rt = es_residual_test(Rmat.flatten(), v.flatten(), e.flatten())
        es_bt[name] = {"z2": z2["z2"], "z2_p": z2["p_value"],
                       "es_resid_ratio": rt["mean_ratio"], "es_resid_p": rt["p_value"]}
        s = bt["summary"]
        print(f"  {LABEL[name]:16s}  viol={s['mean_violation_rate']*100:5.2f}%  "
              f"kupiec_pass={s['kupiec_pass_pct']:5.1f}%  cc_pass={s['cc_pass_pct']:5.1f}%")
    pd.DataFrame(var_bt).T.to_csv(R + "var_backtest.csv")

    print("\n── ES backtests (Acerbi-Szekely Z2; Z2<0 ⇒ risk under-estimated) ────")
    for name in MODELS:
        b = es_bt[name]
        print(f"  {LABEL[name]:16s}  Z2={b['z2']:+.3f} (p={b['z2_p']:.3g})  "
              f"ES-resid ratio={b['es_resid_ratio']:.3f} (p={b['es_resid_p']:.3g})")
    pd.DataFrame(es_bt).T.to_csv(R + "es_backtest.csv")

    # ── 4. Non-overlapping VaR backtest (every 5th day) — robustness (P1-5) ─────
    print("\n── Non-overlapping VaR backtest (every 5th day) ─────────────────────")
    nov_rows = {}
    for name in MODELS:
        v, _ = ensemble(name)
        vdf = pd.DataFrame(v, index=dates, columns=tickers).iloc[::HORIZON]
        bt  = backtest_model(vdf, returns_raw, horizon=HORIZON, alpha=ALPHA)
        nov_rows[name] = bt["summary"]
        s = bt["summary"]
        print(f"  {LABEL[name]:16s}  viol={s['mean_violation_rate']*100:5.2f}%  "
              f"kupiec_pass={s['kupiec_pass_pct']:5.1f}%")
    pd.DataFrame(nov_rows).T.to_csv(R + "nonoverlap_var_backtest.csv")

    # ── 5. Sector-level forward loss (the story) ───────────────────────────────
    print("\n── Sector-level FZ0 loss (GNN vs best GARCH) ────────────────────────")
    sec_map = {t: config.STOCKS.get(t, "Unknown") for t in tickers}
    col_idx = {t: i for i, t in enumerate(tickers)}
    ens = {n: ensemble(n) for n in MODELS}
    gv, ge = ens["gnn"]
    srows = []
    for sec in sorted(set(sec_map.values())):
        cols = [col_idx[t] for t in tickers if sec_map[t] == sec]
        if not cols:
            continue
        rr = Rmat[:, cols]
        d = {"sector": sec, "n_stocks": len(cols)}
        for n in MODELS:
            v, e = ens[n]
            d[f"{LABEL[n]} FZ0"] = fz0_loss(rr, v[:, cols], e[:, cols], ALPHA)
        d["GNN vs GARCH Δ%"] = (d["GARCH(1,1) FZ0"] - d["GNN FZ0"]) / abs(d["GARCH(1,1) FZ0"]) * 100
        # per-sector Diebold-Mariano (GNN vs GARCH and vs MLP) on per-date FZ0 loss
        la = fz0_loss(rr, gv[:, cols], ge[:, cols], ALPHA, reduce=False).reshape(len(dates), -1).mean(1)
        for base, tag in [(b, t) for b, t in [("garch", "GARCH"), ("mlp", "MLP")] if b in MODELS]:
            bv, be = ens[base]
            lb = fz0_loss(rr, bv[:, cols], be[:, cols], ALPHA, reduce=False).reshape(len(dates), -1).mean(1)
            dm = diebold_mariano(la, lb, horizon=HORIZON)
            d[f"DM vs {tag}"] = dm["dm_stat"]
            d[f"p vs {tag}"]  = dm["p_value"]
        srows.append(d)
    sector_df = pd.DataFrame(srows).set_index("sector")
    gnn_wins = (sector_df["GNN FZ0"] < sector_df["GARCH(1,1) FZ0"]).sum()
    print(sector_df.to_string(float_format=lambda x: f"{x:.4f}"))
    print(f"  GNN beats GARCH on FZ0 in {gnn_wins}/{len(sector_df)} sectors")
    sector_df.to_csv(R + "sector_forward_loss.csv")

    # ── 6. Secondary: reconstruction RMSE (ES vs rolling-CVaR label) ────────────
    print("\n── (secondary) Reconstruction RMSE: ES vs rolling-CVaR label ────────")
    rec_rows = []
    yt = targets.values.flatten()
    for name in MODELS:
        _, e = ensemble(name)
        m = np.isfinite(yt) & np.isfinite(e.flatten())
        rec_rows.append({"model": LABEL[name], "recon_RMSE": rmse(yt[m], e.flatten()[m])})
    rec_df = pd.DataFrame(rec_rows).set_index("model")
    print(rec_df.to_string(float_format=lambda x: f"{x:.5f}"))
    rec_df.to_csv(R + "reconstruction_rmse.csv")

    # ── Combined JSON summary ──────────────────────────────────────────────────
    with open(R + "backtest_summary.json", "w") as f:
        json.dump({"var_backtest": var_bt, "es_backtest": es_bt,
                   "forward_losses": loss_df.to_dict(),
                   "dm_tests": dm_rows,
                   "sector_gnn_wins": int(gnn_wins)}, f, indent=2, default=float)

    # ── Plots ──────────────────────────────────────────────────────────────────
    print("\nGenerating plots...")
    _plot_forward_loss(loss_df)
    _plot_sector(sector_df)
    _plot_var_violations(var_bt, {n: ensemble(n)[0] for n in MODELS},
                         returns_raw, dates, tickers, LABEL, MODELS)
    _plot_timeseries(ens, fwd, LABEL, MODELS)
    print(f"\nAll results saved to {R}\nStep 5 complete.\n")


# ── Plots ────────────────────────────────────────────────────────────────────────

def _plot_forward_loss(loss_df):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, col, err, title in [(axes[0], "FZ0_mean", "FZ0_std", "FZ0 joint (VaR, ES) loss"),
                                 (axes[1], "pinball_mean", "pinball_std", "Pinball VaR loss")]:
        models = loss_df.index.tolist()
        ax.bar(models, loss_df[col], yerr=loss_df[err],
               color=[COLORS.get(m, "gray") for m in models], capsize=4,
               edgecolor="black", linewidth=0.5)
        ax.set_title(title); ax.set_xticklabels(models, rotation=15, ha="right", fontsize=8)
    fig.suptitle("Forward-looking losses (lower = better) — mean±std over seeds",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(); fig.savefig("results/plots/forward_loss.png", dpi=150); plt.close(fig)


def _plot_sector(sector_df):
    cols = [c for c in sector_df.columns if c.endswith("FZ0")]
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(sector_df[cols], annot=True, fmt=".4f", cmap="RdYlGn_r",
                linewidths=0.5, ax=ax, cbar_kws={"label": "FZ0 loss"})
    ax.set_title("Sector-level FZ0 loss", fontsize=12, fontweight="bold")
    fig.tight_layout(); fig.savefig("results/plots/sector_forward_loss.png", dpi=150); plt.close(fig)


def _plot_var_violations(var_bt, var_ens, returns_raw, dates, tickers, LABEL, models):
    fig, ax = plt.subplots(figsize=(12, 4))
    for name in models:
        vdf = pd.DataFrame(var_ens[name], index=dates, columns=tickers)
        bt = backtest_model(vdf, returns_raw, horizon=HORIZON, alpha=ALPHA)
        rates = sorted(bt["kupiec"][t]["violation_rate"] * 100 for t in tickers if t in bt["kupiec"])
        ax.plot(rates, label=LABEL[name], linewidth=1.5, color=COLORS.get(LABEL[name]))
    ax.axhline(5.0, color="black", linewidth=1.5, linestyle="--", label="Target 5%")
    ax.set_title("VaR violation rates by stock (sorted)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Stock rank"); ax.set_ylabel("Violation rate (%)"); ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig("results/plots/var_violation_rates.png", dpi=150); plt.close(fig)


def _plot_timeseries(ens, fwd, LABEL, models):
    fig, ax = plt.subplots(figsize=(13, 4))
    realised_loss = (-fwd).mean(axis=1)
    ax.plot(realised_loss.index, realised_loss.values, color="black", alpha=0.35,
            linewidth=0.8, label="Realised mean loss")
    for name in models:
        _, e = ens[name]
        ax.plot(fwd.index, e.mean(axis=1), linewidth=1.1, label=f"{LABEL[name]} ES",
                color=COLORS.get(LABEL[name]))
    ax.set_title("Average 5-day 95% ES vs realised loss — test period",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Date"); ax.set_ylabel("5-day loss"); ax.legend(fontsize=8, ncol=3)
    fig.tight_layout(); fig.savefig("results/plots/es_timeseries.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
