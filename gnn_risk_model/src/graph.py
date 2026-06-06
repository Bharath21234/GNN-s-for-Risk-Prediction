"""Graph construction: sector-based edges with correlation weights."""

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data


def build_sector_edge_index(
    tickers: list[str],
    sectors: dict[str, str],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build undirected intra-sector edges.
    Returns edge_index (2, E) and sector_pairs list.
    """
    src, dst = [], []
    ticker_idx = {t: i for i, t in enumerate(tickers)}

    sector_groups: dict[str, list[int]] = {}
    for t, s in sectors.items():
        if t in ticker_idx:
            sector_groups.setdefault(s, []).append(ticker_idx[t])

    for group in sector_groups.values():
        for a in range(len(group)):
            for b in range(a + 1, len(group)):
                i, j = group[a], group[b]
                src += [i, j]
                dst += [j, i]

    return np.array([src, dst], dtype=np.int64)


def build_correlation_edges(
    returns: pd.DataFrame,
    t: int,
    tickers: list[str],
    sectors: dict[str, str],
    corr_window: int = 60,
    corr_threshold: float = 0.4,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build cross-sector edges for pairs with |rolling correlation| >= threshold.
    Returns edge_index and edge_weights arrays.
    """
    start = max(0, t - corr_window)
    window = returns.iloc[start:t]
    if len(window) < 20:
        return np.zeros((2, 0), dtype=np.int64), np.zeros(0, dtype=np.float32)

    corr = window.corr().values
    N = len(tickers)
    ticker_idx = {tk: i for i, tk in enumerate(tickers)}

    src, dst, weights = [], [], []
    for a in range(N):
        for b in range(a + 1, N):
            ta, tb = tickers[a], tickers[b]
            # Skip intra-sector pairs (already have unconditional edges)
            if sectors.get(ta) == sectors.get(tb):
                continue
            c = corr[a, b]
            if np.isfinite(c) and abs(c) >= corr_threshold:
                src += [a, b]
                dst += [b, a]
                weights += [c, c]

    if not src:
        return np.zeros((2, 0), dtype=np.int64), np.zeros(0, dtype=np.float32)

    return (
        np.array([src, dst], dtype=np.int64),
        np.array(weights, dtype=np.float32),
    )


def build_graph_snapshot(
    returns: pd.DataFrame,
    t: int,
    tickers: list[str],
    sectors: dict[str, str],
    corr_window: int = 60,
    corr_threshold: float = 0.4,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Combine sector edges and correlation edges into one edge_index.
    Also compute edge weights (correlation) for all edges.

    Returns
    -------
    edge_index : (2, E)
    edge_attr  : (E,)  correlation values; sector edges get their computed corr
    """
    N = len(tickers)

    # Rolling correlation matrix for edge weights
    start = max(0, t - corr_window)
    window = returns.iloc[start:t]
    if len(window) >= 10:
        corr_matrix = window.corr().values
    else:
        corr_matrix = np.eye(N)

    # 1. Intra-sector edges (always present)
    sector_ei = build_sector_edge_index(tickers, sectors)

    # Weights for sector edges
    sector_weights = []
    for e in range(sector_ei.shape[1]):
        a, b = sector_ei[0, e], sector_ei[1, e]
        c = corr_matrix[a, b]
        sector_weights.append(float(c) if np.isfinite(c) else 1.0)

    # 2. Cross-sector correlation edges
    cross_ei, cross_weights = build_correlation_edges(
        returns, t, tickers, sectors, corr_window, corr_threshold
    )

    # Merge
    if cross_ei.shape[1] > 0:
        edge_index = np.concatenate([sector_ei, cross_ei], axis=1)
        edge_attr  = np.concatenate([
            np.array(sector_weights, dtype=np.float32),
            cross_weights,
        ])
    else:
        edge_index = sector_ei
        edge_attr  = np.array(sector_weights, dtype=np.float32)

    return edge_index.astype(np.int64), edge_attr


def make_pyg_data(
    x: np.ndarray,
    y: np.ndarray,
    edge_index: np.ndarray,
    edge_attr: np.ndarray,
) -> Data:
    """Wrap arrays into a PyG Data object."""
    return Data(
        x          = torch.from_numpy(x),
        y          = torch.from_numpy(y),
        edge_index = torch.from_numpy(edge_index),
        edge_attr  = torch.from_numpy(edge_attr).unsqueeze(-1),
    )


def precompute_edge_indices(
    returns: pd.DataFrame,
    tickers: list[str],
    sectors: dict[str, str],
    valid_dates: pd.DatetimeIndex,
    corr_window: int = 60,
    corr_threshold: float = 0.4,
    update_freq: int = 21,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Pre-compute graph topology for each time step, only rebuilding every
    `update_freq` days (monthly by default) for CPU efficiency.

    Returns list of (edge_index, edge_attr) tuples, one per time step.
    """
    T = len(valid_dates)
    lookback = len(returns) - T
    edge_snapshots = []
    current_ei, current_ea = None, None

    for idx in range(T):
        if idx % update_freq == 0:
            t = lookback + idx
            current_ei, current_ea = build_graph_snapshot(
                returns, t, tickers, sectors, corr_window, corr_threshold
            )
        edge_snapshots.append((current_ei, current_ea))

    return edge_snapshots
