"""
GraphSAGE model for joint node-level (VaR, ES) tail-risk forecasting.

Each node (stock) outputs a 95% Value-at-Risk and Expected Shortfall (CVaR),
both as positive loss magnitudes with the structural constraint ES >= VaR >= 0.
The model is trained with the strictly consistent Fissler-Ziegel (FZ0) joint
scoring function against *realised* forward `horizon`-day returns, so training
and forward-looking evaluation use the same, decision-relevant objective.

The identical architecture doubles as the no-graph MLP ablation: passing an
empty edge_index reduces each SAGEConv to its root-weight (per-node linear)
map, isolating the value of the graph while keeping the parameter count fixed.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GraphConv, GCNConv, GATConv, GATv2Conv

_EMPTY_EDGES = torch.zeros((2, 0), dtype=torch.long)

_CONV_TYPES = {"sage": SAGEConv, "graph": GraphConv, "gcn": GCNConv,
               "gat": GATConv, "gat_edge": GATConv, "gatv2": GATv2Conv}
_GAT_HEADS = 4


def _make_conv(conv_type: str, in_channels: int, out_channels: int, attn_dropout: float = 0.0):
    Conv = _CONV_TYPES[conv_type]
    if conv_type in ("gat", "gat_edge", "gatv2"):
        # concat=False averages the heads back to `out_channels`, keeping the
        # layer width identical to the other conv types so the rest of the
        # architecture (LayerNorm, trunk, heads) is unchanged. "gat_edge" adds
        # edge_dim=1 so attention also conditions on |correlation| (not just
        # the learned node representations), combining the "let the model
        # discount weak edges" benefit of attention with the correlation
        # magnitude the GraphConv edge-weight ablation showed carries signal.
        # attn_dropout randomly zeroes attention coefficients per forward pass
        # (Velickovic et al. 2018 used 0.6 on citation graphs); default 0.0
        # keeps prior behaviour exactly, opt-in via GraphSAGERisk(attn_dropout=...).
        kwargs = {"heads": _GAT_HEADS, "concat": False, "dropout": attn_dropout}
        if conv_type == "gat_edge":
            kwargs["edge_dim"] = 1
        return Conv(in_channels, out_channels, **kwargs)
    return Conv(in_channels, out_channels)


class GraphSAGERisk(nn.Module):
    """
    2-layer GNN with joint (VaR, ES) output heads.

    conv_type:
      "sage"  — GraphSAGE (mean aggregation, edge weights ignored; primary model)
      "graph" — GraphConv, which uses per-edge |correlation| weights in the
                aggregation (edge-weight ablation)
      "gcn"   — plain GCN (Kipf & Welling; symmetric-degree-normalised
                aggregation, unweighted topology) — the transductive baseline
      "gat"   — graph attention (Velickovic et al. 2018): learns per-edge
                attention weights instead of fixed (uniform or correlation-
                magnitude) aggregation, so the model can learn to down-weight
                uninformative neighbours rather than averaging all of them in
                uniformly, which the construction ablation shows hurts when
                the edge set is dense (combined sector+correlation graph).
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        conv_type: str = "sage",
        attn_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.dropout = dropout
        self.conv_type = conv_type

        # LayerNorm (per-node, over features), not BatchNorm (per-feature, over
        # the ~100-stock cross-section): a task built to flag *idiosyncratic*
        # single-node tail risk shouldn't re-center every node toward that
        # day's cross-sectional mean at every layer.
        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()
        self.convs.append(_make_conv(conv_type, in_channels, hidden_channels, attn_dropout))
        self.bns.append(nn.LayerNorm(hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(_make_conv(conv_type, hidden_channels, hidden_channels, attn_dropout))
            self.bns.append(nn.LayerNorm(hidden_channels))

        # Shared trunk, then two heads: VaR and a non-negative gap (ES = VaR+gap)
        self.trunk = nn.Sequential(nn.Linear(hidden_channels, 32), nn.ReLU())
        self.var_head = nn.Linear(32, 1)
        self.gap_head = nn.Linear(32, 1)

        # Initialise outputs near a realistic 5-day loss scale (~3% VaR, ~1% gap)
        # so FZ0 optimisation starts in a stable region.
        with torch.no_grad():
            self.var_head.bias.fill_(_softplus_inv(0.03))
            self.gap_head.bias.fill_(_softplus_inv(0.01))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_weight: torch.Tensor | None = None) -> torch.Tensor:
        for conv, bn in zip(self.convs, self.bns):
            if self.conv_type == "graph" and edge_weight is not None and edge_index.size(1) > 0:
                x = conv(x, edge_index, edge_weight)
            elif self.conv_type == "gat_edge" and edge_weight is not None and edge_index.size(1) > 0:
                x = conv(x, edge_index, edge_attr=edge_weight.unsqueeze(-1))
            else:
                x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        h = self.trunk(x)
        # Floor at 5e-3: typical 5-day 95% VaR here is ~3-6%, so this never
        # constrains a sane prediction. It exists to stop FZ0 blowing up —
        # the loss divides by es_pos, so if a single node's ES ever collapses
        # toward 0 (it previously floored at 1e-4, ~10-50x below the typical
        # scale) one breach on that node/day could send the whole batch loss
        # to the hundreds and destabilise training; this keeps es_pos in a
        # numerically safe range without biasing normal predictions.
        var = F.softplus(self.var_head(h)).clamp(min=5e-3)   # (N,1) positive
        gap = F.softplus(self.gap_head(h))                   # (N,1) >= 0
        es  = var + gap                                      # ES >= VaR
        return torch.cat([var, es], dim=-1)                  # (N, 2)


def _softplus_inv(y: float) -> float:
    """Inverse softplus: x such that softplus(x) = y."""
    return float(np.log(np.expm1(y)))


# ── Differentiable FZ0 loss (matches src.metrics.fz0_loss) ─────────────────────

def fz0_loss_torch(
    realised: torch.Tensor,
    var_pos: torch.Tensor,
    es_pos: torch.Tensor,
    alpha: float = 0.05,
) -> torch.Tensor:
    """
    Fissler-Ziegel FZ0 loss (Patton, Ziegel & Chen 2019), positive-magnitude
    form.  `realised` is the signed forward return; var_pos, es_pos > 0.
    """
    es_pos = es_pos.clamp(min=5e-3)                   # see forward()'s clamp note
    breach = (-realised >= var_pos).float()          # non-diff indicator (std.)
    loss = breach * (-var_pos - realised) / (alpha * es_pos) \
        + var_pos / es_pos + torch.log(es_pos) - 1.0
    return loss.mean()


class EarlyStopper:
    def __init__(self, patience: int = 20, min_delta: float = 1e-6) -> None:
        self.patience   = patience
        self.min_delta  = min_delta
        self.best_loss  = float("inf")
        self.counter    = 0
        self.best_state = None

    def step(self, val_loss: float, model: nn.Module) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss  = val_loss
            self.counter    = 0
            self.best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
        return self.counter >= self.patience

    def restore(self, model: nn.Module) -> None:
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


def _edges_for(edge_snapshots, t, use_graph: bool, device):
    """Return (edge_index, edge_weight). Weight = |correlation| for weighted convs."""
    if not use_graph:
        return _EMPTY_EDGES.to(device), None
    ei, ea = edge_snapshots[t]
    edge_index = torch.tensor(ei, dtype=torch.long, device=device)
    edge_weight = torch.tensor(np.abs(ea), dtype=torch.float32, device=device) \
        if ea is not None and len(ea) else None
    return edge_index, edge_weight


def train_one_epoch(
    model: GraphSAGERisk,
    features: np.ndarray,
    fwd_returns: np.ndarray,
    edge_snapshots: list,
    indices: np.ndarray,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    alpha: float = 0.05,
    use_graph: bool = True,
    dropedge: float = 0.0,
) -> float:
    model.train()
    total_loss = 0.0
    perm = np.random.permutation(indices)

    for t in perm:
        x = torch.tensor(features[t], dtype=torch.float32, device=device)
        r = torch.tensor(fwd_returns[t], dtype=torch.float32, device=device)
        edge_index, edge_weight = _edges_for(edge_snapshots, t, use_graph, device)

        # DropEdge (Rong et al. 2020): randomly remove a fraction of edges each
        # training step only (never at eval), a structural regulariser distinct
        # from attention dropout (which reweights edges that survive, rather
        # than removing them from the neighbourhood before message passing).
        if dropedge > 0.0 and edge_index.size(1) > 0:
            E = edge_index.size(1)
            keep = torch.rand(E, device=device) >= dropedge
            edge_index = edge_index[:, keep]
            if edge_weight is not None:
                edge_weight = edge_weight[keep]

        optimizer.zero_grad()
        out = model(x, edge_index, edge_weight)      # (N, 2)
        loss = fz0_loss_torch(r, out[:, 0], out[:, 1], alpha=alpha)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(perm)


@torch.no_grad()
def evaluate(
    model: GraphSAGERisk,
    features: np.ndarray,
    fwd_returns: np.ndarray,
    edge_snapshots: list,
    indices: np.ndarray,
    device: torch.device,
    alpha: float = 0.05,
    use_graph: bool = True,
) -> tuple[float, np.ndarray]:
    """Returns (mean FZ0 loss, predictions array of shape (T, N, 2) = [VaR, ES])."""
    model.eval()
    all_preds = []
    total_loss = 0.0

    for t in indices:
        x = torch.tensor(features[t], dtype=torch.float32, device=device)
        r = torch.tensor(fwd_returns[t], dtype=torch.float32, device=device)
        edge_index, edge_weight = _edges_for(edge_snapshots, t, use_graph, device)

        out = model(x, edge_index, edge_weight)      # (N, 2)
        total_loss += fz0_loss_torch(r, out[:, 0], out[:, 1], alpha=alpha).item()
        all_preds.append(out.cpu().numpy())

    return total_loss / len(indices), np.stack(all_preds)   # (T, N, 2)


def train_model(
    model: GraphSAGERisk,
    features: np.ndarray,
    fwd_returns: np.ndarray,
    edge_snapshots: list,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    num_epochs: int = 150,
    patience: int = 20,
    alpha: float = 0.05,
    use_graph: bool = True,
    seed: int = 0,
    device: torch.device | None = None,
    verbose: bool = True,
    dropedge: float = 0.0,
) -> dict:
    if device is None:
        device = torch.device("cpu")

    torch.manual_seed(seed)
    np.random.seed(seed)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    stopper   = EarlyStopper(patience=patience)

    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, num_epochs + 1):
        train_loss = train_one_epoch(
            model, features, fwd_returns, edge_snapshots, train_indices,
            optimizer, device, alpha=alpha, use_graph=use_graph, dropedge=dropedge,
        )
        val_loss, _ = evaluate(
            model, features, fwd_returns, edge_snapshots, val_indices,
            device, alpha=alpha, use_graph=use_graph,
        )
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if verbose and (epoch % 10 == 0 or epoch == 1):
            print(f"  Epoch {epoch:>4d} | train FZ0={train_loss:.5f} | val FZ0={val_loss:.5f}")

        if stopper.step(val_loss, model):
            if verbose:
                print(f"  Early stopping at epoch {epoch}.")
            break

    stopper.restore(model)
    if verbose:
        print(f"  Best val FZ0: {stopper.best_loss:.5f}")
    history["best_val_loss"] = stopper.best_loss
    return history
