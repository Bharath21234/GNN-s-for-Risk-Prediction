import torch
import pandas as pd
from torch_geometric.data import Data


def fully_connected_edge_index(num_nodes: int) -> torch.Tensor:
    """
    Fully connected undirected graph without self-loops.
    Works for any num_nodes >= 2.

    Returns:
        edge_index: LongTensor of shape [2, E]
    """
    edges = []

    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:           # no self-loops
                edges.append((i, j))

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return edge_index

def make_daily_graphs(panel: pd.DataFrame, dates, tickers, device="cpu"):
    """
    Each day t => Data(x, edge_index, y) where
      node x: (3, F) features at day t 
      y: (3,) labels at day t+1 (tail flag)
    """
    edge_index = fully_connected_edge_index(len(tickers))

    # store all the daily graph
    graphs = []

    for dt in dates:
        dt_next = panel.index[panel.index.get_loc(dt) + 1]

        x_list = []
        y_list = []
        for tkr in tickers:
            x_list.append([
                panel.loc[dt, f"Return_{tkr}"],
                panel.loc[dt, f"vol_{tkr}"],
                panel.loc[dt, f"VaR_{tkr}"],
                panel.loc[dt, f"Tail_Risk_{tkr}"], # this day label
            ])
            y_list.append(panel.loc[dt_next, f"Tail_Risk_{tkr}"])  # next-day label

        x = torch.tensor(x_list, dtype=torch.float32) # shape(3,4)
        y = torch.tensor(y_list, dtype=torch.long) #shape(3,)

        # nodes,edges,label
        data = Data(x=x, edge_index=edge_index, y=y)

        graphs.append(data)

    return graphs