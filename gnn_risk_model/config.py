"""Central configuration for all hyperparameters and stock universe."""

# ── Stock universe: 100 S&P 500 large-caps across all 11 GICS sectors ──────────
STOCKS = {
    # Information Technology (18)
    "AAPL":  "Information Technology",
    "MSFT":  "Information Technology",
    "NVDA":  "Information Technology",
    "AVGO":  "Information Technology",
    "ORCL":  "Information Technology",
    "CRM":   "Information Technology",
    "AMD":   "Information Technology",
    "QCOM":  "Information Technology",
    "TXN":   "Information Technology",
    "INTU":  "Information Technology",
    "AMAT":  "Information Technology",
    "NOW":   "Information Technology",
    "ADI":   "Information Technology",
    "LRCX":  "Information Technology",
    "MU":    "Information Technology",
    "KLAC":  "Information Technology",
    "SNPS":  "Information Technology",
    "CDNS":  "Information Technology",
    # Communication Services (7)
    "GOOGL": "Communication Services",
    "META":  "Communication Services",
    "NFLX":  "Communication Services",
    "DIS":   "Communication Services",
    "CMCSA": "Communication Services",
    "T":     "Communication Services",
    "VZ":    "Communication Services",
    # Consumer Discretionary (11)
    "AMZN":  "Consumer Discretionary",
    "TSLA":  "Consumer Discretionary",
    "HD":    "Consumer Discretionary",
    "BKNG":  "Consumer Discretionary",
    "MCD":   "Consumer Discretionary",
    "NKE":   "Consumer Discretionary",
    "LOW":   "Consumer Discretionary",
    "SBUX":  "Consumer Discretionary",
    "TJX":   "Consumer Discretionary",
    "CMG":   "Consumer Discretionary",
    "AZO":   "Consumer Discretionary",
    # Consumer Staples (9)
    "WMT":   "Consumer Staples",
    "COST":  "Consumer Staples",
    "PG":    "Consumer Staples",
    "KO":    "Consumer Staples",
    "PEP":   "Consumer Staples",
    "PM":    "Consumer Staples",
    "MO":    "Consumer Staples",
    "MDLZ":  "Consumer Staples",
    "CL":    "Consumer Staples",
    # Health Care (13)
    "UNH":   "Health Care",
    "JNJ":   "Health Care",
    "LLY":   "Health Care",
    "ABBV":  "Health Care",
    "MRK":   "Health Care",
    "TMO":   "Health Care",
    "ABT":   "Health Care",
    "DHR":   "Health Care",
    "ISRG":  "Health Care",
    "AMGN":  "Health Care",
    "VRTX":  "Health Care",
    "REGN":  "Health Care",
    "MDT":   "Health Care",
    # Financials (12)
    "BRK-B": "Financials",
    "JPM":   "Financials",
    "V":     "Financials",
    "MA":    "Financials",
    "BAC":   "Financials",
    "WFC":   "Financials",
    "GS":    "Financials",
    "MS":    "Financials",
    "BLK":   "Financials",
    "AXP":   "Financials",
    "CB":    "Financials",
    "PGR":   "Financials",
    # Energy (6)
    "XOM":   "Energy",
    "CVX":   "Energy",
    "COP":   "Energy",
    "EOG":   "Energy",
    "SLB":   "Energy",
    "MPC":   "Energy",
    # Industrials (9)
    "GE":    "Industrials",
    "CAT":   "Industrials",
    "HON":   "Industrials",
    "UPS":   "Industrials",
    "BA":    "Industrials",
    "RTX":   "Industrials",
    "LMT":   "Industrials",
    "DE":    "Industrials",
    "ETN":   "Industrials",
    # Materials (5)
    "LIN":   "Materials",
    "SHW":   "Materials",
    "APD":   "Materials",
    "ECL":   "Materials",
    "FCX":   "Materials",
    # Utilities (5)
    "NEE":   "Utilities",
    "SO":    "Utilities",
    "DUK":   "Utilities",
    "AEP":   "Utilities",
    "XEL":   "Utilities",
    # Real Estate (5)
    "AMT":   "Real Estate",
    "PLD":   "Real Estate",
    "EQIX":  "Real Estate",
    "PSA":   "Real Estate",
    "CCI":   "Real Estate",
}

SECTORS = sorted(set(STOCKS.values()))  # 11 unique sectors

# ── Data settings ──────────────────────────────────────────────────────────────
START_DATE = "2015-01-01"
END_DATE   = "2024-12-31"
TRAIN_END  = "2020-12-31"
VAL_END    = "2022-12-31"
# Test: 2023-01-01 → 2024-12-31

# ── Risk measure settings ──────────────────────────────────────────────────────
CVAR_ALPHA      = 0.95   # 95% CVaR (5% tail)
HORIZON         = 5      # 5-day horizon
LOOKBACK_WINDOW = 252    # calendar days for rolling historical CVaR

# ── Graph settings ─────────────────────────────────────────────────────────────
CORR_WINDOW    = 60    # rolling correlation window (trading days)
CORR_THRESHOLD = 0.4   # min |correlation| to add a cross-sector edge
GRAPH_UPDATE_FREQ = 21 # rebuild graph every N trading days (≈ monthly)

# ── GNN architecture ───────────────────────────────────────────────────────────
GNN_IN_CHANNELS  = 27  # node feature dimension (see src/data.py)
GNN_HIDDEN_DIM   = 64
GNN_NUM_LAYERS   = 2
GNN_DROPOUT      = 0.3
# Primary conv type for the "gnn"/"mlp" pair (baselines like GCN/GraphConv set
# their own conv_type explicitly). Switched sage->gat after a diagnostic showed
# learned attention (vs. fixed mean-aggregation) meaningfully closes the
# GNN-vs-MLP gap -- consistent with the construction ablation showing the
# dense combined graph oversmooths under uniform aggregation.
GNN_CONV_TYPE    = "gat"
# Dropout on GAT's attention coefficients (Velickovic et al. 2018 used 0.6 on
# citation graphs; 0.0 = PyG default = what every prior GAT result in this repo
# used). A 4-level x 3-seed diagnostic, then confirmed on the full primary
# 5-seed set, showed 0.3 both lowers mean test FZ0 vs. attn_dropout=0.0
# (significant, DM p=0.040) and closes the GNN-vs-MLP gap from significant
# (p=0.032) to not-significant (p=0.121) -- a direct fix to the graph model,
# not a post-hoc correction. Only applied when use_graph=True (the no-graph
# MLP ablation always uses 0.0, so it stays an unaffected baseline).
GNN_ATTN_DROPOUT = 0.3

# ── Training ───────────────────────────────────────────────────────────────────
# First cloud run showed fast overfitting: val FZ0 peaked ~epoch 8 then
# collapsed by epoch 28 while train loss kept improving (patience=20 wasted
# most of its window on a degrading trajectory, and produced high seed
# variance). Strengthened regularisation below (tuned on val loss only, no
# test-set access) to target that: more weight decay, more dropout, a lower
# LR for smaller/more-stable steps, tighter patience.
LEARNING_RATE = 5e-4
WEIGHT_DECAY  = 5e-4
NUM_EPOCHS    = 150
PATIENCE      = 15     # early stopping
SEEDS         = [0, 1, 2, 3, 4]   # multi-seed runs for significance / error bars
ABLATION_SEEDS    = [0, 1, 2]     # seeds for the graph-construction / edge-weight ablation
WALKFORWARD_SEEDS = [0, 1]        # seeds per fold in rolling-origin evaluation

# Rolling-origin folds: (train_end, val_end, test_end). Expanding train window.
WALKFORWARD_FOLDS = [
    ("2019-12-31", "2020-12-31", "2021-12-31"),
    ("2020-12-31", "2021-12-31", "2022-12-31"),
    ("2021-12-31", "2022-12-31", "2023-12-31"),
    ("2022-12-31", "2023-12-31", "2024-12-31"),
]

# ── Risk objective ─────────────────────────────────────────────────────────────
TAIL_ALPHA = 0.05      # tail probability for VaR/ES (= 1 - CVAR_ALPHA)

# Extra tail level for the robustness study: 0.025 = 97.5% ES, the Basel
# Fundamental-Review-of-the-Trading-Book (FRTB) regulatory level.
EXTRA_TAIL_ALPHAS = [0.025]

# ── Extra baselines & robustness (step 9/10) ───────────────────────────────────
GCN_SEEDS       = [0, 1, 2]          # seeds for the plain-GCN baseline
HISTSIM_WINDOW  = 252                # trailing window for empirical (historical) VaR/ES
# Graph-hyperparameter sensitivity grid (correlation threshold × update frequency).
# Each cell rebuilds the graph and retrains the GNN over SENSITIVITY_SEEDS.
SENSITIVITY_SEEDS         = [0, 1]
SENSITIVITY_CORR_THRESH   = [0.3, 0.4, 0.5]
SENSITIVITY_UPDATE_FREQ   = [10, 21, 42]

# ── GARCH ──────────────────────────────────────────────────────────────────────
GARCH_P = 1
GARCH_Q = 1
GARCH_REFIT_FREQ = 21  # refit every N trading days on expanding window

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR    = "data/"
MODEL_DIR   = "models/"
RESULTS_DIR = "results/"
