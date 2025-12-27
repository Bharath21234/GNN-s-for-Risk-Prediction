import yfinance as yf
import pandas as pd
import numpy as np

def fetch_stock_data_with_tail_risk(tickers, start_date, end_date, window=5, alpha=0.05):

    # Fetch historical stock data and compute tail risk (5% VaR) for each ticker.

    # Parameters:
    # tickers (list of str): Stock ticker symbols.
    # start_date (str): Start date in 'YYYY-MM-DD' format.
    # end_date (str): End date in 'YYYY-MM-DD' format.
    # window (int): Lookback window (in days) to check tail risk.
    # alpha (float): Significance level for VaR.

   #  Returns:
   #  dict: Dictionary of DataFrames, each containing historical data with a 'Tail_Risk' column.
    stock_data_dict = {}

    for ticker in tickers:
        # Fetch historical stock data
        df = yf.download(ticker, start=start_date, end=end_date)
        
        if df.empty:
            print(f"No data for {ticker}")
            continue

        # Compute daily returns
        df['Return'] = df['Close'].pct_change()

        #Compute Volality
        df["vol"] = df["Return"].rolling(window).std()

        # Compute rolling VaR at alpha level
        df['VaR'] = df['Return'].rolling(window).quantile(alpha)

        # Check if return in past window days breached VaR
        returns = df['Return'].values
        tail_risk_flag = []

        for i in range(len(df)):
            if i < window:
                tail_risk_flag.append(False)  # Not enough data yet
            else:
                past_returns = returns[i-window:i]  # exclude current day
                var_threshold = np.quantile(past_returns, alpha)
                tail_risk_flag.append(df['Return'].iloc[i] <= var_threshold)


        df['Tail_Risk'] = tail_risk_flag

        stock_data_dict[ticker] = df

    return stock_data_dict


def build_aligned_panel(dict, tickers):
    """
    Build a time-aligned panel DataFrame across multiple tickers.

    Returns
    -------
    panel : pd.DataFrame
        Wide DataFrame with columns like:
        Return_AAPL, vol_AAPL, VaR_AAPL, Tail_AAPL, ...
    valid_dates : pd.DatetimeIndex
        Dates where both t and t+1 exist (used for next-day labels).
    """

    # Find common dates across all tickers
    common_index = None
    for ticker in tickers:
        index = dict[ticker].index
        common_index = index if common_index is None else common_index.intersection(index)

    common_index = common_index.sort_values()

    #  Build wide panel (one row = one date, many ticker features)
    cols = {}
    for tkr in tickers:
        d = dict[tkr].reindex(common_index)
        cols[f"Return_{tkr}"] = d["Return"]
        cols[f"vol_{tkr}"]    = d["vol"]
        cols[f"VaR_{tkr}"]    = d["VaR"]
        cols[f"Tail_Risk_{tkr}"]   = d["Tail_Risk"]

    panel = pd.DataFrame(cols, index=common_index)

    # 3️ Drop dates where rolling stats are not ready
    panel = panel.dropna()

    # 4 We need (t, t+1), so last date cannot be used
    valid_dates = panel.index[:-1]

    return panel, valid_dates


if __name__ == "__main__":
    #Example trial
    tickers = ['AAPL', 'MSFT', 'GOOGL']
    start_date = '2024-01-01'
    end_date = '2024-12-31'
    window = 5
    alpha = 0.05
    stock_data = fetch_stock_data_with_tail_risk(tickers, start_date, end_date, window, alpha)
    df,valid_dates=build_aligned_panel(stock_data,tickers)
    print(df.head(5))
    print(df.columns)

    for ticker, df in stock_data.items():
        print(f"\nData for {ticker}:")
        print(df.tail(100))
