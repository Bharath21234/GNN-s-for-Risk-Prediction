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




if __name__ == "__main__":
    #Example trial
    tickers = ['AAPL', 'MSFT', 'GOOGL']
    start_date = '2024-01-01'
    end_date = '2024-12-31'
    window = 5
    alpha = 0.05
    stock_data = fetch_stock_data_with_tail_risk(tickers, start_date, end_date, window, alpha)

    for ticker, df in stock_data.items():
        print(f"\nData for {ticker}:")
        print(df.tail(100))
