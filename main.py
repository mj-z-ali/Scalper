import pandas as pd
import numpy as np
import client
import acquire
import analyze
import k_means as km
import matplotlib.pyplot as plt
from alpaca_trade_api.rest import REST, TimeFrame, TimeFrameUnit




pd.set_option('display.max_columns', None)

def create_bars_and_trades_csv(data_client: REST, bar_interval: int, bar_unit: TimeFrameUnit, start_date: str, end_date: str) -> None:

    bars_df = acquire.bars(client=data_client, symbol="SPY", timeframe=TimeFrame(bar_interval, bar_unit), start_date=start_date, end_date=end_date)

    bars_df.to_csv(f"{start_date}-to-{end_date}-{bar_interval}-{bar_unit.value}.csv", index=True)

    trades_df = acquire.trades(client=data_client, symbol="SPY", start_date=start_date, end_date=end_date)
    
    trades_df.to_csv(f"{start_date}-to-{end_date}-tick.csv", index=True)

    return None

def main() -> int:

    params = client.init(paper=True)

    data_client = client.establish_client(params=params)

    # create_bars_and_trades_csv(data_client=data_client, bar_interval=5, bar_unit=TimeFrameUnit.Minute, start_date="2024-06-17", end_date="2024-06-28")
 
    bars_ = pd.read_csv('2024-06-17-to-2024-06-28-5-Min.csv', parse_dates=['timestamp'], index_col='timestamp')

    bars = analyze.insert_top_bottom_columns(bars_)

    trades = pd.read_csv('2024-06-17-to-2024-06-28-tick.csv', parse_dates=['timestamp'], index_col='timestamp')
    
    resist_centroids, _ = analyze.resistance_centroids(bars, trades)

    analyze.resistance_plot(bars, trades, centroids=resist_centroids)
    
    return 0


if __name__ == "__main__":
    main()