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

    bars_df.to_csv(f"{start_date}_to_{end_date}_{bar_interval}_{bar_unit.value}.csv", index=True)

    trades_df = acquire.trades(client=data_client, symbol="SPY", start_date=start_date, end_date=end_date)
    
    trades_df.to_csv(f"{start_date}_to_{end_date}_{bar_interval}_tick.csv", index=True)

    return None

def main() -> int:

    params = client.init(paper=True)

    data_client = client.establish_client(params=params)

    # create_bars_and_trades_csv(data_client=data_client, bar_interval=5, bar_unit=TimeFrameUnit.Minute, start_date="2024-06-24", end_date="2024-06-27")

    bar_df = pd.read_csv('2024_06_24_to_2024_06_27_5M.csv', parse_dates=['timestamp'], index_col='timestamp')

    bar_with_top_bottom_df = analyze.insert_top_bottom_columns(bar_df=bar_df)

    trade_df = pd.read_csv('2024_06_24_to_2024_06_27_tick.csv', parse_dates=['timestamp'], index_col='timestamp')
    
    resist_centroids, _ = analyze.resistance_centroids(bar_df=bar_with_top_bottom_df, trade_df=trade_df)

    analyze.resistance_plot(bar_df=bar_with_top_bottom_df, trade_df=trade_df, centroids=resist_centroids)
    
    return 0


if __name__ == "__main__":
    main()