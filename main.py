import pandas as pd
import numpy as np
import client
import acquire
import analyze_refactoring as analyze
from analyze_refactoring import time_based_partition, only_green, all_bars, exclude_green_red_pattern, all_components, pca_components, lower_bound, resistance_line_range, resistance_zone_range, resistance_euclidean_1d, resistance_euclidean_2d, resistance_k_rmsd, resistance_relative_perc_diff, resistance_slopes, resistance_parabolic_concavity, resistance_parabolic_area, k_root, abs, ratio, product, add, subtract
import k_means as km
import plt
from alpaca_trade_api.rest import REST, TimeFrame, TimeFrameUnit
from functools import reduce


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

    # create_bars_and_trades_csv(data_client=data_client, bar_interval=5, bar_unit=TimeFrameUnit.Minute, start_date="2024-05-17", end_date="2024-08-17")
    
    bars_ = pd.read_csv('2024-05-17-to-2024-08-17-5-Min.csv', parse_dates=['timestamp'], index_col='timestamp')

    bars = analyze.append_top_bottom_columns(bars_)
    
    trades = pd.read_csv('2024-05-17-to-2024-08-17-tick.csv', parse_dates=['timestamp'], index_col='timestamp')
    
    bars_per_day, resistance_data_function = analyze.resistance_data_function(bars, trades, pca_components(2))

    resistance_data = resistance_data_function(all_bars, exclude_green_red_pattern, resistance_parabolic_area, resistance_parabolic_concavity, resistance_euclidean_1d, resistance_euclidean_2d, resistance_k_rmsd(lower_bound, resistance_zone_range, 2), resistance_relative_perc_diff, abs(resistance_slopes))

    resistance_k_means_data = resistance_data[:,-2:]

    # resistance_k_means_data = reduce(lambda acc, x: np.row_stack((acc, x[:, -2:])), resistance_data, np.empty((0,2)))
    
    centroids, labels = km.train(resistance_k_means_data, 10, 0, 100)

    learned_resistance_data = resistance_data[labels==6]

    # learned_resistance_data = list(map(lambda data : data[km.inference(data[:,-2:], centroids) == 3], resistance_data))

    # bars_per_day = time_based_partition(bars, '1D')
    resistance_data_per_day = list(map(lambda day: learned_resistance_data[learned_resistance_data[:,0] == day], list(range(len(bars_per_day))) ))

    list(map(lambda bars, data: plt.plot_resistances(bars, data[:,1], data[:, 4:6], data[:, -2:]), bars_per_day, resistance_data_per_day))

    
    
    return 0


if __name__ == "__main__":
    main()