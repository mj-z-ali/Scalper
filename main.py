import pandas as pd
import numpy as np
from typing import Callable
from numpy.typing import NDArray
import client
import acquire
import k_means as km
import plt
from alpaca_trade_api.rest import REST, TimeFrame, TimeFrameUnit
from functools import reduce


pd.set_option('display.max_columns', None)


def append_top_bottom_columns(bar_df: pd.DataFrame) -> pd.DataFrame:
    # Bar dataframes directly from the Alpaca API do not have bar tops
    # and bottoms information. Compute them and append it as new columns.
    # Results in a new dataframe, leaving the original unchanged.

    red = bar_df['close'].values <= bar_df['open'].values

    # Create a new dataframe with the new columns.
    return bar_df.assign(red=red, top=np.where(red, bar_df['open'], bar_df['close']), bottom=np.where(red, bar_df['close'], bar_df['open']))

def f_create_bar_frame(bar_interval: int, bar_unit: TimeFrameUnit) -> Callable:

    return lambda data_client, symbol, start_date, end_date: append_top_bottom_columns(acquire.bars(data_client, symbol, TimeFrame(bar_interval, bar_unit), start_date, end_date))

def f_create_frame(data_client: REST, symbol: str): 

    return lambda f_df, f_dt: time_based_partition(f_df(data_client, symbol, f_dt('start_date'), f_dt('end_date')), f_dt('interval'))

def f_date(start_date: str, end_date: str, interval: str) -> Callable:

    data = {'start_date': start_date, 'end_date': end_date, 'interval': interval}

    return lambda s: data[s]

def time_based_partition(df: pd.DataFrame, interval: str) -> Callable:
    # Partition a single dataframe into a list of dataframes based on time interval.

    partition_generator = map(lambda df_pair: df_pair[1], df.resample(interval))

    partition = list(filter(lambda df_interval: not df_interval.dropna().empty, partition_generator))

    return lambda f,i: f(partition[i])

def f_bar_frame(f_dt: Callable, f_cf: Callable, bar_interval: int, bar_unit: TimeFrameUnit) -> Callable:

    try:
        f_bf = time_based_partition(pd.read_csv(f"{f_dt('start_date')}-to-{f_dt('end_date')}-{bar_interval}-{bar_unit.value}.csv", parse_dates=['timestamp'], index_col='timestamp'), f_dt(' interval'))
        
        return f_bf
    
    except Exception as e:

        f_bf = f_cf(f_create_bar_frame(bar_interval, bar_unit), f_dt)

        f_bf.to_csv(f"{f_dt('start_date')}-to-{f_dt('end_date')}-{bar_interval}-{bar_unit.value}.csv", index=True)

        return f_bf

def f_trade_frame(f_dt: Callable, f_cf: Callable) -> Callable:

    try:
        f_tf = time_based_partition(pd.read_csv(f"{f_dt('start_date')}-to-{f_dt('end_date')}-tick.csv", parse_dates=['timestamp'], index_col='timestamp'), f_dt(' interval'))

        return f_tf
    
    except Exception as e:

        f_tf = f_cf(acquire.trades, f_dt)

        f_tf.to_csv(f"{f_dt('start_date')}-to-{f_dt('end_date')}-tick.csv")

        return f_tf
    

def time_based_partition(df: pd.DataFrame, interval: str) -> Callable:
    # Partition a single dataframe into a list of dataframes based on time interval.

    partition_generator = map(lambda df_pair: df_pair[1], df.resample(interval))

    partition = list(filter(lambda df_interval: not df_interval.dropna().empty, partition_generator))

    return lambda f,i: f(partition[i])


def main() -> int:

    data_client = client.establish_client(client.init(paper=True))

    f_dt = f_date('2024-05-17', '2024-08-17', '1D')

    f_cf = f_create_frame(data_client, 'SPY')

    f_bf = f_bar_frame(f_dt, f_cf, 5, TimeFrameUnit.Minute)

    f_tf = f_trade_frame(f_dt, f_cf)

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