import pandas as pd
import numpy as np
from typing import Callable
from numpy.typing import NDArray
import client as Client
import acquire
from analyze import resistance_data, k_rmsd, acceleration, first_upper_parabolic_area_enclosed, second_upper_parabolic_area_enclosed, first_lower_parabolic_area_enclosed, second_lower_parabolic_area_enclosed, cubic_area_enclosed, first_euclidean_distance, second_euclidean_distance, first_slope, second_slope, first_percentage_diff, second_percentage_diff
import k_means as km
import plt
from alpaca_trade_api.rest import REST, TimeFrame, TimeFrameUnit
from alpaca_trade_api.rest_async import AsyncRest
from functools import reduce
import asyncio




pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows',None)
np.set_printoptions(threshold=np.inf, linewidth=np.inf)

def append_top_bottom_columns(bf: pd.DataFrame) -> pd.DataFrame:
    # Bar dataframes directly from the Alpaca API do not have bar tops
    # and bottoms information. Compute them and append it as new columns.
    # Results in a new dataframe, leaving the original unchanged.

    red = bf['close'].values <= bf['open'].values

    # Create a new dataframe with the new columns.
    return bf.assign(red=red, top=np.where(red, bf['open'], bf['close']), bottom=np.where(red, bf['close'], bf['open']))

def f_create_bar_frame(bar_interval: int, bar_unit: TimeFrameUnit) -> Callable:

    return lambda data_client, symbol, start_date, end_date: append_top_bottom_columns(acquire.bars(data_client, symbol, TimeFrame(bar_interval, bar_unit), start_date, end_date))

def f_create_frame(data_client: REST, symbol: str): 

    return lambda f_df, f_tm: f_df(data_client, symbol, f_tm('start_date'), f_tm('end_date'))

def f_time(start_date: str, end_date: str) -> Callable:

    data = {'start_date': start_date, 'end_date': end_date}

    return lambda s: data[s]

def time_based_partition(df: pd.DataFrame, interval: str) -> tuple[Callable, int]:
    # Partition a single dataframe into a list of dataframes based on time interval.

    partition_generator = map(lambda df_pair: df_pair[1], df.resample(interval))

    partition = list(filter(lambda df_interval: not df_interval.dropna().empty, partition_generator))

    return lambda i: partition[i], len(partition)

def f_trade_frame_list(f_hd: Callable) -> Callable:

    f_tf_l, _ = time_based_partition(f_hd(trade_frame), '1D')

    return lambda i: f_trade_frame(f_tf_l(i))

def f_bar_frame_list(f_hd: Callable, b_intv: int, b_u: TimeFrameUnit) -> tuple[Callable, int]:

    f_bf_l, n_days = time_based_partition(f_hd(lambda f_tm, f_cf: bar_frame(f_tm, f_cf, b_intv, b_u)), '1D')

    return lambda i: f_bar_frame(f_bf_l(i)), n_days

def f_multi_day_resistance_data(f_hd: Callable, min_line_width: int, bar_interval: int, bar_unit: TimeFrameUnit) -> Callable:

    f_bf_l, n_days = f_bar_frame_list(f_hd, bar_interval, bar_unit)

    f_res = lambda i, f_args: resistance_data(i, min_line_width, f_bf_l(i), *f_args(i))

    return lambda f_args: reduce(lambda acc, x: np.concatenate((acc, f_res(x, f_args))), range(1, n_days), f_res(0, f_args))


def f_bar_frame(df: pd.DataFrame) -> Callable:

    data = {
        'time': df.index.values,
        'range_x': np.arange(len(df)-1),
        'top_p': df['top'].values[:-1],
        'bottom_p': df['bottom'].values[:-1],
        'high_p': df['high'].values[:-1]
        
    }

    return lambda s: data[s]

def f_trade_frame(tf: Callable) -> Callable:

    data = {
        'time': tf.index.values,
        'price': tf['price'].values,
        'size': tf['size'].values
    }

    return lambda s: data[s]

def bar_frame(f_tm: Callable, f_cf: Callable, bar_interval: int, bar_unit: TimeFrameUnit) -> Callable:

    try:
        bf = pd.read_csv(f"{f_tm('start_date')}-to-{f_tm('end_date')}-{bar_interval}-{bar_unit.value}.csv", parse_dates=['timestamp'], index_col='timestamp')
        
        return bf
    
    except Exception as e:

        bf = f_cf(f_create_bar_frame(bar_interval, bar_unit), f_tm)

        bf.to_csv(f"{f_tm('start_date')}-to-{f_tm('end_date')}-{bar_interval}-{bar_unit.value}.csv", index=True)

        return bf

def trade_frame(f_tm: Callable, f_cf: Callable) -> Callable:

    try:
        tf = pd.read_csv(f"{f_tm('start_date')}-to-{f_tm('end_date')}-tick.csv", parse_dates=['timestamp'], index_col='timestamp')

        return tf
    
    except Exception as e:

        tf = f_cf(acquire.trades, f_tm)

        tf.to_csv(f"{f_tm('start_date')}-to-{f_tm('end_date')}-tick.csv")

        return tf
    
def historical_data(symbol: str, start_date: str, end_date: str):
    
    data_client = client.establish_client(client.init(paper=True))

    f_tm = f_time(start_date, end_date)

    f_cf = f_create_frame(data_client, symbol)

    return lambda f_frame: f_frame(f_tm, f_cf)

def main() -> int:

    # f_hd = historical_data('SPY', '2024-05-17', '2024-08-17')
    f_hd = historical_data('SPY', '2024-10-05', '2024-10-11')


    f_data = f_multi_day_resistance_data(f_hd, 3, 5, TimeFrameUnit.Minute)

    f_tf_l = f_trade_frame_list(f_hd)

    # len?
    f_args = lambda i: (k_rmsd(2), acceleration(f_tf_l(i)), first_upper_parabolic_area_enclosed(), second_upper_parabolic_area_enclosed(), first_lower_parabolic_area_enclosed(), second_lower_parabolic_area_enclosed(), cubic_area_enclosed(), first_euclidean_distance(), second_euclidean_distance(), first_slope(), second_slope(), first_percentage_diff(), second_percentage_diff())

    res_data = f_data(f_args)

    print(res_data.shape)
    print(res_data)

    # create_bars_and_trades_csv(data_client=data_client, bar_interval=5, bar_unit=TimeFrameUnit.Minute, start_date="2024-05-17", end_date="2024-08-17")
    
    # bars_ = pd.read_csv('2024-05-17-to-2024-08-17-5-Min.csv', parse_dates=['timestamp'], index_col='timestamp')

    # bars = analyze.append_top_bottom_columns(bars_)
    
    # trades = pd.read_csv('2024-05-17-to-2024-08-17-tick.csv', parse_dates=['timestamp'], index_col='timestamp')
    
    # bars_per_day, resistance_data_function = analyze.resistance_data_function(bars, trades, pca_components(2))

    # resistance_data = resistance_data_function(all_bars, exclude_green_red_pattern, resistance_parabolic_area, resistance_parabolic_concavity, resistance_euclidean_1d, resistance_euclidean_2d, resistance_k_rmsd(lower_bound, resistance_zone_range, 2), resistance_relative_perc_diff, abs(resistance_slopes))

    # resistance_k_means_data = resistance_data[:,-2:]

    # resistance_k_means_data = reduce(lambda acc, x: np.row_stack((acc, x[:, -2:])), resistance_data, np.empty((0,2)))
    
    # centroids, labels = km.train(resistance_k_means_data, 10, 0, 100)

    # learned_resistance_data = resistance_data[labels==6]

    # learned_resistance_data = list(map(lambda data : data[km.inference(data[:,-2:], centroids) == 3], resistance_data))

    # bars_per_day = time_based_partition(bars, '1D')
    # resistance_data_per_day = list(map(lambda day: learned_resistance_data[learned_resistance_data[:,0] == day], list(range(len(bars_per_day))) ))

    # list(map(lambda bars, data: plt.plot_resistances(bars, data[:,1], data[:, 4:6], data[:, -2:]), bars_per_day, resistance_data_per_day))


    return 0

# def list_dates(date_range: tuple[str,str]) -> list[str]:

#     dates = pd.date_range(start=pd.to_datetime(date_range[0]), end=pd.to_datetime(date_range[1]), freq='1D').strftime("%Y-%m-%d").tolist()
    
#     return dates


# def data(symbol: str) -> Callable:

#     g = lambda data_list: (len(data_list),lambda i: data_list[i])

#     f = lambda date_list: lambda f_fetch: g(f_fetch(symbol)(date_list))

#     return lambda t_daterange: f(list_dates(t_daterange))


def data(symbol: str) -> Callable:

    return lambda daterange: (
        dates:= pd.date_range(start=pd.to_datetime(daterange[0]), end=pd.to_datetime(daterange[1]), freq='1D').strftime("%Y-%m-%d").tolist(),
        lambda f_fetch: (len(data_list:= f_fetch(symbol)(dates)), lambda i: data_list[i])
    )[1]
    
f = lambda data_list: (len(data_list),lambda i: data_list[i])
lambda f_fetch: f(f_fetch(symbol)(dates))
def filter_open_market(df: pd.DataFrame) -> pd.DataFrame:

    is_dst = df.index[0].tz_convert('America/New_York').dst() != pd.Timedelta(0)
    start_time = pd.Timestamp('13:30') if is_dst else pd.Timestamp('14:30')
    end_time = pd.Timestamp('20:00') if is_dst else pd.Timestamp('21:00')

    return df.between_time(start_time.time(), end_time.time())

def bars_concurrently(data_client: AsyncRest) -> Callable:

    open_market_only = lambda results: [f_bar_frame(open_market_bf) for _, bf in results if not bf.empty and not (open_market_bf := filter_open_market(bf)).dropna().empty]

    async def fetch_bars(tasks:map) : return open_market_only(await asyncio.gather(*tasks))

    return lambda symbol: lambda date_list: asyncio.run(fetch_bars(map(lambda date: asyncio.create_task(data_client.get_bars_async(symbol, date, date)))))

def trades_concurrently(data_client: AsyncRest) -> Callable:
    
    open_market_only = lambda results: [f_trade_frame(open_market_tf) for _, tf in results if not tf.empty and not (open_market_tf := filter_open_market(tf)).dropna().empty]

    async def fetch_trades(tasks:map) : return open_market_only(await asyncio.gather(*tasks))

    return lambda symbol: lambda date_list: asyncio.run(fetch_trades(map(lambda date: asyncio.create_task(data_client.get_trades_async(symbol, date, date, limit=0xFFFFFFFF)), date_list)))
        
def trades(data_client: REST) -> Callable:

    open_market_only = lambda results: [f_trade_frame(open_market_tf) for tf in results if not tf.empty and not (open_market_tf := filter_open_market(tf)).dropna().empty]

    return lambda symbol: lambda date_list: open_market_only(list(map(lambda date: data_client.get_trades(symbol, date, date).df, date_list)))




if __name__ == "__main__":
    # main()
    # data_client = client.establish_client(client.init(paper=True))
    client = Client.initialize(paper=True)
    trade_client = client('trade')
    data_client = client('data')

    print('Getting Trades')

    # Fetch trades concurrently
    # tf = asyncio.run(fetch_trades_concurrently(async_rest_data_client))
    spy_october = data('SPY')(('2024-10-11', '2024-10-11'))
    spy_october_trades = spy_october(trades_concurrently(data_client))
    # sym, tf  = asyncio.run(async_rest_data_client.get_trades_async('SPY', '2024-10-11', '2024-10-11',limit=0xFFFFFFFF))
    # acquire.trades(data_client, "SPY", '2024-10-11', '2024-10-11')
    # tf = rest_data_client.get_trades('SPY','2024-10-09', '2024-10-11').df
    print(len(spy_october_trades[1](0)('time')))
    print(spy_october_trades[1](0)('time')[:5])




'''
Observation:

sym, tf  = asyncio.run(async_rest_data_client.get_trades_async('SPY', '2024-10-09', '2024-10-11', limit=0xFFFFFFFF))

real    0m57.450s
user    0m14.748s
sys     0m1.998s



tf = asyncio.run(fetch_trades_concurrently(async_rest_data_client))

real    0m28.336s
user    0m13.975s
sys     0m1.096s


tf = rest_data_client.get_trades('SPY','2024-10-09', '2024-10-11').df

real    3m18.544s
user    0m13.238s
sys     0m1.389s

'''
