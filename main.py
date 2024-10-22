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
import datetime


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows',None)
np.set_printoptions(threshold=np.inf, linewidth=np.inf)


def main() -> int:

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

def append_top_bottom_columns(bf: pd.DataFrame) -> pd.DataFrame:
    # Bar dataframes directly from the Alpaca API do not have bar tops
    # and bottoms information. Compute them and append it as new columns.
    # Results in a new dataframe, leaving the original unchanged.

    red = bf['close'].values <= bf['open'].values

    # Create a new dataframe with the new columns.
    return bf.assign(red=red, top=np.where(red, bf['open'], bf['close']), bottom=np.where(red, bf['close'], bf['open']))


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

def data(symbol: str) -> Callable:

    return lambda daterange: (
        dates:= pd.date_range(start=pd.to_datetime(daterange[0]), end=pd.to_datetime(daterange[1]), freq='1D').strftime("%Y-%m-%d").tolist(),
        t_dates := (len(dates), lambda  i: dates[i]),
        lambda f_fetch: f_fetch(symbol)(t_dates)
    )[-1]
    

def trades(t_retrieval_method:tuple[Callable,Callable]):

    return lambda symbol: lambda t_dates: f_open_market_only(t_retrieval_method[0](symbol)(t_dates))

def bars(timeframe:TimeFrame) -> Callable:

    return lambda t_retrieval_method: lambda symbol: lambda t_dates: (
        t_df_list := f_open_market_only(t_retrieval_method[1](timeframe)(symbol)(t_dates)),
        results := list(map(lambda i: append_top_bottom_columns(t_df_list[1](i)), range(t_df_list[0]))),
        (len(results), lambda i: results[i])
    )[-1]

async def execute_tasks_concurrently(t_task: tuple[int,Callable]) -> tuple[int,Callable] : 

    tasks = map(t_task[1], range(t_task[0]))
        
    results = await asyncio.gather(*tasks)

    return (
        (len(results), lambda i: results[i][1]),
        (len(results), lambda i: results[i])
    )

def concurrently(client: AsyncRest) -> tuple[Callable,Callable]:

    f_trades = lambda symbol: lambda t_dates: asyncio.run(execute_tasks_concurrently(
        (t_dates[0], lambda i: asyncio.create_task(client.get_trades_async(symbol, t_dates[1](i), t_dates[1](i), limit=0xFFFFFFFF)))
    ))[0]

    f_bars = lambda timeframe: lambda symbol: lambda t_dates: asyncio.run(execute_tasks_concurrently(
        (t_dates[0], lambda i: asyncio.create_task(client.get_bars_async(symbol, t_dates[1](i), t_dates[1](i), timeframe)))
    ))[0]

    return f_trades, f_bars


def synchronously(client: REST) -> tuple[Callable,Callable]:

    f_trades = lambda symbol: lambda t_dates: (
        results:=list(map(lambda i: client.get_trades(symbol, t_dates[1](i), t_dates[1](i), feed='sip').df, range(t_dates[0]))),
        (len(results), lambda i: results[i])
    )[-1]

    f_bars = lambda timeframe: lambda  symbol: lambda t_dates: ( 
        results:=list(map(lambda i: client.get_bars(symbol, timeframe, t_dates[1](i), t_dates[1](i), feed='sip').df, range(t_dates[0]))),
        (len(results), lambda i: results[i])
    )[-1]
    
    return f_trades, f_bars

def market_time(index_0_time: pd.Timestamp) -> tuple[datetime.time, datetime.time]:

    is_dst = index_0_time.tz_convert('America/New_York').dst() != pd.Timedelta(0)
    start_time = pd.Timestamp('13:30') if is_dst else pd.Timestamp('14:30')
    end_time = pd.Timestamp('20:00') if is_dst else pd.Timestamp('21:00')

    return start_time.time(), end_time.time()
    
def f_open_market_only(t_df_list: tuple[int,Callable]) -> tuple[int,Callable]:

    df_list = [
        open_market_df for i in range(t_df_list[0]) 
        if not t_df_list[1](i).empty and not 
        (open_market_df:= t_df_list[1](i).between_time(*market_time(t_df_list[1](i).index[0]))).dropna().empty
    ]

    return len(df_list), lambda i: df_list[i]



def read_write_create_trades(symbol: str) -> Callable:

    return ( 
        lambda t_dates: lambda i: f'ticks/{t_dates[1](i)}-ticks-{symbol}.csv',
        lambda t_retrieval_method: trades(t_retrieval_method)(symbol)
    )

def read_write_create_bars(timeframe: TimeFrame) -> Callable:

    return lambda symbol: (
        lambda t_dates: lambda i: f'bars/{t_dates[1](i)}-{timeframe}-{symbol}.csv',
        lambda t_retrieval_method: bars(timeframe)(t_retrieval_method)(symbol)
    )


async def read_csv_concurrently(path:str):

    return await asyncio.to_thread(read_csv, path)

async def write_csv_concurrently(df: pd.DataFrame, path:str):

    return await asyncio.to_thread(df.to_csv, path, index=True)

def write_csv(df: pd.DataFrame, path: str) -> None:

    return df.to_csv(path, index=True)

def read_csv(path:str):

    try:
        return pd.read_csv(path, parse_dates=['timestamp'], index_col='timestamp')

    except Exception:
        return pd.DataFrame()
    
def merge_df_list(f_lists: tuple[int, int, tuple[int,Callable],tuple[int,Callable]]) -> tuple[int,Callable]:

    if f_lists[0] >= f_lists[2][0]: 
        df_list = list(map(lambda i: f_lists[3][1](i), range(f_lists[1], f_lists[3][0])))
        return len(df_list), lambda i: df_list[i]
    elif f_lists[1] >= f_lists[3][0]: 
        df_list = list(map(lambda i: f_lists[2][1](i), range(f_lists[0], f_lists[2][0])))
        return len(df_list), lambda i: df_list[i]

    if f_lists[2][1](f_lists[0]).index[0] < f_lists[3][1](f_lists[1]).index[0]:
        df_list = [f_lists[2][1](f_lists[0])] + list(map(lambda i: t_merge[1](i), range((t_merge:=merge_df_list((f_lists[0]+1, f_lists[1], f_lists[2], f_lists[3])))[0])))
        return len(df_list), lambda i: df_list[i]
    else:
        df_list = [f_lists[3][1](f_lists[1])] + list(map(lambda i: t_merge[1](i), range((t_merge:=merge_df_list((f_lists[0], f_lists[1]+1, f_lists[2], f_lists[3])))[0])))
        return len(df_list), lambda i: df_list[i]

def read_write_create_optimized(client: AsyncRest) -> tuple[int,Callable]:

    return lambda symbol: lambda t_dates: lambda f_rwc_type: (
        t_rwc:= f_rwc_type(symbol),
        f_path:= t_rwc[0],
        f_path_all_dates:= f_path(t_dates),
        f_fetch:= t_rwc[1](concurrently(client)),
        t_partial_empty_df_list:= asyncio.run(execute_tasks_concurrently((t_dates[0], lambda i: read_csv_concurrently(f_path_all_dates(i)))))[1],
        failed_dates := [t_dates[1](i) for i in range(t_partial_empty_df_list[0]) if t_partial_empty_df_list[1](i).empty],
        t_failed_dates := (len(failed_dates), lambda i : failed_dates[i]),
        t_remaining_df_list:= f_fetch(t_failed_dates),
        f_path_remaining_dates:= f_path((t_remaining_df_list[0], lambda i: t_remaining_df_list[1](i).index[0].date())),
        _:= asyncio.run(execute_tasks_concurrently((t_remaining_df_list[0], lambda i: write_csv_concurrently(t_remaining_df_list[1](i),f_path_remaining_dates(i))))),
        partial_df_list:= [df for i in range(t_partial_empty_df_list[0]) if not (df:=t_partial_empty_df_list[1](i)).empty],
        t_partial_df_list:= (len(partial_df_list), lambda i: partial_df_list[i]),
        merge_df_list((0,0,t_partial_df_list,t_remaining_df_list))
    )[-1]

def read_write_create(client: REST) -> tuple[int,Callable]:

    return lambda symbol: lambda t_dates: lambda f_rwc_type: (
        t_rwc:= f_rwc_type(symbol),
        f_path:= t_rwc[0],
        f_path_all_dates:= f_path(t_dates),
        f_fetch:= t_rwc[1](synchronously(client)),
        partial_empty_df_list:= list(map(lambda i: read_csv(f_path_all_dates(i)), range(t_dates[0]))),
        t_partial_empty_df_list:= (len(partial_empty_df_list), lambda i: partial_empty_df_list[i]),
        failed_dates := [t_dates[1](i) for i in range(t_partial_empty_df_list[0]) if t_partial_empty_df_list[1](i).empty],
        t_failed_dates := (len(failed_dates), lambda i : failed_dates[i]),
        t_remaining_df_list:= f_fetch(t_failed_dates),
        f_path_remaining_dates:= f_path((t_remaining_df_list[0], lambda i: t_remaining_df_list[1](i).index[0].date())),
        _:= list(map(lambda i: write_csv(t_remaining_df_list[1](i), f_path_remaining_dates(i)), range(t_remaining_df_list[0]))),
        partial_df_list:= [df for i in range(t_partial_empty_df_list[0]) if not (df:=t_partial_empty_df_list[1](i)).empty],
        t_partial_df_list:= (len(partial_df_list), lambda i: partial_df_list[i]),
        merge_df_list((0,0,t_partial_df_list,t_remaining_df_list))
    )[-1]

def pca(X: NDArray[np.float64], n_components: np.uint64) -> NDArray[np.float64]:
    # from sklearn.decomposition import PCA
    # pca_=PCA(n_components=n_components)
    # return pca_.fit_transform(X)
    '''
    Perform PCA on the dataset X and reduce it to n_components dimensions.

    Parameters:
    * X, NDArray(n,d) of n samples and d features.
    * n_components, an integer denoting number of principle
    components to retain.

    Output:
    X_pca, NDArray(n, n_components), the transformed data with n samples
    and n_components features.
    '''

    X_meaned = X - np.mean(X, axis=0)

    cov_matrix = np.cov(X_meaned, rowvar=False)

    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    eigenvector_subset = eigenvectors[:, -n_components:]

    X_pca = eigenvector_subset.transpose() @ X_meaned.transpose()

    return X_pca.transpose()

if __name__ == "__main__":
   
    client = Client.initialize(paper=True)
    sync_client = client('trade')
    async_client = client('data')

    print('Getting Trades')

    start_date = '2024-10-11'
    end_date = '2024-10-18'

    spy_october = data('SPY')((start_date, end_date))
    spy_october_csv_optimized = spy_october(read_write_create_optimized(async_client))
  
    t_spy_oct_tf_list = spy_october_csv_optimized(read_write_create_trades)
    t_spy_oct_bf_list = spy_october_csv_optimized(read_write_create_bars(TimeFrame(5,TimeFrameUnit.Minute).value))

    print(t_spy_oct_tf_list[0])
    print(t_spy_oct_bf_list[0])

    args = lambda i: (k_rmsd(2), acceleration(f_trade_frame(t_spy_oct_tf_list[1](i))), first_upper_parabolic_area_enclosed(), second_upper_parabolic_area_enclosed(), first_lower_parabolic_area_enclosed(), second_lower_parabolic_area_enclosed(), cubic_area_enclosed(), first_euclidean_distance(), second_euclidean_distance(), first_slope(), second_slope(), first_percentage_diff(), second_percentage_diff())
    resistance_data_list = list(map(lambda i: resistance_data(i, 3, f_bar_frame(t_spy_oct_bf_list[1](i)), *args(i)), range(t_spy_oct_bf_list[0])))

    combined_resistance_data = np.row_stack(resistance_data_list)

    pca_resistance_data = pca(combined_resistance_data[:,1:], 2)

    print(pca_resistance_data.shape)
    print(pca_resistance_data)
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
