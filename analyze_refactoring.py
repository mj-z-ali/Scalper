import pandas as pd
import numpy as np
from functools import reduce, partial
from typing import Callable
from numpy.typing import NDArray
import k_means as km
import acquire
import plt

def time_based_partition(df: pd.DataFrame, interval: str) -> list[pd.DataFrame]:
    # Partition a single dataframe into a list of dataframes based on time interval.

    partition_generator = map(lambda df_pair: df_pair[1], df.resample(interval))

    non_empty_partition = filter(lambda df_interval: not df_interval.dropna().empty, partition_generator)

    return list(non_empty_partition)

def time_based_column_partition(df: pd.DataFrame, interval: str, col: str) -> list[NDArray[np.any]]:
    '''
    Partition a single dataframe's column into a list of NDArrays based on the given time interval.

    Parameters: a time-series dataframe df, string interval for partioning, and string col which
        is the name of the target column in the dataframe.
    
    Output: a list of NDArrays s.t. each NDArray contains values from the column col within the
        corresponding time interval.
    '''
    col_partition_generator = map(lambda part_df: part_df[col].values, time_based_partition(df, interval))

    return list(col_partition_generator)

def partition_trades(trades: pd.DataFrame, interval: str) -> list[pd.DataFrame]:
    # Partition trades dataframe into a list of dataframes based on time interval.

    return time_based_partition(trades, interval)

def bar_dict(trades: pd.DataFrame) -> dict:
    # Create a bar dictionary based on trades dataframe.

    # If closing price is lower than opening, then red bar is true.
    # Otherwise, false. 
    red_bar = trades['price'].iloc[-1] <= trades['price'].iloc[0]

    # Top of bar. Opening price if a red bar, otherwise, closing price.
    top = trades['price'].iloc[0] if red_bar else trades['price'].iloc[-1]

    # Bottom of bar. The opposite condition of top.
    bottom = trades['price'].iloc[-1] if red_bar else trades['price'].iloc[0] 

    return {
        'timestamp': trades.index[0].floor('S'),
        'open': trades['price'].iloc[0], 
        'high': trades['price'].max(), 
        'low': trades['price'].min(), 
        'close': trades['price'].iloc[-1],
        'top': top,
        'bottom': bottom,
        'red': red_bar
    }

def bar_transformation(trade_partitons: list[pd.DataFrame]) -> pd.DataFrame:
    # Transform trade partions into a dataframe of bars.

    bar_generator = map(bar_dict, trade_partitons)

    return pd.DataFrame(bar_generator).set_index('timestamp')

def new_column(f: Callable[[pd.Series], any],  df: pd.DataFrame) -> pd.Series:
    # Applies given function f on all rows of df.
    # Results in a new Series, leaving the original 
    # dataframe unchanged.

    return df.apply(f, axis=1)

def apply_list(f: Callable[[any], any], data: list) -> list[any]:
    # Applies function f on each element in data list.
    # Basically, a wrapper for list(map()) combination.

    return list(map(f, data))

def append_top_bottom_columns(bars: pd.DataFrame) -> pd.DataFrame:
    # Bar dataframes directly from the Alpaca API do not have bar tops
    # and bottoms information. Compute them and append it as new columns.
    # Results in a new dataframe, leaving the original unchanged.

    red_lambda = lambda bar_row : bar_row['close'] <= bar_row['open']
    
    top_lambda = lambda bar_row : bar_row['open'] if red_lambda(bar_row) else bar_row['close']
    
    bottom_lambda = lambda bar_row : bar_row['close'] if red_lambda(bar_row) else bar_row['open'] 

    # Retrieve red, top, and bottom columns for bar dataframe.
    new_cols =  apply_list(partial(new_column, df=bars), [red_lambda, top_lambda, bottom_lambda])

    # Create a new dataframe with the new columns.
    return bars.assign(red=new_cols[0], top=new_cols[1], bottom=new_cols[2])

def transpose_1d(array: NDArray[np.any]) -> NDArray[np.any]:
    '''
    Transpose a row array into a column array for broadcasting.

    Parameters: (n,) or (1,n) NDArray.

    Output: (n,1) column NDArray.
    '''
    return array.reshape(-1, 1)

def transpose_xd(matrix: NDArray[np.any]) -> NDArray[np.any]:
    '''
    Transpose a matrix onto a new axis for broadcasting.

    Parameters: (n,d) NDArray.

    Output: (n,1,d) column NDArray.
    '''
    return matrix[:, np.newaxis, :]

def euclidean_distances_1d(points_a: NDArray[np.float64], points_b: NDArray[np.float64]) -> NDArray[np.float64]:
    '''
    Euclidean distance of 1-dimensional points.

    Parameters: points_a and points_b of (n,) or (1,n) NDArrays s.t.
        n is the number of points in each.

    Output: n-array of euclidean distances for corresponding point pairs.
    '''
    
    return np.abs(points_a - points_b)

def euclidean_distances_xd(points_a: NDArray[np.float64], points_b: NDArray[np.float64]) -> NDArray[np.float64]:
    '''
    Euclidean distance of d-dimensional points s.t. d > 1.

    Parameters: points_a and points_b of (n,d) NDArrays s.t.
        n is the number of points in each and d is the dimension where d > 1.

    Output: n-array of euclidean distances for corresponding point pairs.
    '''
    
    return np.sqrt(np.sum((points_a - points_b)**2, axis=2))

def distance_matrix_1d(distance_function: Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]], points: NDArray[np.float64]) -> NDArray[np.float64]:
    '''
    Distance of 1d points where distance formula is defined by input function.

    Parameters: distance_function to compute distance formula on points.
        points of (n,) or (1,n) NDarray of n points.

    Output: (n,n) matrix of euclidean distances for all point pairs.
    '''
    points_t = transpose_1d(points)

    return distance_function(points_t, points)

def distance_matrix_xd(distance_function: Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]], points: NDArray[np.float64]) -> NDArray[np.float64]:
    '''
    Distances of d-dimensional points s.t. d > 1 where distance formula is defined by input function.

    Parameters: (n,d)  NDArray of n points with dimension d > 1.

    Output: (n,n) matrix of euclidean distances for all point pairs.
    '''
    points_t = transpose_xd(points)

    return distance_function(points_t, points)

def perc_diff_matrix_1d(array: NDArray[np.float64]) -> NDArray[np.float64]:
    '''
    Relative difference of all element pairs in array.

    Parameters: (n,) or (1, n) NDArray of n points.

    Output: (n,n) matrix of percentage differences for all element pairs.
    '''
    return distance_matrix_1d(lambda p1, p2: 100*(p1 - p2) / p1, array)

def k_root_mean_sqr_dev_1d(array: NDArray[np.float64], k: int, a: float) -> NDArray[np.float64]:
    '''
    Calculates the k-root of the average of the squared differences 
        between each data point in array and the specified value a.

    Parameters: array of (n,) or (1,n) NDArray,
        k for k-root, and specified value a.

    Output: n-array of k-RMSD values from specied value a.
    '''
    sqr_diff = (array - a)**2

    return np.power(sqr_diff.mean(), 1/k)

def apply_upper_matrix_tri(f: Callable[[NDArray[np.any]], NDArray[np.any]], array: NDArray[np.any]) -> NDArray[np.any]:
    '''
    Applies function f on NDArray array but return only the upper triangle of 
        the resulting matrix from f.

    Parameter: Function f to be applied on (n,) NDArray array. Function f must input
        an NDArray and output a (n,n) square matrix. 

    Output: A (c,) NDArray containing the values in the upper triange, excluding the diagonal,
        of the resulting matrix of f(array) s.t. c = (n*(n-1)) / 2
    '''
    n = len(array)

    upper_row, upper_col = np.triu_indices(n, k=1)

    return f(array)[upper_row, upper_col]

def for_each(f: Callable[[any], any], array: list[any]) -> NDArray[np.any]:
    '''
    Apply function f on each element in array.

    Parameters: Function f consisting of one parameter. A list of any element.
        Function f must take elements in array as inputs.

    Output: an NDArray of results from f.
    '''
    array_generator = map(f, array)

    return reduce(lambda acc, x: np.append(acc, x), array_generator, np.array([]))


bars_ = pd.read_csv('2024-06-17-to-2024-06-28-5-Min.csv', parse_dates=['timestamp'], index_col='timestamp')

bars = append_top_bottom_columns(bars_)

bar_top_values =  time_based_column_partition(bars, '1D', 'top')
bar_top_diffs = for_each(partial(apply_upper_matrix_tri, partial(distance_matrix_1d, euclidean_distances_1d)), bar_top_values)

print(bar_top_diffs.shape)
print(bar_top_diffs)
