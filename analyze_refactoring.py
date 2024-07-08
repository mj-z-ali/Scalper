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

    Parameters: 
    a time-series dataframe df, string interval for partioning, and string col which
    is the name of the target column in the dataframe.
    
    Output: 
    a list of NDArrays s.t. each NDArray contains values from the column col within the
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

    Parameters: 
    points_a and points_b are (n,) or (1, n) NDArray of n points.
    If either points_a or points_b are transposed as (n,1), then 
    the result is euclidean distances of points(i) to points.T(j) 
    for all i,j, instead.

    Output: 
    either n-array or (n,n) matrix of euclidean distances depending 
    on one of the parameters being transposed.
    '''
    
    return np.abs(points_a - points_b)

def euclidean_distances_xd(points_a: NDArray[np.float64], points_b: NDArray[np.float64]) -> NDArray[np.float64]:
    '''
    Euclidean distance of points_a(i) and points_b(j) for all i,j
    of d-dimensional points s.t. d > 1.
    

    Parameters: 
    points_a of (n,d) NDArrays and points_b of (n,1,d) s.t.
    n is the number of points in each and d is the dimension where d > 1.
    If either one of the parameters are reshaped as (n,1,d), then result 
    is a 

    Output: 
    (n,n) matrix of euclidean distances.
    '''
    
    return np.sqrt(np.sum((points_a - points_b)**2, axis=2))

def matrix_operation_1d(f: Callable[[NDArray[np.any], NDArray[np.any]], NDArray[np.any]], array: NDArray[np.any]) -> NDArray[np.any]:
    '''
    Apply some function f on NDArray of (n,) or (1,n).

    Parameters: 
    funtion f with two NDArray parameters that should output
    a matrix when one parameter is transposed.
    array of (n,) or (1,n) NDarray.

    Output: f matrix output of (n,n).
    '''
    array_t = transpose_1d(array)

    return f(array_t, array)

def matrix_operation_xd(f: Callable[[NDArray[np.any], NDArray[np.any]], NDArray[np.any]], array: NDArray[np.any]) -> NDArray[np.any]:
    '''
    Apply function f on NDArray of (n,d) s.t. d > 1.

    Parameters: 
    function f with two NDArray parameters that should output
    a matrix when one parameter is reshaped to (n,1,d)
    (n,d)  NDArray of n elements with dimension d > 1.

    Output: f matrix output of (n,n).
    '''
    array_t = transpose_xd(array)

    return f(array_t, array)

def relative_perc_1d(points_a: NDArray[np.float64], points_b: NDArray[np.float64]) -> NDArray[np.float64]:
    '''
    Relative differences of points_b(i) to points_a(i) for all i.

    Parameters: 
    points_a and points_b are (n,) or (1, n) NDArray of n points.
    If either points_a or points_b are transposed as (n,1), then 
    the result is relative differences of points.T(j) to points(i) 
    for all i,j, instead.

    Output: 
    either n-array or (n,n) matrix depending on one of the parameters 
    being transposed.
    '''
    return 100*(points_a - points_b) / points_a

def slopes_2d(points_a: NDArray[np.float64], points_b: NDArray[np.float64]) -> NDArray[np.float64]:
    '''
    (y2-y1) / (x2-x1) s.t. x1,y1 are points in points_a and x2,y2 are in points_b array.

    Parameters:
    (n,2) and (n,1,2) for points_a and points_b, respectively.

    Output:
    (n,n) slope matrix for all point pairs. Divide-by-zero is an undefined slope
    and should be ignored in resulting matrix. For instance, in the case where
    points_a and points_b are the same set of points, the diagonal should be 
    ignored.
    '''
    diff = points_a - points_b

    x_pts, y_pts = diff[:,:,0], diff[:,:,1]

    return np.divide(y_pts, x_pts, where=x_pts!=0)

def k_root_mean_sqr_dev_1d(array: NDArray[np.float64], k: int, a: float) -> NDArray[np.float64]:
    '''
    Calculates the k-root of the average of the squared differences 
    between each data point in array and the specified value a.

    Parameters: 
    array of (n,) or (1,n) NDArray, k for k-root, and specified value a.

    Output: n-array of k-RMSD values from specied value a.
    '''
    sqr_diff = (array - a)**2

    return np.power(sqr_diff.mean(), 1/k)

def apply_upper_matrix_tri(f: Callable[[NDArray[np.any]], NDArray[np.any]], array: NDArray[np.any]) -> NDArray[np.any]:
    '''
    Applies function f on NDArray array but return only the upper triangle of 
    the resulting matrix from f.

    Parameter: 
    Function f to be applied on (n,) NDArray array. Function f must input
    an NDArray and output a (n,n) square matrix. 

    Output: 
    A (c,) NDArray containing the values in the upper triange, excluding the diagonal,
    of the resulting matrix of f(array) s.t. c = (n*(n-1)) / 2
    '''
    n = len(array)

    upper_row, upper_col = np.triu_indices(n, k=1)

    return f(array)[upper_row, upper_col]

def for_each(f: Callable[[any], any], array: list[any]) -> NDArray[np.any]:
    '''
    Apply function f on each element in array.

    Parameters: 
    Function f consisting of one parameter. A list of any element.
    Function f must take elements in array as inputs.

    Output: an NDArray of results from f on each element in array.
    '''
    array_generator = map(f, array)
    return np.array(list(array_generator))
    return reduce(lambda acc, x: np.append(acc, x), array_generator, np.array([]))

def flatten(array: NDArray[np.any]) -> NDArray[np.any]:
    '''
    Flatten multi-dimensional array to 1 dimensional (n,).

    Output: (n,) NDArray s.t. n is the number of elements in array.
    '''
    return reduce(lambda acc, x: np.append(acc, x), array, np.array([]))


def compose_functions(functions: list[Callable[[Callable, np.any], np.any]]) -> Callable[[np.any], np.any]:
    '''
    Transforms a sequence of functions into a single function with one parameter that would
    have otherwise been a nested chain of function calls with the pattern f(g(h(...)), a)
    s.t.  f,g, and h are functions. In other words, something of the form f(g(h(...)), a) 
    now becomes lambda a : f(g(h(...)), a)

    Parameters: 
    list of functions each with two parameters; the first being another function of the same
    pattern and the second parameter being any value as long as it is acceptable to the 
    passing funcion. A slight caveat; The final function in the  list need not to follow 
    the same structure. However, it will be called in the second-to-last function and it 
    must adhere to how it is used there. List must have a length of ≥ 2

    Output: a function of the form lambda x : f(g(h(...)), x).

    Example Usage: The following chain of functions:

    >> for_each(partial(apply_upper_matrix_tri, partial(distance_matrix_1d, euclidean_distances_1d)), bar_top_values)

    Now becomes:

    >> dist_funct = compose_functions([for_each, apply_upper_matrix_tri, distance_matrix_1d, euclidean_distances_1d])

    >> dist_funct(bar_top_values)

    '''
    if len(functions) == 2:
        return lambda array : functions[0](functions[1], array)
    else:
        return lambda array : functions[0](compose_functions(functions[1:]), array)

bars_ = pd.read_csv('2024-06-17-to-2024-06-28-5-Min.csv', parse_dates=['timestamp'], index_col='timestamp')

bars = append_top_bottom_columns(bars_)

bar_top_values =  time_based_column_partition(bars, '1D', 'top')

euclid_dist_1d = compose_functions([for_each, apply_upper_matrix_tri, matrix_operation_1d, euclidean_distances_1d])

print(f"euclidean dist {euclid_dist_1d(bar_top_values)}")
print(euclid_dist_1d(bar_top_values).shape)

perc_diff_1d = compose_functions([for_each, apply_upper_matrix_tri, matrix_operation_1d, relative_perc_1d])

perc_diff_top_vals = perc_diff_1d(bar_top_values)
print(f"perc diff {perc_diff_top_vals}")
print(perc_diff_top_vals.shape)

indices_mat = euclid_dist_1d(for_each(lambda t: np.arange(t.shape[0]),bar_top_values))

perc_slope = perc_diff_top_vals / indices_mat


print(indices_mat)

top_diffs_2d = for_each(lambda t: np.column_stack((t,np.arange(t.shape[0]))),bar_top_values)

euclid_dist_2d = compose_functions([for_each, apply_upper_matrix_tri, matrix_operation_xd, euclidean_distances_xd])

print(f"euclid dist 2d {euclid_dist_2d(top_diffs_2d)}")
print(euclid_dist_2d(top_diffs_2d).shape)

slopes = compose_functions([for_each, apply_upper_matrix_tri, matrix_operation_xd, slopes_2d])

print(f"slopes {slopes(top_diffs_2d)}")
print(slopes(top_diffs_2d).shape)


def diff_matrix(matrix: np.array) -> np.array:

    n = len(matrix)

    matrix_reshaped = matrix.reshape((n, 1))

    diff_matrix = 100*(matrix_reshaped - matrix_reshaped.T) / matrix_reshaped

    return diff_matrix

def top_slopes_day(bars_for_day: pd.DataFrame) -> np.array:

    n = len(bars_for_day['top'].values)

    # Upper triangle indices excluding diagonal since it and the lower part 
    # is relationship of bar to itself and past bars.
    upper_row, upper_col = np.triu_indices(n=n, k=1)

    # Percentage differences of bar top to all others.
    diff_mat = diff_matrix(matrix=bars_for_day['top'].values)[upper_row, upper_col]

    indices = np.arange(n).reshape(1,n)

    # Line lengths in units of bars. Bar top to all other bar top distances.
    lengths = (indices - indices.T)[upper_row, upper_col]

    slopes = diff_mat / lengths

    return slopes

top_slopes = for_each(top_slopes_day, time_based_partition(bars,"1D"))

print(f"top slopes {top_slopes}")
print(f"top slopes {top_slopes.shape}")

print(np.all(perc_slope==top_slopes))

