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

def transpose_array(array: NDArray[np.any]) -> NDArray[np.any]:
    '''
    Transpose an NDArray for broadcasting.

    Parameters: (n,) or (1,n) NDArray.

    Output: (n,1) NDArray.
    '''
    return array.reshape(-1, 1)

def transpose_matrix(matrix: NDArray[np.any]) -> NDArray[np.any]:
    '''
    Transpose a matrix onto a new axis for broadcasting.

    Parameters: (n,d) NDArray.

    Output: (n,1,d) NDArray.
    '''
    return matrix[:, np.newaxis, :]

def matrix_operation_1d(f: Callable[[NDArray[np.any], NDArray[np.any]], NDArray[np.any]], array: NDArray[np.any]) -> NDArray[np.any]:
    '''
    Apply some matrix function f on NDArray of (n,) or (1,n).

    Parameters: 
    funtion f of type f(arr_0, arr_1) -> NDArray(n,n) s.t.
    arr_0 is of type NDArray(n,) or NDArray(1,n)  and arr_1
    is of type NDArray(n,1).
    NDarray array of type (n,) or (1,n).

    Output: f matrix output of NDArray(n,n).
    '''
    array_t = transpose_array(array)

    return f(array, array_t)

def matrix_operation_xd(f: Callable[[NDArray[np.any], NDArray[np.any]], NDArray[np.any]], matrix: NDArray[np.any]) -> NDArray[np.any]:
    '''
    Apply some matrix function f on NDArray of (n,d) s.t. d > 1.

    Parameters: 
    function f of type f(NDArray(n,d), NDArray(n,1,d)) -> NDArray(n,n) s.t.
    d > 1. 
    NDArray array of type NDArray(n,d) s.t. d > 1.

    Output: f matrix output of NDArray(n,n).
    '''
    matrix_t = transpose_matrix(matrix)

    return f(matrix, matrix_t)

def euclidean_distances_1d(points_a: NDArray[np.float64], points_b: NDArray[np.float64]) -> NDArray[np.float64]:
    '''
    Euclidean distance of 1-dimensional points_a(i) to points_b(i) for 
    all i. If points_a and points_b are of different shape, then the 
    function produces Euclidean distances for all point pairs.

    Parameters: 
    points_a and points_b of n and m 1-dimensional points, respectively.
    If points_a and points_b are of different shape, then n=m.

    Output: 
    either n-array or (n,n) matrix of euclidean distances depending 
    on points_a and points_b being of same shape or not.
    '''
    
    return np.abs(points_a - points_b)

def euclidean_distances_xd(points_a: NDArray[np.float64], points_b: NDArray[np.float64]) -> NDArray[np.float64]:
    '''
    Euclidean distances of points_a(i) to points_b(i) for all i
    of d-dimensional points s.t. d > 1. If points_a and points_b
    are of different shapes, ie. (n,d) and (m,1,d), then the function 
    produces the euclidean distances of points_a(i) to points_b(j) for 
    all i,j (all possible pair of points).
    

    Parameters: 
    points_a and points_b NDArrays of n and m d-dimensional points, respectively,
    s.t. d > 1. If points_a and points_b are the same shape, n=m.

    Output: 
    Either n-array or (n,m) matrix of euclidean distances depending on whether
    points_a and points_b are of same shape or not.
    '''
    
    return np.sqrt(np.sum((points_a - points_b)**2, axis=-1))

def diff_1d(points_a: NDArray[np.float64], points_b: NDArray[np.float64]) -> NDArray[np.float64]:
    '''
    Differences of 1-dimensional points_a(i) to points_b(i) for 
    all i. If points_a and points_b are of different shape, then the 
    function produces differences for all point pairs.

    Parameters: 
    points_a and points_b of n and m 1-dimensional points, respectively.
    If points_a and points_b are of different shape, then n=m.

    Output: 
    either n-array or (n,n) matrix of euclidean distances depending 
    on points_a and points_b being of same shape or not.
    '''
    
    return points_a - points_b

def relative_perc_1d(points_a: NDArray[np.float64], points_b: NDArray[np.float64]) -> NDArray[np.float64]:
    '''
    Relative differences of points_b(i) to points_a(i) for all i.
    If points_a and points_b are different shape, ie. (n,) and (n,1),
    then relative differences of points_b(j) to points_a(i) for all i,j.

    Parameters: 
    points_a and points_b NDArrays of n and m points, respectively.
    If different shape, then n=m.

    Output: 
    either n-array or (n,m) matrix depending points_a and points_b
    being same shape or not.
    '''
    return 100*(points_a - points_b) / points_a

def slopes_2d(points_a: NDArray[np.float64], points_b: NDArray[np.float64]) -> NDArray[np.float64]:
    '''
    (y2-y1) / (x2-x1) s.t. x1,y1 are points in points_a and x2,y2 are in points_b array,
    for points_a(i) and points_b(i) for all i. If parameters are of different shape, 
    ie. (n,2) and (n,1,2), then produce slopes for all possible point pairs.

    Parameters:
    n and m 2-dimensional points for points_a and points_b, respectively.
    If parameters are same shape, then n=m.

    Output:
    n-array or (n,m) slope matrix depending on whether parameters are same shape 
    or not. Divide-by-zero is an undefined slope and should be ignored in resulting
    matrix. For instance, in the case where points_a and points_b are the same set 
    of points, the diagonal should be ignored.
    '''
    diff = points_a - points_b

    x_pts, y_pts = diff[...,0], diff[...,1]

    return np.divide(y_pts, x_pts, where=x_pts!=0)

def k_root_mean_sqr_dev_1d(array: NDArray[np.float64], k: int, a: float) -> float:
    '''
    Calculates the k-root of the average of the squared differences 
    between each data point in array and the specified value a.

    Parameters: 
    array of (n,) or (1,n) NDArray, k for k-root, and specified value a.

    Output: float of k-RMSD value from specied value a.
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

def for_each(f: Callable[..., any], *array: list[any]) -> list[any]:
    '''
    Apply function f on each element in array.

    Parameters: 
    * Function f consisting of same number of parameters as input arrays. 
    Function f must take elements in array as inputs.
    * Lists of any elements but must be same size.

    Output: an NDArray of results from f on each element in array.
    '''
    array_generator = map(f, *array)

    return list(array_generator)

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
    
def upper_matrix_tri_mask(shape: tuple[int,...]) -> NDArray[np.bool_]:
    '''
    Mask of upper triangle excluding the diagonal of a square matrix.

    Parameters: a Tuple(n,n) denoting the shape of the matrix.

    Output: Boolean (n,n) matrix where only upper triangle is set to True.
    '''
    return np.triu(np.ones(shape, dtype=bool), k=1)

def lower_matrix_tri_mask(shape: tuple[int,...]) -> NDArray[np.bool_]:
    '''
    Mask of lower triangle excluding the diagonal of a square matrix.

    Parameters: a Tuple(n,n) denoting the shape of the matrix.

    Output: Boolean (n,n) matrix where only lower triangle is set to True.
    '''
    return np.tril(np.ones(shape, dtype=bool), k=-1)

def diagonal_matrix_mask(n: int) -> NDArray[np.bool_]:
    
    return np.eye(n, dtype=bool)

def row_indices(matrix: NDArray[np.any]) -> NDArray[np.int64]:
    '''
    Row indices of a 2D matrix.

    Parameters: NDArray(n,m)

    Output: NDArray(n,)
    '''

    return np.arange(matrix.shape[1])

def column_indices(matrix: NDArray[np.any]) -> NDArray[np.int64]:
    '''
    Column indices of a 2D matrix.

    Parameters: NDArray(n,m)

    Output: NDArray(m,)
    '''

    return np.arange(matrix.shape[1])

def first_occurence_indices(mask_matrix: NDArray[np.bool_]) -> NDArray[np.int64]:
    '''
    Indices of the first Truth value in each row of a 2D boolean matrix.

    Parameters: NDArray(n,m) boolean matrix.

    Outputs: NDArray(n,). If no Truth value is found in a row, -1 is the result
    rather than the index.
    '''

    first_true_indices = np.argmax(mask_matrix, axis=1)

    return np.where((first_true_indices == 0) & ~mask_matrix[:, 0], -1, first_true_indices)

def last_occurence_indices(mask_matrix: NDArray[np.bool_]) -> NDArray[np.int64]:
    '''
    Indices of the last Truth value in each row of a 2D boolean matrix.

    Parameters: NDArray(n,m) boolean matrix.

    Outputs: NDArray(n,). If no Truth value is found in a row, m is the result
    rather than the index.
    '''
    last_true_indices = mask_matrix.shape[1] - np.argmax(mask_matrix[:,::-1], axis=1) - 1

    return np.where((last_true_indices == mask_matrix.shape[1] - 1) & ~mask_matrix[:, -1], mask_matrix.shape[1], last_true_indices)

def occurence_mask(mask_matrix: NDArray[np.bool_], occurence_indices: Callable[[NDArray[np.bool_]], NDArray[np.int64]]) -> NDArray[np.bool_]:
    '''
    Set all boolean values to False except the first or last Truth value,
    depending on the occurence_indices function passed, in each row of a 
    boolean matrix.

    Parameters: 
    * a boolean matrix of NDArray(n,m)
    * a function with a mask_matrix paramter that outputs the first or last True
    value indices.

    Output: a (n,m) boolean matrix where only the first or last True values in each
    row is left alone and all else is set to False.
    '''
    return column_indices(mask_matrix) == occurence_indices(mask_matrix)[:, None]

def combined_false_region_endpoints(start_matrix_mask: NDArray[np.bool_], end_matrix_mask: NDArray[np.bool_]) -> NDArray[np.int64]:

    start_indices = last_occurence_indices(start_matrix_mask)

    end_indices = first_occurence_indices(end_matrix_mask)

    endpoints = np.column_stack((start_indices, end_indices))

    return endpoints

def between_endpoints_mask(endpoints: NDArray[np.int64]) -> NDArray[np.bool_]:

    columns = np.arange(endpoints.shape[0])

    return (columns > endpoints[:, 0][:, None]) & (columns < endpoints[:, 1][:, None])


def positive_relation_mask(relational_matrix: NDArray[np.float64], consideration_mask: NDArray[np.bool_]) -> NDArray[np.bool_]:
    '''
    Produces a 2D boolean matrix s.t. M(i, j) = True denotes bar j
    surpasses resistance line i.

    Parameters: 
    * relational_matrix of NDArray(n,n) s.t. values are some relational value of all bar
    pairs in an n-array of bars. A relational value of > 0 must indicate bar y is greater
    than bar x for relational_matrix(x,y).
    * consideration_mask of NDArray(n,) consisting of boolean values where true indicates bar
    should even be considered as a breakout. For instance, if only green bars should be
    considered as a breakout, then the mask should be True for green bars and false o.w.
    If any bar should be considered, ie. red bar with a top value exceeding resistance line,
    then mask should be an array of True values.

    Output: A boolean matrix of shape NDArray(n,n) representing all breakouts.
    '''

    return (relational_matrix > 0) & consideration_mask

def direction_mask(matrix_mask: NDArray[np.bool_]) -> Callable[[Callable[[tuple[int, int]], NDArray[np.bool_]]], NDArray[np.bool_]]:
    '''
    Produces a function that operates on a breakouts matrix. The output function can
    refine a breakout matrix to an upper or lower breakouts matrix.

    Parameters: NDArray(n,n) breakout matrix of boolean values.

    Output: 
    f(g) -> NDArray(n,n) such that g is a function with a parameter of (n,n)
    and output NDArray(n,n) is a boolean matrix.
    '''

    return lambda direction : matrix_mask & direction(matrix_mask.shape)


def time_ranges(df: pd.DataFrame, ranges: NDArray[np.int64]) -> NDArray[np.str_]:

    '''
    Converts an array of indices to corresponding timestamps located in dataframe.

    Parameters: 
    * dataframe that is time indexed.
    * NDArray of indices.

    Output: 
    NDArray of times as strings with same dimension as ranges parameter. 

    '''
    return df.index.strftime('%H:%M:%S').to_numpy()[ranges]

def lower_bound(prices: NDArray[np.float64], value: float) -> NDArray[np.float64]:

    lt_mask = prices < value

    return prices[lt_mask]

def upper_bound(prices: NDArray[np.float64], value: float) -> NDArray[np.float64]:

    gt_mask = prices > value

    return prices[gt_mask]

def trades_between_bars(trades: pd.DataFrame, time_range: NDArray[np.str_]) -> pd.DataFrame:

    return trades.between_time(start_time=time_range[0], end_time=time_range[1], inclusive="left")

def resistance_k_rmsd(bound: Callable[[NDArray[np.float64], float], NDArray[np.float64]], k: int) -> Callable[[Callable[[str], any]],  NDArray[np.float64]]:
    
    zone_endpt_times = lambda bars, zone_endpoints : time_ranges(bars, zone_endpoints + np.array([1, 0]))

    prices_per_zone = lambda trades, time_ranges: for_each(lambda time_range: trades_between_bars(trades, time_range)['price'], time_ranges)

    get_prices_per_zone = lambda data : prices_per_zone(data('trades'), zone_endpt_times(data('bars'), data('resistance_zone_endpoints')))
    
    get_levels_per_zone = lambda data : data('resistance_levels')

    return lambda data: resistance_k_rmsd_data(bound, k, get_prices_per_zone(data), get_levels_per_zone(data))

def resistance_k_rmsd_data(bound: Callable[[NDArray[np.float64], float], NDArray[np.float64]], k: int, prices_per_zone: NDArray[np.float64], resistance_levels_per_zone: NDArray[np.float64]) -> NDArray[np.float64]:

    k_rmsd_per_zone = for_each(lambda tp, rl: k_root_mean_sqr_dev_1d(bound(tp, rl), k, rl), prices_per_zone, resistance_levels_per_zone)

    return k_rmsd_per_zone

def resistance_euclidean_1d(data: Callable[[str], any]) -> NDArray[np.float64]:

    bar_tops, resistance_points = data('bars')['top'].values, data('resistance_points')

    resistance_point_prices = bar_tops[resistance_points]

    return euclidean_distances_1d(resistance_point_prices[:,0], resistance_point_prices[:,1])

def only_green(bars: pd.DataFrame) -> NDArray[np.bool_]:
    return ~bars['red'].values
def only_red(bars: pd.DataFrame) -> NDArray[np.bool_]:
    return bars['red'].values
def all_bars(bars: pd.DataFrame) -> NDArray[np.bool_]:
    return np.ones_like(bars['red'].values, dtype=bool)

def resistance_breakout_region_function(relational_matrix: NDArray[np.bool_], breakout_consideration_mask: NDArray[np.bool_]) -> Callable[[Callable], NDArray[np.bool_]]:

    breakouts_mask = positive_relation_mask(relational_matrix, breakout_consideration_mask)

    direction_function =  direction_mask(breakouts_mask)

    return lambda f : direction_function(f)

def resistance_zone_endpoints(breakout_region_function: Callable[[Callable], NDArray[np.bool_]]) -> NDArray[np.int64]:
    
    upper_breakouts_mask = breakout_region_function(upper_matrix_tri_mask)

    lower_breakouts_mask = breakout_region_function(lower_matrix_tri_mask)

    zone_endpoints = combined_false_region_endpoints(lower_breakouts_mask, upper_breakouts_mask)

    return zone_endpoints

def pattern_mask_function(red_bars: NDArray[np.bool_], green_bars: NDArray[np.bool_]) -> NDArray[np.bool_]:

    indices = np.arange(red_bars.shape[0])

    return lambda pattern_function, i : indices == np.where(pattern_function(red_bars, green_bars), indices + i, -1)[:, None]

def prev_green_red_pattern_mask(red_bars: NDArray[np.bool_], green_bars: NDArray[np.bool_]) -> NDArray[np.bool_]:
    
    prev_green_mask = np.concatenate(([False], green_bars[:-1]))

    return prev_green_mask & red_bars

def next_green_red_pattern_mask(red_bars: NDArray[np.bool_], green_bars: NDArray[np.bool_]) -> NDArray[np.bool_]:

    next_red_mask = np.concatenate((red_bars[1:], [False]))

    return green_bars & next_red_mask

def exclude_green_red_pattern(bars: pd.DataFrame) -> NDArray[np.bool_]:

    red_bars = only_red(bars)

    green_bars = only_green(bars)

    green_red_pattern_function = pattern_mask_function(red_bars, green_bars)

    return ~(green_red_pattern_function(prev_green_red_pattern_mask,  -1) | green_red_pattern_function(next_green_red_pattern_mask, 1))

def exclude_none(bars: pd.DataFrame) -> NDArray[np.bool_]:

    return np.ones_like(bars.shape, dtype=bool)

def valid_resistance_mask(resistance_zone_endpoints: NDArray[np.int64], resistance_points: NDArray[np.int64]) -> NDArray[np.bool_]:

    return (resistance_zone_endpoints[:, 1] < resistance_zone_endpoints.shape[0]) & \
        (resistance_points[:, 0] != -1)

def resistance_points_mask_function(resistance_zone_endpoints: NDArray[np.int64]) -> NDArray[np.bool_]:

    in_zone_mask = between_endpoints_mask(resistance_zone_endpoints)

    non_self_resistance_mask = ~diagonal_matrix_mask(in_zone_mask.shape[0])

    return lambda m : in_zone_mask & non_self_resistance_mask & m

def resistance_points(relational_matrix: NDArray[np.bool_], resistance_points_mask: NDArray[np.bool_]) -> NDArray[np.int64]:

    inf_relational_matrix = np.where(~resistance_points_mask, np.inf, relational_matrix)
    
    point_1_indx = np.arange(inf_relational_matrix.shape[0])

    point_2_indx_ = np.argmin(inf_relational_matrix, axis=1)

    point_2_indx = np.where((point_2_indx_ == 0) & (inf_relational_matrix[:,0] == np.inf), -1, point_2_indx_)

    points = np.sort(np.column_stack((point_1_indx, point_2_indx)), axis=1)

    return points

def resistance_data(bars: pd.DataFrame, trades: pd.DataFrame, breakout_consideration: Callable[[pd.DataFrame], NDArray[np.bool_]], exclude_pattern: Callable[[pd.DataFrame], NDArray[np.bool_]], op1: Callable, op2: Callable):

    relational_matrix = matrix_operation_1d(diff_1d, bars['top'].values)

    breakout_region_function = resistance_breakout_region_function(relational_matrix, breakout_consideration(bars))

    zone_endpoints = resistance_zone_endpoints(breakout_region_function)

    points_mask_function = resistance_points_mask_function(zone_endpoints)

    points_mask = points_mask_function(exclude_pattern(bars))

    points = resistance_points(relational_matrix, points_mask)

    valid_resistance = valid_resistance_mask(zone_endpoints, points)

    data_dict = {'bars': bars, 'trades': trades, 'resistance_levels': bars['top'].values[valid_resistance], 'resistance_zone_endpoints': zone_endpoints[valid_resistance], 'resistance_points': points[valid_resistance]}
    
    data_f = lambda i : data_dict[i]

    print(op1, op2)

    return np.column_stack((data_f('resistance_levels'), data_f('resistance_zone_endpoints'), data_f('resistance_points'), op1(data_f), op2(data_f)))


def resistance_data_function(bars: pd.DataFrame, trades: pd.DataFrame) -> Callable[..., NDArray[np.any]]:

    bars_per_day = time_based_partition(bars, '1D')

    trades_per_day = time_based_partition(trades, '1D')

    return lambda breakout_consideration, exclude_pattern, op1, op2: for_each(partial(resistance_data, breakout_consideration=breakout_consideration, exclude_pattern=exclude_pattern, op1=op1, op2=op2), bars_per_day, trades_per_day)

'''
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

print(f"perc slope {perc_slope}")
print(perc_slope.shape)

top_diffs_2d = for_each(lambda t: np.column_stack((np.arange(t.shape[0]),t)),bar_top_values)

euclid_dist_2d = compose_functions([for_each, apply_upper_matrix_tri, matrix_operation_xd, euclidean_distances_xd])

print(f"euclid dist 2d {euclid_dist_2d(top_diffs_2d)}")
print(euclid_dist_2d(top_diffs_2d).shape)

slopes = compose_functions([for_each, matrix_operation_xd, slopes_2d])

print(f"slopes {slopes(top_diffs_2d)}")
print(slopes(top_diffs_2d).shape)

bar_dfs = time_based_partition(bars,"1D")
first_bar_df = bar_dfs[0]
first_bar_df_green_mask = ~first_bar_df['red'].values

green_breakouts = breakouts_mask(slopes(top_diffs_2d)[0], first_bar_df_green_mask)
first_upper_green_breakouts = green_breakouts(upper_matrix_tri_mask, last_occurence_indices)
print(np.argmax(first_upper_green_breakouts, axis=1))
print(first_bar_df['top'].values)

bar_time_range = time_ranges(first_bar_df, np.array([[0,2],[3,9],[4,8]]))

print(bar_time_range)

print(first_bar_df.between_time(start_time=bar_time_range[0,0], end_time=bar_time_range[0,1]))

'''