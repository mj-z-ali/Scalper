import pandas as pd
import numpy as np
from functools import reduce, partial
import k_means as km
import acquire
import time



def non_empty_df_intervals(df: pd.DataFrame, interval: str) -> list[pd.DataFrame]:

    df_interval_generator = map(lambda df_pair: df_pair[1], df.resample(interval))

    return list(filter(lambda df_interval: not df_interval.dropna().empty, df_interval_generator))



def trade_intervals_list(trades_df: pd.DataFrame, interval: str) -> list[pd.DataFrame]:

    return non_empty_df_intervals(df=trades_df, interval=interval)
    


def bars_from_trade_intervals(trade_intervals: list[pd.DataFrame]) -> pd.DataFrame:

    red_bar = lambda trade_interval : trade_interval['price'].iloc[-1] <= trade_interval['price'].iloc[0]
    top = lambda trade_interval : trade_interval['price'].iloc[0] if red_bar(trade_interval) else trade_interval['price'].iloc[-1]
    bottom = lambda trade_interval : trade_interval['price'].iloc[-1] if red_bar(trade_interval) else trade_interval['price'].iloc[0] 

    bar_generator = map(lambda trade_interval: {
                'timestamp': trade_interval.index[0].floor('S'),
                'open': trade_interval['price'].iloc[0], 
                'high': trade_interval['price'].max(), 
                'low': trade_interval['price'].min(), 
                'close': trade_interval['price'].iloc[-1],
                'top':top(trade_interval),
                'bottom': bottom(trade_interval),
                'red': red_bar(trade_interval)}, trade_intervals)

    return pd.DataFrame(bar_generator).set_index('timestamp')



def insert_top_bottom_columns(bar_df: pd.DataFrame) -> pd.DataFrame:

    red_lambda = lambda bar_row : bar_row['close'] <= bar_row['open']
    top_lambda = lambda bar_row : bar_row['open'] if red_lambda(bar_row) else bar_row['close']
    bottom_lambda = lambda bar_row : bar_row['close'] if red_lambda(bar_row) else bar_row['open'] 

    return bar_df.assign(top=bar_df.apply(top_lambda, axis=1), bottom=bar_df.apply(bottom_lambda,axis=1), 
                red=bar_df.apply(red_lambda, axis=1))


def diff_matrix(matrix: np.array) -> np.array:

    n = len(matrix)

    matrix_reshaped = matrix.reshape((n, 1))

    diff_matrix = 100*(matrix_reshaped - matrix_reshaped.T) / matrix_reshaped

    return diff_matrix

def top_differences_day(day_bars_df: pd.DataFrame) -> np.array:

    n = len(day_bars_df['top'].values)

    upper_row, upper_col = np.triu_indices(n=n, k=1)

    return diff_matrix(matrix=day_bars_df['top'].values)[upper_row, upper_col]


def top_differences(bar_df: pd.DataFrame) -> np.array:

    bar_day_interv_df_list = non_empty_df_intervals(df=bar_df, interval="1D")

    top_diffs_list = map(top_differences_day, bar_day_interv_df_list)

    return reduce(lambda acc, x: np.append(acc, x), top_diffs_list, np.array([]))


def top_slopes_day(day_bars_df: pd.DataFrame) -> np.array:

    n = len(day_bars_df['top'].values)

    # Upper triangle indices excluding diagonal since it and the lower part 
    # is relationship of bar to itself and past bars.
    upper_row, upper_col = np.triu_indices(n=n, k=1)

    # Percentage differences of bar top to all others.
    diff_mat = diff_matrix(matrix=day_bars_df['top'].values)[upper_row, upper_col]

    indices = np.arange(n).reshape(1,n)

    # Line lengths in units of bars. Bar top to all other bar top distances.
    lengths = (indices - indices.T)[upper_row, upper_col]

    slopes = diff_mat / lengths

    return np.column_stack((slopes, lengths))

def top_slopes(bar_df: pd.DataFrame) -> np.array:

    bar_day_interv_df_list = non_empty_df_intervals(df=bar_df, interval="1D")

    top_slopes_list = map(top_slopes_day, bar_day_interv_df_list)

    return reduce(lambda acc, x: np.concatenate((acc, x)), top_slopes_list, np.empty((0, 2)))

def find_last_true_index(bool_matrix: np.array) -> np.array:
    return bool_matrix.shape[1] - np.argmax(bool_matrix[:,::-1], axis=1) - 1

def resistance_zone(day_bars_df: pd.DataFrame, bar_top_diffs: np.array) -> np.array:

    # Bar tops differences that are greater than each current bar top diffs.
    ltz_bool = bar_top_diffs < 0

    # Upper triangle not including diagonal. 
    # In particular, all bar top diffs after the current.
    triu_bool = np.triu(np.ones(bar_top_diffs.shape, dtype=bool), k=1)

    # First breakout occurence after each current bar.
    frst_brkout_indices = np.argmax(ltz_bool&triu_bool, axis=1)
    
    col_indices = np.arange(bar_top_diffs.shape[1])

    # Set all values including and after first breakout to True in order to change to inf later.
    brkout_mask = (col_indices >= frst_brkout_indices[:, None]) & (frst_brkout_indices[:, None] != 0)

    # Lower triangle excluding diagonal.
    tril_bool = np.tril(np.ones(bar_top_diffs.shape, dtype=bool), k=-1)

    # First bar top diffs greater than each current that occured before.
    pre_brkout_indices = find_last_true_index(bool_matrix=ltz_bool&tril_bool)
    
    # Set all  values including and before first pre-breakout to True in order to change to inf later.
    pre_brkout_mask = (col_indices <= pre_brkout_indices[:, None]) & (pre_brkout_indices[:, None] != bar_top_diffs.shape[1]-1)

    # Mask where only diagonal is set to true. This represents the current 
    # bar top in relation to itself. We exclude this from the resistance zone.
    diagonal_mask = np.eye(bar_top_diffs.shape[0], dtype=bool)

    # If bar was red.
    red_bool = day_bars_df['red'].values

    # If  previous bar was red.
    prev_red_bool = np.concatenate(([False], red_bool[:-1]))

    # If next bar was red.
    nxt_red_bool = np.concatenate((red_bool[1:], [False]))

    # If there was a green-red pattern starting from the previous bar.
    prev_green_curr_red_bool = ~prev_red_bool & red_bool

    # If there was a a green-red pattern starting at the current bar.
    curr_green_nxt_red_bool = ~red_bool & nxt_red_bool

    # If green-red starting at previous, set to the previous bar index.
    prev_green_curr_red_indices =  np.where(prev_green_curr_red_bool, np.arange(len(prev_green_curr_red_bool)) - 1, -1)

    # Set that previous bar to True in order to ignore it in the future.
    prev_green_curr_red_mask = (col_indices == prev_green_curr_red_indices[:, None])

    # If green-red starting at current, set to next bar index.
    curr_green_nxt_red_indices = np.where(curr_green_nxt_red_bool, np.arange(len(curr_green_nxt_red_bool)) + 1, -1)

    # Set that next bar to True in order to ignore it in the future.
    curr_green_nxt_red_mask = (col_indices == curr_green_nxt_red_indices[:, None])

    # Combine all masks to retain the resistance zone mask.
    resistance_zone_mask = pre_brkout_mask | prev_green_curr_red_mask | diagonal_mask | \
                        curr_green_nxt_red_mask | brkout_mask 
    
    # Results in the resistance zone left unchanged (set to bar_top_diffs). 
    # Any values beyond this zone are set to inf.
    return np.where(resistance_zone_mask, np.inf, bar_top_diffs)

'''
(1) Get index of min for each row. That is the other bar that creates the
resistance line. If argmin is 0 and inf in that location, then there is none.

'''
def resistance_start_end_point_indices(day_bars_df: np.array, bar_top_diffs: np.array) -> np.array:

    # Filter out bar top differences beyond the zone (set to inf).
    resistance_zone_mat = resistance_zone(day_bars_df=day_bars_df, bar_top_diffs=bar_top_diffs)

    # First point of resistance line is the row index.
    point_1_indx_ = np.arange(resistance_zone_mat.shape[0])

    # The bar top with the smallest difference to the current bar top is another point.
    # If argmin is zero, it may be that inf was the minimum, meaning no qualifying points...
    point_2_indx_ = np.argmin(resistance_zone_mat, axis=1)

    # So, correct that case.
    point_2_indx = np.where((point_2_indx_ == 0) & (resistance_zone_mat[:,0] == np.inf), -1, point_2_indx_)

    point_1_indx = np.where(point_2_indx == -1, -1, point_1_indx_)

    # Stack current point indx and other points indx and sort from  start to end.
    # Both point indices are -1 if no resistance line.
    end_point_indx = np.sort(np.column_stack((point_1_indx, point_2_indx)), axis=1)
    
    return end_point_indx

'''

(2) If argmin is greater than current row index, argmin is where the line ends.
If argmin is less than the current row index, row index is where the line ends.
The exact start time of our trade range we seek is the start time of the bar 
that succeeds the end index.

(3) Find the first bar after end index that is green and has a top greater than current. 
The bar after it is our end time of the range we seek.
'''
def green_bars(day_bars_df: pd.DataFrame) -> np.array:
    return ~day_bars_df['red'].values

def first_green_brkout_index(day_bars_df: pd.DataFrame, bar_top_diffs: np.array, resistance_end_pt: np.array) -> np.array:

    col_indices = np.arange(bar_top_diffs.shape[1])

    # Mask to retrieve bars after end-points.
    after_end_pt_bool = (col_indices > resistance_end_pt[:, None]) & (resistance_end_pt[:, None] != -1)

    # Mast to retrieve green bars.
    green_bars_mask = green_bars(day_bars_df=day_bars_df)

    # Mask to retrieve green bars that are breakouts.
    green_brkout_bool =  (bar_top_diffs < 0) & green_bars_mask

    # Mask to retrieve green breakout bars after end-points.
    green_brkout_after_end_pt_bool = green_brkout_bool & after_end_pt_bool

    # Indices of first green breakout bars after end-points.
    frst_green_brkout_indx_ = np.argmax(green_brkout_after_end_pt_bool, axis=1)

    # Clean, since argmax returns 0 when green breakout bar after end-point not found.
    frst_green_brkout_indx = np.where((frst_green_brkout_indx_ == 0) & ~green_brkout_after_end_pt_bool[:, 0], -1, frst_green_brkout_indx_)
    
    return frst_green_brkout_indx

def range_indices(day_bars_df: pd.DataFrame, bar_top_diffs: np.array, resistance_start_end_pts: np.array) -> np.array:

    # End-points to resistance lines.
    resist_end_pt = resistance_start_end_pts[:, 1]

    # Start index for desired range takes place at bar after resistance line end-point.
    # range_start_indx = np.where(resist_end_pts != -1, resist_end_pts + 1, resist_end_pts)

    # First green breakout bar after resistance end-point surpassing the resistance line.
    frst_green_brkout = first_green_brkout_index(day_bars_df=day_bars_df, bar_top_diffs=bar_top_diffs, resistance_end_pt=resist_end_pt)

    # End index for desired range is the bar after the first green breakout.
    # range_end_indx = np.where(frst_green_brkout != -1, frst_green_brkout + 1, frst_green_brkout)

    return np.column_stack((resist_end_pt, frst_green_brkout))

def bar_interval_in_minutes(day_bars_df: pd.DataFrame) -> int:
    # Find bar interval:
    time_diffs = day_bars_df.index.to_series().diff().dropna()

    interval_minutes = time_diffs.dt.total_seconds() / 60

    common_interval = interval_minutes.mode()[0]

    return common_interval

def matching_trade_interval(day_trades_df: pd.DataFrame, day_bars_df: pd.DataFrame) -> list[pd.DataFrame]:
    
    bar_interval = bar_interval_in_minutes(day_bars_df=day_bars_df)

    # Trades grouped in bar intervals.
    return trade_intervals_list(trades_df=day_trades_df, interval=f"{bar_interval}T")

def trade_range_list(trade_interval: list[pd.DataFrame], range_index: np.array):
    return trade_interval[range_index[0]:range_index[1]]


def trades_df_in_range(bars_for_day_df: pd.DataFrame, trades_for_day_df: pd.DataFrame, range_indices: np.array) -> pd.DataFrame:
    
    resist_end_bar_indx = range_indices[0]

    brkout_bar_indx = range_indices[1]

    brkout_bar_time = bars_for_day_df.index[brkout_bar_indx].strftime('%H:%M:%S')

    bar_interval = bar_interval_in_minutes(bars_for_day_df)

    start_time = bars_for_day_df.index[resist_end_bar_indx].strftime('%H:%M:%S')

    end_time = acquire.add_minutes_to_time(time_str=brkout_bar_time, minutes_to_add=bar_interval)

    return trades_for_day_df.between_time(start_time=start_time, end_time=end_time, inclusive="left")

def momentum_data(resistance_level: float, trade_df: pd.DataFrame) -> np.array:

    trade_prices = trade_df['price'].values

    trade_sizes = trade_df['size'].values

    upper_resist_bool = (trade_prices > resistance_level)

    lower_resist_bool = (trade_prices < resistance_level)
    
    trade_sizes_upper_resist = trade_sizes[upper_resist_bool]

    trade_sizes_lower_resist = trade_sizes[lower_resist_bool]

    trade_occurence = trade_df.index.to_series().diff().dt.total_seconds().fillna(0)

    trade_occur_upper_resist = trade_occurence[upper_resist_bool]

    trade_occur_lower_resist = trade_occurence[lower_resist_bool]

    trade_occur_diff = trade_occur_upper_resist.mean() / trade_occur_lower_resist.mean()

    trade_sizes_diff = trade_sizes_upper_resist.mean() / trade_sizes_lower_resist.mean()

    trades_lower_count = len(trade_prices[(trade_prices > (resistance_level-0.05)) & (trade_prices < resistance_level)])

    trades_upper_count = len(trade_prices[(trade_prices > resistance_level) & (trade_prices < (resistance_level+0.05))])

    trades_count_diff = trades_upper_count / trades_lower_count

    
    return trade_occur_diff

def resistance_slopes_data(bar_top_diffs: np.array, resistance_points: np.array):

    row_indices = np.arange(bar_top_diffs.shape[0]).reshape(-1,1)

    # One bar top diff is the resistance line iteself, which will have 
    # a diff of 0 to itself. The other bar creates the resistance line,
    # which has a top diff  of ≥ 0 relative to the resistance line.
    # This is because we selected the next highest bar after the current
    # resistance line to define the second point. 
    slope_numerator = np.max(bar_top_diffs[row_indices, resistance_points], axis=1).reshape(-1,1)

    # Number of bars in between the two points.
    # This can never be 0 as we filtered out ineligible 
    # resistance lines.
    slope_denominator = np.diff(resistance_points)


    return slope_numerator / slope_denominator

def data_each_day(bars_for_day_df: pd.DataFrame, trades_for_day_df: pd.DataFrame):

    # Percentage differences of all bar top pairs. 
    bar_top_diffs = diff_matrix(matrix=bars_for_day_df['top'].values)

    resist_pts = resistance_start_end_point_indices(day_bars_df=bars_for_day_df, bar_top_diffs=bar_top_diffs)

    range_pts = range_indices(day_bars_df=bars_for_day_df, bar_top_diffs=bar_top_diffs, resistance_start_end_pts=resist_pts)

    valid_resist_bool = (range_pts[:, 0] != -1) & (range_pts[:, 1] != -1)

    resist_slopes_data = resistance_slopes_data(bar_top_diffs=bar_top_diffs[valid_resist_bool], resistance_points=resist_pts[valid_resist_bool])

    valid_range_pts = range_pts[valid_resist_bool]

    trade_df_btwn_range_pts = list(map(partial(trades_df_in_range, bars_for_day_df, trades_for_day_df), valid_range_pts))

    valid_resistance_levels =  bars_for_day_df["top"].values[valid_resist_bool]

    moment_data = np.array(list(map(momentum_data, valid_resistance_levels, trade_df_btwn_range_pts)))
    
    print(np.column_stack((
        np.arange(valid_range_pts.shape[0]),
        valid_range_pts,
        valid_resistance_levels,
        resist_slopes_data,
        moment_data,
        np.array([trade_df['price'].max() for trade_df in trade_df_btwn_range_pts])
    )))

    return np.column_stack((resist_slopes_data, moment_data))

def data(bar_df: pd.DataFrame, trade_df: pd.DataFrame):

    bar_day_interv_df_list = non_empty_df_intervals(df=bar_df, interval="1D")

    trades_day_interv_df_list = non_empty_df_intervals(df=trade_df, interval="1D")

    data_generator = map(data_each_day, bar_day_interv_df_list, trades_day_interv_df_list)

    return reduce(lambda acc, x : np.concatenate((acc, x)), data_generator, np.empty((0,2)))
    
    

'''

(4) Get the trade df of for the  current date. Split the trades into bar intervals only
if the time is greater than or equal start time and less than end time from steps (2) and (3),
respectively.

(5) The top prices from the bar df are the resistance levels for each corresponding row.
For the ones that qualify from step (1), search the array from step (4) for the first trade
that exceeds this resistance level in each bar. Average the +second trade volume and divide it by the
average of the -second trade volume.



'''

'''


data = top_slopes(bar_df=bar_df_)

centroids, labels = km.train(data=data, number_of_centroids=10, iteration=0, max_iterations=100)



def resistance_levels(bar_df: pd.DataFrame, centroids: np.array) -> tuple[np.array, np.array]:

    n = len(bar_df['top'].values)

    top_diff_2d = diff_matrix(matrix=bar_df['top'].values)

    top_diffs = np.abs(top_diff_2d.flatten())

    zeroes = np.zeros(top_diffs.shape[0])

    top_diff_points = np.column_stack((top_diffs, zeroes))

    labels = km.k_means_model(data=top_diff_points, centroids=centroids)

    labels_2d = labels.reshape(n,n)

    centroid_nearest_zero = np.argmin(centroids, axis=0)[0]

    break_condition = (top_diff_2d < 0) & (labels_2d != centroid_nearest_zero)

    upper_tri_mask = np.triu(np.ones(top_diff_2d.shape, dtype=bool), k=1)

    cut_off_indices = np.argmax(break_condition & upper_tri_mask, axis=1)

    col_indices = np.arange(n)

    cut_off_mask = (col_indices >= cut_off_indices[:, None]) & (cut_off_indices[:, None] != 0)

    resistance_condition = (~cut_off_mask) & (upper_tri_mask) & (labels_2d == centroid_nearest_zero)

    bar_tops = bar_df['top'].values

    bar_tops_tiled = np.tile(bar_tops, (n, 1)).T

    start_end_indices = np.column_stack(np.where(resistance_condition))

    return bar_tops_tiled[resistance_condition], start_end_indices





def area_matrix(matrix: np.array) -> np.array:

    n = len(matrix)

    matrix_reshaped = matrix.reshape((n, 1))

    diff_matrix  = 100*(matrix_reshaped - matrix_reshaped.T) / matrix_reshaped

    lower_row, lower_columns = np.tril_indices(n=n, k=0)

    diff_matrix[lower_row, lower_columns] = 0

    area_matrix = diff_matrix.cumsum(axis=1)

    total_area = np.sum(area_matrix)

    area_ratio = area_matrix

    upper_row, upper_columns = np.triu_indices(n=n, k=1)

    return area_ratio[upper_row, upper_columns]

# start_time_for_read_csv  = time.time()
# trades = pd.read_csv('18_12_2023_5M.csv', parse_dates=['timestamp'], index_col='timestamp')
# end_time_for_read_csv = time.time()

# start_time_for_creating_bars = time.time()
# bars_list = bars(trades_df=trades, interval='5T')
# end_time_for_creating_bars = time.time()

# start_time_for_creating_ohlctb = time.time()
# ohlctb_df = ohlctb(bars=bars_list)
# end_time_for_creating_ohlctb = time.time()

# ohlctb_csv = ohlctb_df.to_csv('small_ohlctb_df', index=True)

# ohlctb_df = pd.read_csv('18_12_2023_5M.csv', parse_dates=['timestamp'], index_col='timestamp')

# start_time_for_top_differences = time.time()
# top_differences = np.abs(np.array([diff_matrix(matrix=five_minute_chart_df['top'].values)
#                         for _, five_minute_chart_df in ohlctb_df.resample('1D') 
#                         if not five_minute_chart_df.dropna().empty]).flatten())
# end_time_for_top_differences = time.time()

medium_five_minute_df = pd.read_csv('01_04_2024_to_01_05_2024_5M.csv', parse_dates=['timestamp'], index_col='timestamp')

medium_five_minute_ohlctb_df = insert_top_bottom_columns(bar_df=medium_five_minute_df)

top_diffs = np.abs(top_differences(bar_df=medium_five_minute_ohlctb_df))

zeroes = np.zeros(top_diffs.shape[0])

top_diffs_2d = np.column_stack((top_diffs, zeroes))

centroids, labels, k, score = km.k_means_fit(data=top_diffs_2d, start_k=500, max_k=501)
print(f"centroids, labels, k, score {centroids, labels, k, score}")

# areas = np.array([area_matrix(matrix=five_minute_chart_df['top'].values)
#                         for _, five_minute_chart_df in ohlctb_df.resample('1D') 
#                         if not five_minute_chart_df.dropna().empty]).flatten()

# print(f"time_for_read_csv {end_time_for_read_csv-start_time_for_read_csv}")
# print(f"time_for_creating_bars {end_time_for_creating_bars-start_time_for_creating_bars}")
# print(f"time_for_creating_ohlctb {end_time_for_creating_ohlctb-start_time_for_creating_ohlctb}")
# print(f"time_for_top_differences {end_time_for_top_differences-start_time_for_top_differences}")

# zeroes = np.zeros(top_differences.shape[0])

# top_differences_2d = np.column_stack((top_differences, zeroes))

# print(f"small_differences shape  {top_differences.shape}")
# print(f"top differences_2d shape  {top_differences_2d.shape}")


# centroids, labels, k, score = km.k_means_fit(data=top_differences_2d, start_k=25, max_k=50)

# print(f"centroids, labels, k, score {centroids, labels, k, score}")

# medium_centroids_csv = centroids.to_csv('medium_centroids.csv', index=True)
# ohlctb_df.to_csv('out.csv', index=True)

# centroids = np.array(
# [[0.09664392, 0.        ],
# [1.13651666, 0.        ],
# [0.44012713, 0.        ],
# [0.23816486, 0.        ],
# [0.72944893, 0.        ],
# [0.01223435, 0.        ],
# [1.33493495, 0.        ],
# [0.92921289, 0.        ],
# [0.16389102, 0.        ],
# [0.32862124, 0.        ],
# [0.57206279, 0.        ],
# [1.19805531, 0.        ],
# [0.06704812, 0.        ],
# [1.00281727, 0.        ],
# [0.82669234, 0.        ],
# [0.38175386, 0.        ],
# [0.19932711, 0.        ],
# [0.12932954, 0.        ],
# [0.27988558, 0.        ],
# [0.64591077, 0.        ],
# [1.73549001, 0.        ],
# [0.50263636, 0.        ],
# [1.06943946, 0.        ],
# [1.26135835, 0.        ],
# [0.03902596, 0.        ]]
# )

'''