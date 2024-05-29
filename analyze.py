import pandas as pd
import numpy as np
import k_means as km

import time


def bars(trades_df: pd.DataFrame, interval: str = '5T') -> list[pd.DataFrame]:

    return [bar for _, bar in trades_df.resample(interval) if not bar.dropna().empty]


def ohlctb(bars: list[pd.DataFrame]) -> pd.DataFrame:

    ohlc = [{'timestamp': bar_df.index[0].floor('S'),
                'open': bar_df['price'].iloc[0], 
                'high': bar_df['price'].max(), 
                'low': bar_df['price'].min(), 
                'close': bar_df['price'].iloc[-1]} for bar_df in bars]
    
    ohlc_df = pd.DataFrame(ohlc).set_index('timestamp')

    red_bars = ohlc_df['close'] <= ohlc_df['open']
    ohlc_df['top'] = np.where(red_bars, ohlc_df['open'], ohlc_df['close'])
    ohlc_df['bottom'] = np.where(red_bars, ohlc_df['close'], ohlc_df['open'])
    ohlc_df['red'] = red_bars
    
    return ohlc_df

def insert_top_bottom_columns(bar_df: pd.DataFrame) -> pd.DataFrame:

    bar_df_copy = bar_df.copy(deep=True)

    red_bars = bar_df_copy['close'] <= bar_df_copy['open']
    
    bar_df_copy['top'] = np.where(red_bars, bar_df_copy['open'], bar_df_copy['close'])

    bar_df_copy['bottom'] = np.where(red_bars, bar_df_copy['close'], bar_df_copy['open'])

    bar_df_copy['red'] = red_bars

    return  bar_df_copy

def top_differences(bar_df: pd.DataFrame) -> np.array:

    top_differences = np.abs(np.array([diff_matrix(matrix=current_bar_df['top'].values)
                        for _, current_bar_df in bar_df.resample('1D') 
                        if not current_bar_df.dropna().empty]).flatten())
    
    return top_differences

def resistance_levels(bar_df: pd.DataFrame, centroids: np.array) -> np.array:

    top_diffs = top_differences(bar_df=bar_df)

    zeroes = np.zeros(top_diffs.shape[0])

    top_diffs_2d = np.column_stack((top_diffs, zeroes))   

    bar_tops = bar_df['top'].values

    n = len(bar_tops)

    bar_tops_tiled = np.tile(bar_tops, (n, 1))

    upper_row, upper_columns = np.triu_indices(n=n, k=1)

    bar_tops_upper_tri = bar_tops_tiled[upper_row][upper_columns]

    labels = km.k_means_model(data=top_diffs_2d, centroids=centroids)

    centroid_nearest_zero = np.argmin(centroids, axis=0)[0]

    small_top_diff_indices = np.where(labels==centroid_nearest_zero)

    return bar_tops_upper_tri[small_top_diff_indices]

def diff_matrix(matrix: np.array) -> np.array:

    n = len(matrix)

    matrix_reshaped = matrix.reshape((n, 1))

    diff_matrix = 100*(matrix_reshaped - matrix_reshaped.T) / matrix_reshaped

    upper_row, upper_columns = np.triu_indices(n=n, k=1)

    return diff_matrix[upper_row, upper_columns]

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

centroids = np.array(
[[0.09664392, 0.        ],
[1.13651666, 0.        ],
[0.44012713, 0.        ],
[0.23816486, 0.        ],
[0.72944893, 0.        ],
[0.01223435, 0.        ],
[1.33493495, 0.        ],
[0.92921289, 0.        ],
[0.16389102, 0.        ],
[0.32862124, 0.        ],
[0.57206279, 0.        ],
[1.19805531, 0.        ],
[0.06704812, 0.        ],
[1.00281727, 0.        ],
[0.82669234, 0.        ],
[0.38175386, 0.        ],
[0.19932711, 0.        ],
[0.12932954, 0.        ],
[0.27988558, 0.        ],
[0.64591077, 0.        ],
[1.73549001, 0.        ],
[0.50263636, 0.        ],
[1.06943946, 0.        ],
[1.26135835, 0.        ],
[0.03902596, 0.        ]]
)