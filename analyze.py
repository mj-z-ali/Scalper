import pandas as pd
import numpy as np
import k_means as km



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

def diff_matrix(matrix: np.array) -> np.array:

    n = len(matrix)

    matrix_reshaped = matrix.reshape((n, 1))

    diff_matrix = 100*(matrix_reshaped - matrix_reshaped.T) / matrix_reshaped

    upper_row, upper_columns = np.triu_indices(n=n, k=1)

    return diff_matrix[upper_row, upper_columns]



trades = pd.read_csv('small_trades.csv', parse_dates=['timestamp'], index_col='timestamp')

bars_list = bars(trades_df=trades, interval='5T')

ohlctb_df = ohlctb(bars=bars_list)

top_differences = np.abs(np.array([diff_matrix(matrix=five_minute_chart_df['top'].values)
                        for _, five_minute_chart_df in ohlctb_df.resample('1D') 
                        if not five_minute_chart_df.dropna().empty]).flatten())

print(f"top_differences shape  {top_differences.shape}")

zeroes = np.zeros(top_differences.shape[0])

top_differences_2d = np.column_stack((top_differences, zeroes))

print(f"top differences_2d shap  {top_differences_2d.shape}")


centroids, labels, k, score = km.k_means_fit(data=top_differences_2d, max_k=25)

print(f"centroids, labels, k, score {centroids, labels, k, score}")
# ohlctb_df.to_csv('out.csv', index=True)

