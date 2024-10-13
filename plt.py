import mplfinance as mpf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from functools import partial

def plot_resistance(bar_df_for_day: pd.DataFrame, resistance_index: np.array, resistance_level: float, resistance_data: np.array, date: str) -> None:
    
    print(f'resistance index {resistance_index}')
    print(f'resistance level {resistance_level}')
    print(bar_df_for_day)
    # Add the horizontal resistance line
    fig, ax = plt.subplots(figsize=(12, 8))

    mpf.plot(bar_df_for_day, type='candle', ax=ax, volume=False, style='charles')

    ax.plot([resistance_index[0], resistance_index[1]], [resistance_level, resistance_level], color='red', linestyle='--', linewidth=1.5)

    # Set labels and title
    ax.set_title(f'Bars with Resistance Line {resistance_level} for {date} with data {resistance_data}')
    ax.set_xlabel('Time')
    ax.set_ylabel('Price')

    # Show the plot
    plt.show()

def plot_resistances(bar_df_for_day: pd.DataFrame, resistance_levels: np.array, resistance_indices: np.array, resistance_data: np.array) -> None:

    chart_date = bar_df_for_day.index[0].date()  # Get the date from the first index entry

    plot_res_func = partial(plot_resistance, bar_df_for_day, date=str(chart_date))
    
    list(map(lambda params: plot_res_func(resistance_index=params[0], resistance_level=params[1], resistance_data=params[2]), zip(resistance_indices, resistance_levels, resistance_data)))
