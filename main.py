import pandas as pd
import numpy as np
import client
import acquire
import matplotlib.pyplot as plt
from alpaca_trade_api.rest import REST, TimeFrame, TimeFrameUnit




pd.set_option('display.max_columns', None)

def main() -> int:

    # data_client = Client(paper=True).establish_client()

    # acquire = Acquire(data_client=data_client)

    # trades = acquire.bars(symbol='SPY', timeframe=TimeFrame(5, TimeFrameUnit.Minute), start_date="2024-04-01", end_date="2024-05-01")

    # trades.to_csv('01_04_2024_to_01_05_2024_5M.csv', index=True)

    params = client.init(paper=True)

    data_client = client.establish_client(params=params)

    bars_df = acquire.bars(client=data_client, symbol="SPY", timeframe=TimeFrame(5, TimeFrameUnit.Minute), start_date="2024-06-24", end_date="2024-06-27")

    bars_df.to_csv('2024_06_24_to_2024_06_27_5M.csv', index=True)

    trades_df = acquire.trades(client=data_client, symbol="SPY", start_date="2024-06-24", end_date="2024-06-27")
    
    trades_df.to_csv('2024_06_24_to_2024_06_27_tick.csv', index=True)

    return 0


if __name__ == "__main__":
    main()