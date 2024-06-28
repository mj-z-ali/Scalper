import pandas as pd
import pytz
from datetime import datetime, time, timedelta
from alpaca_trade_api.rest import REST, TimeFrame, TimeFrameUnit
from functools import partial
import client



def list_dates(start_date_str: str, end_date_str: str, freq: str='1D') -> list[str]:

    date_format = "%Y-%m-%d"

    start_date = datetime.strptime(start_date_str, date_format).date()

    end_date =  datetime.strptime(end_date_str, date_format).date()

    date_range = pd.date_range(start=start_date, end=end_date, freq=freq)
    
    date_list = date_range.strftime(date_format).tolist()

    return date_list
'''   
    def _number_of_days_in_between(start_date_str: str, end_date_str: str) -> int:

        date_format = "%Y-%m-%d"
        start_date = datetime.strptime(start_date_str, date_format)
        end_date =  datetime.strptime(end_date_str, date_format)

        delta = end_date - start_date

        return delta.days
'''

def add_days_to_date(date_str: str, days_to_add: int):

    date_format = "%Y-%m-%d"

    date_obj = datetime.strptime(date_str, date_format)

    new_date_obj = date_obj + timedelta(days=days_to_add)

    new_date_str = new_date_obj.strftime(date_format)

    return new_date_str

    
def check_dst(utc_date: datetime) -> bool:

    # Convert UTC date to 'America/New_York' timezone
    eastern_dt = utc_date.astimezone(pytz.timezone('America/New_York'))

    return eastern_dt.dst() != timedelta(0)
    
def apply_time_filter(group: pd.DataFrame) -> pd.DataFrame:

    # Check DST status for the date of the group
    is_dst = check_dst(group.index[0])


    # DST time range: 13:30 to 20:00 local time
    start_time = time(13, 30) if is_dst else time(14, 30)
    end_time = time(20, 0) if is_dst else time(21, 0)
    
    return group.between_time(start_time, end_time, inclusive="left")

def start_min_end_date_pair(days_to_add: int, curr_end_date: str, curr_start_date: str) -> tuple[str, str]:

    min_end_date = min(curr_end_date, add_days_to_date(date_str=curr_start_date, days_to_add=days_to_add))
    return curr_start_date, min_end_date

def fetch_bars(client: REST, symbol: str, timeframe: TimeFrame, start_end_date_pair: tuple[str, str]) -> pd.DataFrame:
    return client.get_bars(symbol=symbol, timeframe=timeframe, start=start_end_date_pair[0], end=start_end_date_pair[1]).df

def bars(client: REST, symbol: str = 'SPY', timeframe: TimeFrame = TimeFrame(5, TimeFrameUnit.Minute), start_date: str = "2023-12-18", 
            end_date: str = "2023-12-18") -> pd.DataFrame:

    date_list = list_dates(start_date_str=start_date, end_date_str=end_date, freq='366D')

    start_end_date_pairs = list(map(partial(start_min_end_date_pair, 365, end_date), date_list))

    bars_df_list = list(map(partial(fetch_bars, client, symbol, timeframe), start_end_date_pairs))

    bars_df = pd.concat(bars_df_list)

    if bars_df.empty:
        empty_df = pd.DataFrame(columns=["timestamp","close","high","low","trade_count","open","volume","vwap"]).set_index('timestamp')
        return empty_df
    
    # Group by date and apply the time filter
    open_market_bars_df = bars_df.groupby(bars_df.index.date) \
        .apply(apply_time_filter) \
        .reset_index().set_index('timestamp')

    return open_market_bars_df.drop(columns=["level_0"])

def fetch_trades(client: REST, symbol: str, date: str) -> pd.DataFrame:
    return client.get_trades(symbol=symbol, start=date, end=date).df

def trades(client: REST, symbol: str = "SPY", start_date: str = "2023-12-18", 
                end_date: str = "2023-12-18") -> pd.DataFrame:
        
        date_list = list_dates(start_date_str=start_date, end_date_str=end_date, freq="1D")

        trades_df_list = list(map(partial(fetch_trades, client, symbol), date_list))

        trades_df = pd.concat(trades_df_list)

        if trades_df.empty:
            empty_df = pd.DataFrame(columns=["timestamp","conditions","id","price","size","exchange","tape"]).set_index('timestamp')
            return empty_df
        
        # Group by date and apply the time filter
        open_market_trades_df = trades_df.groupby(trades_df.index.date) \
            .apply(apply_time_filter) \
            .reset_index().set_index("timestamp")

        return open_market_trades_df.drop(columns=["level_0"])
