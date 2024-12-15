import os
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.live import CryptoDataStream, StockDataStream
from alpaca.data.enums import CryptoFeed, DataFeed
from alpaca.trading.client import TradingClient
from alpaca.trading.stream import TradingStream
from typing import Callable



# Alpaca API client setup for trading, streaming, and historical data.

def initialize(paper: bool) -> Callable:

    key_id = os.getenv('ALPACA_API_KEY') if paper else os.getenv('ALPACA_API_KEY_LIVE')
    secret_key = os.getenv('ALPACA_SECRET_KEY') if paper else os.getenv('ALPACA_SECRET_KEY_LIVE')

    clients = {
        'stock_data': lambda: StockHistoricalDataClient(key_id, secret_key),
        'trade': lambda: TradingClient(api_key=key_id, secret_key=secret_key, paper=paper),
        'trade_stream': lambda: TradingStream(api_key=key_id, secret_key=secret_key, paper=paper),
        'stock_stream': lambda: StockDataStream(api_key=key_id, secret_key=secret_key, feed=DataFeed.IEX),
        'crypto_stream': lambda: CryptoDataStream(api_key=key_id, secret_key=secret_key, feed=CryptoFeed.US)
    }

    return lambda client_str: clients[client_str]()

