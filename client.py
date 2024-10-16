import os
from alpaca_trade_api.rest import REST
from  alpaca_trade_api.rest_async import AsyncRest
from alpaca_trade_api.stream import Stream


# Alpaca API client setup for trading, streaming, and historical data.

def initialize(paper: bool) -> dict:

    key_id = os.getenv('ALPACA_API_KEY') if paper else os.getenv('ALPACA_API_KEY_LIVE')
    secret_key = os.getenv('ALPACA_SECRET_KEY') if paper else os.getenv('ALPACA_SECRET_KEY_LIVE')
    base_url = 'https://paper-api.alpaca.markets' if paper else 'https://api.alpaca.markets'
   
    clients = {
        'trade': lambda: REST(key_id, secret_key, base_url, 'v2' ),
        'data': lambda: AsyncRest(key_id, secret_key,'https://data.alpaca.markets', 'v2'),
        'stream': lambda: Stream(key_id, secret_key, base_url, 'https://stream.data.alpaca.markets', 'sip')
    }

    return lambda client_str: clients[client_str]()

