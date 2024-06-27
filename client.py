import os
from alpaca_trade_api.rest import REST
from alpaca_trade_api.stream import Stream

# Alpaca API client setup for trading, streaming, and historical data.

def init(paper: bool, api_key: str = None, secret_key: str = None) -> dict:
    API_KEY_ENV_VAR = 'ALPACA_API_KEY' if paper else 'ALPACA_API_KEY_LIVE'
    SECRET_KEY_ENV_VAR = 'ALPACA_SECRET_KEY' if paper else 'ALPACA_SECRET_KEY_LIVE'
    BASE_URL = 'https://paper-api.alpaca.markets/v2' if paper else 'https://api.alpaca.markets'
    return { 
        'key_id': api_key or os.getenv(API_KEY_ENV_VAR), 
        'secret_key': secret_key or os.getenv(SECRET_KEY_ENV_VAR), 
        'base_url': BASE_URL 
    }

# Set up the Alpaca client for historical data and executing trades
def establish_client(params: dict, api_version: str = 'v2') -> REST:
    return REST(**params, api_version=api_version) 

# Set up the Alpaca client for streaming live data 
def establish_stream(params: dict) -> Stream:
    return Stream(**params)
    
