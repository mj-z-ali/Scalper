import pandas as pd
import numpy as np
import datetime
import client as Client
import asyncio
from dataclasses import dataclass
from typing import Callable
from numpy.typing import NDArray
from alpaca.data.models import Bar, Trade
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest, StopOrderRequest, GetOrdersRequest
from alpaca.trading.enums import OrderSide, OrderType, OrderClass, TimeInForce, TradeEvent, QueryOrderStatus
from alpaca.trading.models import TradeUpdate
from alpaca.trading.client import TradingClient
import threading


@dataclass(frozen=True)
class Bar_State:
    _size: int=0
    _1m_highs: NDArray[np.float64]=np.array([])
    _1m_lows: NDArray[np.float64]=np.array([])
    _1m_tops: NDArray[np.float64]=np.array([])
    _1m_bottoms: NDArray[np.float64]=np.array([])

@dataclass(frozen=True)
class Line_State:
    _1m_r_lines: NDArray[np.float64]=np.array([])
    _1m_s_lines: NDArray[np.float64]=np.array([])

@dataclass(frozen=True)
class Live_State:
    _price: np.float64=0.0
    _open: np.float64=0.0
    _high: np.float64=-np.inf
    _low: np.float64=np.inf

@dataclass(frozen=True)
class Lock_State:
    _buy: bool=False

@dataclass(frozen=True)
class Order_State:
    _position_counts: tuple[int, ...]=(0,)*10
    _unique_ids: tuple[int, ...]=(0,)*10
    _prices: tuple[int, ...]=(0.0,)*10

def diff_matrix(a: NDArray[np.float64]) -> NDArray[np.float64]:
    return lambda b: a - b.reshape(-1,1)

def upper_tri_mask(s: tuple[np.uint64, np.uint64]) -> NDArray[np.bool_]:
    return np.triu(np.ones(s, dtype=np.bool_), k=1)

def lower_tri_mask(s: tuple[np.uint64, np.uint64]) -> NDArray[np.bool_]:
    return np.tril(np.ones(s, dtype=np.bool_), k=-1)

def left_mask(m: NDArray[np.bool_]) -> NDArray[np.bool_]:
    return m & lower_tri_mask(m.shape)

def right_mask(m: NDArray[np.bool_]) -> NDArray[np.bool_]:
    return m & upper_tri_mask(m.shape)

def last_points(m: NDArray[np.bool_]) -> NDArray[np.uint64]:
    return m.shape[1] - np.argmax(m[:,::-1], axis=1) - 1

def first_points(m: NDArray[np.bool_]) -> NDArray[np.uint64]:
    return np.argmax(m, axis=1)

def right_boundary_points(m: NDArray[np.bool_]) -> NDArray[np.uint64]:
    
    default_points = lambda p: np.where(p == 0, len(p), p)

    return default_points(first_points(m))

def left_boundary_points(m: NDArray[np.bool_]) -> NDArray[np.uint64]:

    default_points = lambda p: np.where(p == len(p)-1, -1, p)

    return default_points(last_points(m))

def q_r_matrix(highs: NDArray[np.float64]) -> Callable:

    q = diff_matrix(highs)(highs)

    return lambda lower_range: lambda upper_range: (q >= lower_range) & (q <= upper_range)

def q_b_matrix(tops: NDArray[np.float64]) -> NDArray[np.bool_]:

    def q_b(highs: NDArray) -> NDArray[np.bool_]:

        q_mask = diff_matrix(tops)(highs) > 0
        cols = q_mask.shape[1]

        return (cols > left_boundary_points(left_mask(q_mask))[:,None]) & \
            (cols < right_boundary_points(right_mask(q_mask))[:,None])
    return q_b

def resistance_lines(bar_state:Bar_State) -> NDArray[np.float64]:

    q_r = q_r_matrix(bar_state._1m_highs)(-0.01)(0)

    q_b = q_b_matrix(bar_state._1m_tops)(bar_state._1m_highs)

    return bar_state._1m_highs[(q_r & q_b).sum(axis=1) >= 2]

def deflection_zone(res_points:NDArray[np.float64]) -> np.float64:

    return lambda p: lambda eps: (indx:=np.searchsorted(res_points, p), 
                res_points[indx-1]+eps if indx > 0 else -np.inf
            )[1] 

def buy_resistance(state:tuple[Line_State, Live_State]) -> bool:

    line_state, live_state = state

    return (live_state._price>live_state._open) and (live_state._price==live_state._high) and \
        ((live_state._open-live_state._low) >= 3*(live_state._price-live_state._open)) and \
        (live_state._low <= deflection_zone(line_state._1m_r_lines)(live_state._price)(0.05))


def find_available_order_id(order_state: Order_State) -> int:

    return order_state._position_counts.index(0) if 0 in order_state._position_counts else -1

def generate_call_market_buy(trade: Trade) -> MarketOrderRequest:

    return lambda qty: lambda order_id: MarketOrderRequest(
        symbol= f"{trade.symbol}{trade.timestamp.strftime('%y%m%d')}C{int(int(trade.price)*1000):08d}",
        qty= qty,
        side= OrderSide.BUY,
        type= OrderType.MARKET,
        time_in_force= TimeInForce.DAY,
        order_class= OrderClass.SIMPLE,
        client_order_id= str(order_id)
    )

def generate_call_limit_sell(symbol: str) -> Callable:

    return lambda qty: lambda price: lambda order_id: LimitOrderRequest(
        symbol=symbol, 
        qty=qty, 
        side= OrderSide.SELL,
        type= OrderType.LIMIT,
        limit_price= price,
        time_in_force= TimeInForce.DAY,
        order_class= OrderClass.SIMPLE,
        client_order_id= str(order_id)
    )

def generate_call_stop_sell(symbol: str) -> Callable:

    return lambda qty: lambda price: lambda order_id: StopOrderRequest(
        symbol=symbol, 
        qty=qty, 
        side= OrderSide.SELL,
        stop_price= price,
        time_in_force= TimeInForce.DAY, 
        client_order_id= str(order_id)
    )


def streams(client:Client) -> Callable:

    async def stock_data(t_handlers: tuple[Callable,Callable]) -> None:
        stock_stream_client = client('crypto_stream')
        stock_stream_client.subscribe_bars(t_handlers[0],'BTC/USD')
        stock_stream_client.subscribe_trades(t_handlers[1],'BTC/USD')
        await stock_stream_client.run()

    async def trade_update(handler:Callable) -> None:
        trade_stream_client = client('trade_stream')
        trade_stream_client.subscribe_trade_updates(handler)
        await trade_stream_client.run()
    
    async def run(t_handlers: tuple[Callable, Callable, Callable]) -> None:
        await asyncio.gather(stock_data(t_handlers[:-1]), trade_update(t_handlers[-1]))
    return run

def main():

    client = Client.initialize(paper=True)

    initial_state = (Bar_State(), Line_State(), Live_State(), Order_State(), Lock_State())
    
    def create_stream_handler(state: tuple[Bar_State, Line_State, Live_State]) -> Callable:

        trade_client = client('trade')
        bar_state, line_state, live_state, order_state, lock_state = state

        async def on_bar(bar:Bar) -> None:
            print(bar)

            
            nonlocal bar_state, line_state, live_state, lock_state

            new_size = bar_state._size+1
            new_1m_highs = np.append(bar_state._1m_highs,bar.high)
            new_1m_lows = np.append(bar_state._1m_lows, bar.low)
            bar_top, bar_bottom = (bar.close,bar.open) if bar.close >= bar.open else (bar.open,bar.close)
            new_1m_tops = np.append(bar_state._1m_tops,bar_top)
            new_1m_bottoms = np.append(bar_state._1m_bottoms,bar_bottom)

            new_bar_state = Bar_State(new_size, new_1m_highs, new_1m_lows, new_1m_tops, new_1m_bottoms)

            new_1m_r_lines = np.sort(resistance_lines(new_bar_state)) if new_size >= 2 else line_state._1m_r_lines
            # new_1m_s_lines = np.sort(support_lines(new_bar_state)) if new_size >= 2 else line_state._1m_s_lines
            new_1m_s_lines=0
            new_line_state = Line_State(new_1m_r_lines, new_1m_s_lines)
                
            new_live_state = Live_State()

            new_lock_state = Lock_State()

            bar_state, line_state, live_state, lock_state = new_bar_state, new_line_state, new_live_state, new_lock_state

            return None
        
        async def on_trade(trade:Trade) -> None:
            
            nonlocal bar_state, line_state, live_state, order_state, lock_state

            if trade.timestamp.time() > datetime.time(16,0,0):
                print(f'bar state {bar_state}')
                print(f'line state {line_state}')
                print(f'live state {live_state}')

            new_price = trade.price
            new_open = trade.price if live_state._open == 0.0 else live_state._open
            new_high = trade.price if trade.price > live_state._high else live_state._high
            new_low = trade.price if trade.price < live_state._low else live_state._low

            new_live_state = Live_State(new_price,new_open,new_high,new_low)

            if buy_resistance((line_state,new_live_state)) and (trade_client.get_account().cash >= 27_000) and (not lock_state._buy):

                if (available_indx:=find_available_order_id(order_state)) != -1:
                    order_id = f'{order_state._unique_ids[available_indx]}_{available_indx}'
                    qty=10
                    trade_client.submit_order(generate_call_market_buy(trade)(qty)(order_id))

                    new_order_state = Order_State(
                        order_state._position_counts[:available_indx] + (qty,) + order_state._position_counts[available_indx+1:], 
                        order_state._unique_ids[:available_indx] + (order_state._unique_ids[available_indx]+1,) + order_state._unique_ids[available_indx+1:],
                        order_state._prices
                    )

                    order_state = new_order_state

                    lock_state = Lock_State(True)


        async def on_trade_update(trade_update: TradeUpdate):
            
            nonlocal order_state

            if (trade_update.event == TradeEvent.FILL) and (trade_update.order.side == OrderSide.BUY):
                
                unique_id,index = [int(i) for i in trade_update.order.client_order_id.split('_')]

                limit_id = f'{unique_id+2}_{index}'
                stop_id = f'{unique_id+1}_{index}'

                limit_sell_req = generate_call_limit_sell(trade_update.order.symbol)(1)(round(trade_update.price+0.1,2))(limit_id)
                stop_sell_req = generate_call_stop_sell(trade_update.order.symbol)(1)(max(0.01,round(trade_update.price-0.05),2))(stop_id)

                trade_client.submit_order(limit_sell_req)
                trade_client.submit_order(stop_sell_req)

                order_state = Order_State(
                    order_state._position_counts, 
                    order_state._unique_ids[:index] + (order_state._unique_ids[index]+2,) + order_state._unique_ids[index+1:],
                    order_state._prices[:index] + (trade_update.price,) + order_state._prices[index+1:]
                )

            elif (trade_update.event == TradeEvent.FILL) and (trade_update.order.type == OrderType.LIMIT):

                unique_id,index = [int(i) for i in trade_update.order.client_order_id.split('_')]

                order=trade_client.get_order_by_client_id(f'{unique_id-1}_{index}')
                trade_client.cancel_order_by_id(order.id)
                
                new_position_count = order_state._position_counts[index] - int(trade_update.order.qty)
                ref_price = float(trade_update.order.limit_price)

                if new_position_count > 0:

                    limit_id = f'{unique_id+2}_{index}'
                    stop_id = f'{unique_id+1}_{index}'

                    limit_sell_req = generate_call_limit_sell(trade_update.order.symbol)(1)(round(ref_price+0.05,2))(limit_id)
                    stop_sell_req = generate_call_stop_sell(trade_update.order.symbol)(1)(max(0.01,round(ref_price-0.1),2))(stop_id)

                    trade_client.submit_order(limit_sell_req)
                    trade_client.submit_order(stop_sell_req)

                order_state = Order_State(
                    order_state._position_counts[:index] + [new_position_count] + order_state._position_counts[index+1:], 
                    order_state._unique_ids[:index] + (order_state._unique_ids[index]+2,) + order_state._unique_ids[index+1:],
                    order_state._prices
                )
            elif (trade_update.event == TradeEvent.FILL) and (trade_update.order.type == OrderType.STOP):
                unique_id,index = [int(i) for i in trade_update.order.client_order_id.split('_')]

                order=trade_client.get_order_by_client_id(f'{unique_id+1}_{index}')
                trade_client.cancel_order_by_id(order.id)

                new_position_count = order_state._position_counts[index] - int(trade_update.order.qty)
                ref_price = float(trade_update.order.stop_price)

                if new_position_count > 0:

                    limit_id = f'{unique_id+3}_{index}'
                    stop_id = f'{unique_id+2}_{index}'

                    stop_sell_qty = new_position_count if (ref_price-0.05) <= (order_state._prices[index] - 0.1) else 1

                    limit_sell_req = generate_call_limit_sell(trade_update.order['symbol'])(1)(round(ref_price+0.1,2))(limit_id)
                    stop_sell_req = generate_call_stop_sell(trade_update.order['symbol'])(stop_sell_qty)(max(0.01,round(ref_price-0.05),2))(stop_id)

                    trade_client.submit_order(limit_sell_req)
                    trade_client.submit_order(stop_sell_req)

                order_state = Order_State(
                    order_state._position_counts[:index] + [new_position_count] + order_state._position_counts[index+1:], 
                    order_state._unique_ids[:index] + (order_state._unique_ids[index]+2,) + order_state._unique_ids[index+1:],
                    order_state._prices
                )
                
        return on_bar, on_trade, on_trade_update
    

    # asyncio.run(streams(client)(create_stream_handler(initial_state)))
    # Set up the stream handlers
    t_handlers = create_stream_handler(initial_state)

    # Run the streams in the already running loop
    loop = asyncio.get_event_loop()
    nest_asyncio.apply()  # This allows re-entering the running event loop, useful in certain environments

    try:
        # Schedule the coroutine without blocking the existing loop
        loop.run_until_complete(streams(client)(t_handlers))
    except KeyboardInterrupt:
        print("Shutting down streams gracefully.")
    finally:
        # Closing the loop explicitly (optional, good for cleanup)
        loop.close()

if __name__ == "__main__":

    

    main()
    '''
    try:
        order = trade_client.submit_order(
            symbol='SPY241202C00604000',
            qty=1,
            side='buy',
            type='market',           # Use 'limit' if you want to specify a limit price
            time_in_force='day',
            order_class='simple'     # Options are generally 'simple' orders
        )
        print("Order submitted:")
        print(order)
    except Exception as e:
        print(f"An error occurred: {e}")
    
    

    positions = trade_client.list_positions()

    for position in positions:
        if position.symbol == 'SPY241202C00604000':
            print("Current position in the option:")
            print(f"Symbol: {position.symbol}")
            print(f"Quantity: {position.qty}")
            print(f"Current Price: {position.current_price}")
            print(f"Market Value: {position.market_value}")
    '''