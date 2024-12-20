import pandas as pd
import numpy as np
from datetime import datetime, time, timedelta
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
import ipdb
import math

@dataclass(frozen=True)
class Bar_State:
    _size: int=0
    _bars: tuple=()
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
    _time: np.float64=np.inf
    _open_time: np.float64=np.inf
    _buffer: tuple=()
    _open: np.float64=0.0
    _high: np.float64=-np.inf
    _low: np.float64=np.inf
    _close: np.float64=0.0

@dataclass(frozen=True)
class Lock_State:
    _buy: bool=False

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

    return lambda lower_range: lambda upper_range: (np.isclose(q,lower_range) | (q > lower_range)) & (q <= upper_range)

def q_b_matrix(tops: NDArray[np.float64]) -> NDArray[np.bool_]:

    def q_b(highs: NDArray) -> NDArray[np.bool_]:
        
        q_mask = diff_matrix(tops)(highs) > 0
        cols = np.arange(q_mask.shape[1])

        return (cols > left_boundary_points(left_mask(q_mask))[:,None]) & \
            (cols < right_boundary_points(right_mask(q_mask))[:,None])
    return q_b

B = Bar_State(_1m_highs=np.array([1,2,1,1.99,8]),_1m_tops=np.array([0.9,1.9,0.98,1.89,7]))
def resistance_lines(bar_state:Bar_State) -> NDArray[np.float64]:
    
    print('entered resistance_line function')
    q_r = q_r_matrix(bar_state._1m_highs)(-0.01)(0)
    print('processed q_r')
    q_b = q_b_matrix(bar_state._1m_tops)(bar_state._1m_highs)
    print('processed q_b')
    return bar_state._1m_highs[(q_r & q_b).sum(axis=1) >= 2]


def deflection_zone(res_points:NDArray[np.float64]) -> np.float64:
    
    return lambda p: lambda eps: (indx:=np.searchsorted(res_points, p), 
                res_points[indx-1]+eps if indx > 0 else -np.inf
            )[1] 

def buy_resistance(state:tuple[Line_State, Live_State]) -> bool:
    
    line_state, live_state = state

    return (live_state._close>live_state._open) and (live_state._close==live_state._high) and \
        ((live_state._open-live_state._low) >= 3*(live_state._close-live_state._open)) and \
        (live_state._low <= deflection_zone(np.sort(line_state._1m_r_lines))(live_state._close)(0.05))



def generate_call_market_buy(trade: Trade) -> MarketOrderRequest:

    return lambda qty: MarketOrderRequest(
        symbol= f"{trade.symbol}{trade.timestamp.strftime('%y%m%d')}C{int(int(trade.price)*1000):08d}",
        qty= qty,
        side= OrderSide.BUY,
        type= OrderType.MARKET,
        time_in_force= TimeInForce.DAY,
        order_class= OrderClass.SIMPLE
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
        client_order_id= order_id
    )

def generate_call_stop_sell(symbol: str) -> Callable:

    return lambda qty: lambda price: lambda order_id: StopOrderRequest(
        symbol=symbol, 
        qty=qty, 
        side= OrderSide.SELL,
        stop_price= price,
        time_in_force= TimeInForce.DAY, 
        client_order_id= order_id
    )


def streams(client:Client) -> Callable:

    def stock_data_task(t_handlers: tuple[Callable,Callable,Callable]) -> Callable:
        stock_stream_client = client('stock_stream')
        stock_stream_client.subscribe_bars(t_handlers[0],'SPY')
        stock_stream_client.subscribe_updated_bars(t_handlers[1],'SPY')
        stock_stream_client.subscribe_trades(t_handlers[2],'SPY')
        return stock_stream_client.run

    def trade_update_task(handler:Callable) -> Callable:
        trade_stream_client = client('trade_stream')
        trade_stream_client.subscribe_trade_updates(handler)
        return trade_stream_client.run
    
    def run(t_handlers: tuple[Callable, Callable, Callable, Callable]) -> tuple[Callable, Callable]:
        return stock_data_task(t_handlers[:-1]), trade_update_task(t_handlers[-1])
    return run


def sale_code(conditions:list[str]) -> int:

    table = {
        ' ': 1,
        'A': 1,
        'B': 0,
        'C': 0,
        'D': 1,
        'E': 1,
        'F': 1,
        'G': 2,
        'H': 0,
        'I': 0,
        'K': 1,
        'L': 3,
        'M': 0,
        'N': 0,
        'O': 1,
        'P': 2,
        'Q': 0,
        'R': 0,
        'S': 1,
        'T': 0,
        'U': 0,
        'V': 0,
        'W': 0,
        'X': 1,
        'Y': 1,
        'Z': 2,
        '1': 1,
        '4': 2,
        '5': 1,
        '6': 1,
        '7': 0,
        '8': 0,
        '9': 1
    }

    return  math.prod(set(map(lambda condition: table[condition], conditions)))

def main():

    client = Client.initialize(paper=True)

    initial_state = (Bar_State(), Line_State(), Live_State(), Lock_State())
    
    def create_stream_handler(state: tuple[Bar_State, Line_State, Live_State, Lock_State]) -> Callable:

        trade_client = client('trade')
        bar_state, line_state, live_state, lock_state = state

        async def on_bar(bar:Bar) -> None:
            print(f'bar {bar}\n')
            
            if bar.timestamp.time() < time(14,30):
                print('Closed')
                return
            nonlocal bar_state, line_state, live_state, lock_state

            new_size = bar_state._size+1
            new_bars = bar_state._bars + (bar,)
            new_1m_highs = np.append(bar_state._1m_highs,bar.high)
            new_1m_lows = np.append(bar_state._1m_lows, bar.low)
            bar_top, bar_bottom = (bar.close,bar.open) if bar.close >= bar.open else (bar.open,bar.close)
            new_1m_tops = np.append(bar_state._1m_tops,bar_top)
            new_1m_bottoms = np.append(bar_state._1m_bottoms,bar_bottom)

            new_bar_state = Bar_State(new_size, new_bars, new_1m_highs, new_1m_lows, new_1m_tops, new_1m_bottoms)

            new_1m_r_lines = resistance_lines(new_bar_state) if new_size >= 2 else line_state._1m_r_lines
            # new_1m_s_lines = np.sort(support_lines(new_bar_state)) if new_size >= 2 else line_state._1m_s_lines
            new_1m_s_lines=0
            new_line_state = Line_State(new_1m_r_lines, new_1m_s_lines)

            print(f"bar_state {new_bar_state}\n")
            print(f"line_state {new_line_state}\n")
            print(f"old live_state {live_state}\n")
                
            new_live_state = Live_State()

            print(f"new live_state {new_live_state}\n")

            print(f"old lock_state {lock_state}\n")

            new_lock_state = Lock_State()

            print(f"new lock_state {new_lock_state}\n")

            bar_state, line_state, live_state, lock_state = new_bar_state, new_line_state, new_live_state, new_lock_state
            
            return None
        
        async def on_bar_update(bar:Bar) -> None:
            print(f'bar update {bar}\n')
            if bar.timestamp.time() < time(14,30):
                print('Closed')
                return
            nonlocal bar_state, line_state

            new_bars = bar_state._bars[:-1] + (bar,)
            new_1m_highs = np.append(bar_state._1m_highs[:-1],bar.high)
            new_1m_lows = np.append(bar_state._1m_lows[:-1], bar.low)
            bar_top, bar_bottom = (bar.close,bar.open) if bar.close >= bar.open else (bar.open,bar.close)
            new_1m_tops = np.append(bar_state._1m_tops[:-1],bar_top)
            new_1m_bottoms = np.append(bar_state._1m_bottoms[:-1],bar_bottom)

            new_bar_state = Bar_State(bar_state._size, new_bars, new_1m_highs, new_1m_lows, new_1m_tops, new_1m_bottoms)

            new_1m_r_lines = np.sort(resistance_lines(new_bar_state)) if bar_state._size >= 2 else line_state._1m_r_lines
            # new_1m_s_lines = np.sort(support_lines(new_bar_state)) if new_size >= 2 else line_state._1m_s_lines
            new_1m_s_lines=0
            new_line_state = Line_State(new_1m_r_lines, new_1m_s_lines)

            bar_state, line_state = new_bar_state, new_line_state
        
        async def on_trade(trade:Trade) -> None:
            print(f'trade {trade}\n')
            if trade.timestamp.time() < time(14,30):
                print('closed')
                return
            nonlocal bar_state, line_state, live_state, lock_state
            if (bar_state._bars != tuple()) and trade.timestamp.timestamp() < bar_state._bars[-1].timestamp.timestamp()+60:
                return None
            if trade.timestamp.timestamp() <= live_state._time:
                new_time = trade.timestamp.timestamp() if live_state._time == np.inf else live_state._time
                new_open_time = min(trade.timestamp.timestamp(), live_state._open_time)
                new_open = trade.price if trade.timestamp.timestamp() < live_state._open_time else live_state._open
                new_high = max(trade.price, live_state._high)
                new_low = min(trade.price, live_state._low)
                new_close = trade.price if live_state._time == np.inf else live_state._close
                live_state = Live_State(new_time, new_open_time, live_state._buffer, new_open, new_high, new_low, new_close)
            
            else:
                new_buffer = live_state._buffer + (trade,)
                live_state = Live_State(live_state._time, live_state._open_time, new_buffer, live_state._open, live_state._high, live_state._low, live_state._close)

            if datetime.now().timestamp() >= live_state._time + 0.5:

                if buy_resistance((line_state,live_state)) and (trade_client.get_account().cash >= 27_000) and (not lock_state._buy) and (trade.timestamp.time() < time(20,0,0)):
                        print('Buying')
                        print(f"trigger state {live_state}\n")
                        qty=10
                        trade_client.submit_order(generate_call_market_buy(trade)(qty))
                        lock_state = Lock_State(True)
                
                sorted_buffer = tuple(sorted(live_state._buffer, key= lambda t: t.timestamp.timestamp()))
                if sorted_buffer != tuple():
                    new_trade = sorted_buffer[0]
                    new_time = new_trade.timestamp.timestamp()
                    new_open_time = min(new_trade.timestamp.timestamp(), live_state._open_time)
                    new_open = new_trade.price if new_trade.timestamp.timestamp() < live_state._open_time else live_state._open
                    new_high = max(new_trade.price, live_state._high)
                    new_low = min(new_trade.price, live_state._low)
                    new_close = new_trade.price
                    live_state = Live_State(new_time, new_open_time, sorted_buffer[1:], new_open, new_high, new_low, new_close)
                else:
                    live_state = Live_State(np.inf, live_state._open_time, live_state._buffer, live_state._open, live_state._high, live_state._low, live_state._close)
        

        async def on_trade_update(trade_update: TradeUpdate):
            
            if (trade_update.event == TradeEvent.FILL) and (trade_update.order.side == OrderSide.BUY):
                print(f'Bought {trade_update.order.symbol}')
                unique_id, max_stop_loss = int(trade_update.execution_id), round(trade_update.price-0.1,2)
                
                
                limit_id, stop_id = f'{unique_id}_{max_stop_loss}_{trade_update.qty}_0', f'{unique_id}_{max_stop_loss}_{trade_update.qty}_1'
                limit_price, stop_price = round(trade_update.price+0.1,2), max(0.01,round(trade_update.price-0.05,2))
                print(f'limit price {limit_price}')
                print(f'stop price {stop_price}')
                limit_sell_req = generate_call_limit_sell(trade_update.order.symbol)(1)(limit_price)(limit_id)
                stop_sell_req = generate_call_stop_sell(trade_update.order.symbol)(1)(stop_price)(stop_id)
                
                print("\nsubmitting limit sell")
                trade_client.submit_order(limit_sell_req)
                print('submitting stop sell')
                trade_client.submit_order(stop_sell_req)

            elif (trade_update.event == TradeEvent.FILL) and (trade_update.order.side == OrderSide.SELL):

                print(f"sell id {trade_update.order.client_order_id}\n")
                unique_id, max_stop_loss, qty, is_stop_sell = [float(n) if i == 1 or i == 2 else int(n) for i,n in enumerate(trade_update.order.client_order_id.split('_'))]

                order_to_cancel=trade_client.get_order_by_client_id(f'{unique_id}_{max_stop_loss}_{qty}_{int(not is_stop_sell)}')
                print(f"canceling order {order_to_cancel}\n")
                trade_client.cancel_order_by_id(order_to_cancel.id)
                
        
                new_qty = qty - trade_update.qty
                print(f'new qty {new_qty}')
                if new_qty > 0:
                    
                    limit_id, stop_id = f'{unique_id}_{max_stop_loss}_{new_qty}_0', f'{unique_id}_{max_stop_loss}_{new_qty}_1'

                    limit_price, stop_price = round(float(trade_update.order.stop_price)+0.1,2), max(0.01,round(float(trade_update.order.stop_price)-0.05,2)) if is_stop_sell else round(float(trade_update.order.limit_price)+0.05,2), max(0.01,round(float(trade_update.order.limit_price)-0.1,2))
                    print(f'limit price {limit_price}')
                    print(f'stop price {stop_price}')
                    stop_sell_qty = new_qty if stop_price <= max_stop_loss else 1

                    limit_sell_req = generate_call_limit_sell(trade_update.order.symbol)(1)(limit_price)(limit_id)
                    stop_sell_req = generate_call_stop_sell(trade_update.order.symbol)(stop_sell_qty)(stop_price)(stop_id)

                    print("\nsubmitting limit sell")
                    trade_client.submit_order(limit_sell_req)
                    print('submitting stop sell')
                    trade_client.submit_order(stop_sell_req)
                
        return on_bar, on_bar_update, on_trade, on_trade_update
    

    t_handlers = create_stream_handler(initial_state)

    target_0, target_1 = streams(client)(t_handlers)

    

    print('starting threads')
    thread_0=threading.Thread(target=target_0)
    thread_1=threading.Thread(target=target_1)

    thread_0.start()
    thread_1.start()
    print('finishing')
    thread_0.join()
    thread_1.join()


    

if __name__ == "__main__":

    main()
    