import time
from datetime import datetime, timezone, timedelta
import pandas as pd

from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.enums import CryptoFeed

from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

from indicators import add_indicators
from config import data_client, trading_client

SYMBOL = 'BTC/USD'
TIMEFRAME = TimeFrame.Day
NOTIONAL_USD = 100.0

def latest_df(n=300):
    now = datetime.now(timezone.utc)
    start = now - timedelta(days=3)
    req = CryptoBarsRequest(
        symbol=SYMBOL,
        timeframe=TIMEFRAME,
        start=start,
        end=now,
    )
    
    bars = data_client.get_crypto_bars(req, feed=CryptoFeed.US).df
    df = bars.xs(SYMBOL).tz_convert('UTC').sort_index()

    return add_indicators(df).dropna().tail(n)

def submit_market(side: OrderSide, notional: float):
    order = MarketOrderRequest(
        symbol_or_symbols=SYMBOL,
        side=side,
        notional=notional,
        time_in_force=TimeInForce.IOC
    )
    return trading_client.submit_order(order)

in_position = False

print("Starting live trading...")

while True:
    try:
        df = latest_df()
        row = df.iloc[-1]
        px = float(row['close'])

        if not in_position and row['signal_long']:
            resp = submit_market(OrderSide.BUY, NOTIONAL_USD)
            in_position = True
            print(f"BUY at {px} - Order ID: {resp.id}")
        
        elif in_position and row['signal_flat']:
            resp = submit_market(OrderSide.SELL, NOTIONAL_USD)
            in_position = False
            print(f"SELL at {px} - Order ID: {resp.id}")

        time.sleep(60)  # Wait for 1 minute before next check

    except KeyboardInterrupt:
        break
    except Exception as e:
        print(f"Error: {e}")
        time.sleep(10)