from datetime import datetime, timedelta, timezone
import pandas as pd
import numpy as np

from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import CryptoFeed

from indicators import add_indicators
from config import data_client

FEE = 0.001
SLIPPAGE = 0.0005

def fetch_bars(symbol='BTC/USD', timeframe=TimeFrame.Hour, days=120):
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    
    req = CryptoBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=timeframe,
        start=start,
        end=end
    )

    bars = data_client.get_crypto_bars(req, feed=CryptoFeed.US).df
    df = bars.xs(symbol)
    df = df.tz_convert('UTC').sort_index()

    return df[['open', 'high', 'low', 'close', 'volume']]

def backtest(df: pd.DataFrame):
    df = add_indicators(df).dropna().copy()
    position = 0
    entry = 0
    equity = 10_000.0
    trades = []

    for ts, row in df.iterrows():
        px = float(row['close'])
        if position == 0 and row['signal_long']:
            entry = px * (1 + SLIPPAGE)
            position = 1
            trades.append({'time': ts, 'side': 'BUY', 'price': entry, 'equity': equity})
        elif position == 1 and row['signal_flat']:
            exitp = px * (1 - SLIPPAGE)
            ret = (exitp /entry) * (1-FEE)**2 - 1
            equity *= (1 + ret)
            trades.append({'time': ts, 'side': 'SELL', 'price': exitp, 'equity': equity})
            position = 0
        
    if position == 1:
        px = float(df['close'].iloc[-1])
        exitp = px * (1 - SLIPPAGE)
        ret = (exitp / entry) * (1 - FEE) ** 2 - 1
        equity *= (1 + ret)
        trades.append({'time': df.index[-1], 'side': 'SELL', 'price': exitp, 'equity': equity})
    
    tr = pd.DataFrame(trades)
    returns = tr.loc[tr['side'] == 'SELL', 'equity'].pct_change().dropna()
    cumret = equity / 10_000.0 - 1
    dd = (tr['equity'].cummax() - tr['equity']) / tr['equity'].cummax()
    maxdd = float(dd.max()) if not dd.empty else 0.0
    sharpe = (returns.mean() / (returns.std() + 1e-9)) * np.sqrt(365) if len(returns) > 1 else 0.0

    return {
        'final_equity': round(equity, 2),
        'cumulative_return_%': round(cumret * 100, 2),
        'max_drawdown_%': round(maxdd * 100, 2),
        'num_trades': int((tr['side'] == 'SELL').sum()),
        'sharpe_ratio': round(float(sharpe), 2)
    }, tr

if __name__ == "__main__":
    df = fetch_bars('BTC/USD', TimeFrame.Minute, days=120)
    summary, trades = backtest(df)
    print(summary)
    trades.to_csv('trades.csv', index=False)
    

