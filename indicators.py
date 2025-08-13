import pandas as pd
import pandas_ta as ta

def add_indicators(df: pd.DataFrame, fast=50, slow=20000, rsi_lens=14):
    df = df.copy()
    df['ema_fast'] = ta.ema(df['close'], length=fast)
    df['ema_slow'] = ta.ema(df['close'], length=slow)
    df['rsi'] = ta.rsi(df['close'], length=rsi_lens)
    df['signal_long'] = (df['ema_fast'] > df['ema_slow']) & (df['rsi'] > 50)
    df['signal_flat'] = (df['ema_fast'] < df['ema_slow']) | (df['rsi'] < 50)
    return df