# garch_hour_backtest_simple.py
"""
Simplest hourly GARCH(1,1) backtest for BTC/USD using Alpaca crypto data.

- Direction: long-only
- Sizing: inverse volatility from GARCH(1,1) (Student-t) forecast
- Costs: applied only when position changes
- No signals, no hysteresis, no caps beyond max_leverage

Requirements:
  pip install alpaca-py arch statsmodels pandas numpy python-dotenv
  config.py must provide an authenticated `data_client` (alpaca.data client)
"""

from __future__ import annotations
from math import sqrt
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd

from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import CryptoFeed

from arch import arch_model
from config import data_client

# ---------------- Settings (minimal) ----------------
SYMBOL = "BTC/USD"
TIMEFRAME = TimeFrame.Hour
DAYS = 360               # ~1 year of hourly bars
MIN_TRAIN = 300         # warmup (bars) before first forecast
REFIT_EVERY = 24         # refit once per day
FEE = 0.0010             # per-trade fee (one-way)
SLIPPAGE = 0.0005        # per-trade slippage (one-way)
TARGET_DAILY_VOL = 0.02  # 2% daily target
BARS_PER_DAY = 24
TARGET_VOL_PER_BAR = TARGET_DAILY_VOL / sqrt(BARS_PER_DAY)
MAX_LEVERAGE = 1.0
SEED = 7
# ----------------------------------------------------

def fetch_bars(symbol=SYMBOL, timeframe=TIMEFRAME, days=DAYS) -> pd.DataFrame:
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    req = CryptoBarsRequest(symbol_or_symbols=symbol, timeframe=timeframe, start=start, end=end)
    bars = data_client.get_crypto_bars(req, feed=CryptoFeed.US).df
    df = bars.xs(symbol).tz_convert("UTC").sort_index()
    return df[["open","high","low","close","volume"]]

def garch_sigma_walkforward(r: pd.Series,
                            min_train: int = MIN_TRAIN,
                            refit_every: int = REFIT_EVERY) -> pd.Series:
    """Walk-forward GARCH(1,1,t) forecast of next-step sigma (no look-ahead)."""
    r = r.dropna().astype(float)
    n = len(r)
    sig_pred = np.full(n, np.nan)
    last_refit = -10**9
    model = None

    for i in range(min_train, n-1):
        rtrain = r.iloc[:i]
        if (i - last_refit) >= refit_every or model is None:
            model = arch_model(rtrain, mean="Zero", vol="GARCH", p=1, q=1, dist="t").fit(disp="off")
            last_refit = i
        sig_next = float(model.forecast(horizon=1).variance.iloc[-1,0]) ** 0.5
        sig_pred[i+1] = sig_next

    return pd.Series(sig_pred, index=r.index, name="sigma_hat")

def backtest(df: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Long-only inverse-vol sizing; shift execution by 1 bar; apply costs on changes."""
    df = df.copy()
    df["ret"] = df["close"].pct_change()
    df = df.dropna()

    sigma_hat = garch_sigma_walkforward(df["ret"])
    df = df.join(sigma_hat)

    frac = np.clip(TARGET_VOL_PER_BAR / (df["sigma_hat"] + 1e-12), 0, MAX_LEVERAGE)
    df["pos_frac_exec"] = frac.shift(1).fillna(0.0)  # execute next bar

    initial_cash = 10_000.0
    cash = initial_cash
    position = 0.0  # BTC amount
    eq = []
    trades = []
    prev_frac = 0.0
    for idx, row in df.iterrows():
        price = row["close"]
        target_frac = row["pos_frac_exec"]
        equity = cash + position * price
        # Determine target notional and position
        target_notional = equity * target_frac
        target_position = target_notional / price
        delta_position = target_position - position
        side = None
        trade_qty = 0.0
        trade_value = 0.0
        cost = 0.0
        # Only trade if position changes
        if abs(delta_position) > 1e-8:
            if delta_position > 0:
                # Buy: check if enough cash
                max_affordable = cash / price
                actual_buy = min(delta_position, max_affordable)
                trade_qty = actual_buy
                trade_value = trade_qty * price
                cost = abs(trade_value) * (FEE + SLIPPAGE)
                cash -= trade_value + cost
                position += trade_qty
                side = "buy"
            else:
                # Sell: can't sell more than you have
                actual_sell = min(-delta_position, position)
                trade_qty = actual_sell
                trade_value = trade_qty * price
                cost = abs(trade_value) * (FEE + SLIPPAGE)
                cash += trade_value - cost
                position -= trade_qty
                side = "sell"
            trade = {
                "time": idx,
                "side": side,
                "qty": trade_qty,
                "price": price,
                "value": trade_value,
                "cost": cost,
                "cash": cash,
                "position": position,
                "equity": cash + position * price
            }
            trades.append(trade)
        # Mark-to-market equity
        eq.append(cash + position * price)
        prev_frac = target_frac
    df["equity"] = eq

    rets = pd.Series(eq, index=df.index).pct_change().dropna()
    sharpe = (rets.mean() / (rets.std() + 1e-12)) * np.sqrt(252 * BARS_PER_DAY) if len(rets)>1 else 0.0
    final_equity = eq[-1] if eq else initial_cash
    cumret = final_equity / initial_cash - 1.0
    dd = (pd.Series(eq).cummax() - eq) / pd.Series(eq).cummax()
    maxdd = float(dd.max()) if len(dd) else 0.0

    summary = {
        "final_equity": round(final_equity, 2),
        "cumulative_return_%": round(cumret * 100, 2),
        "max_drawdown_%": round(maxdd * 100, 2),
        "sharpe_ratio": round(float(sharpe), 2),
        "avg_pos_%": round(float(df["pos_frac_exec"].mean() * 100), 2),
        "num_bars": int(len(df)),
        "min_train": MIN_TRAIN,
        "refit_every": REFIT_EVERY,
    }
    trades_df = pd.DataFrame(trades)
    return summary, df, trades_df

def main():
    np.random.seed(SEED)
    df = fetch_bars()
    summary, dfbt, trades_df = backtest(df)
    print("Summary:", summary)
    dfbt[["close","ret","sigma_hat","pos_frac_exec","equity"]].to_csv("garch_hour_simple_results.csv")
    trades_df.to_csv("garch_hour_trades.csv", index=False)
    print("Saved garch_hour_simple_results.csv and garch_hour_trades.csv")

if __name__ == "__main__":
    main()
