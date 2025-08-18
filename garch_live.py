# live_garch_hour_simple.py
"""
Live trading: BTC/USD hourly, long-only, inverse-vol sizing from GARCH(1,1) (Student-t) on returns.

Logic per bar (on bar close):
  1) Fetch history (hourly) from Alpaca.
  2) Fit GARCH(1,1,t) on last MIN_TRAIN returns.
  3) Forecast next sigma (σ̂). Compute target fraction f = clip(target_vol_per_hour / σ̂, 0, max_leverage).
  4) Size desired notional = account equity * f (capped by buying power).
  5) Trade the difference: buy notional if desired > current; sell qty if desired < current.
  6) Repeat on each new closed hour bar.

Assumptions:
- `config.py` provides authenticated `data_client` (alpaca.data) and `trading_client` (alpaca.trading).
- Your Alpaca account can trade crypto with symbol "BTC/USD".
- Market orders are acceptable for this simple example.
"""

from __future__ import annotations
from math import sqrt
from datetime import datetime, timedelta, timezone
import time
import numpy as np
import pandas as pd

from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import CryptoFeed
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest

from arch import arch_model

from config import data_client, trading_client


# ---------------- Settings (minimal) ----------------
SYMBOL = "BTC/USD"
TIMEFRAME = TimeFrame.Hour
LOOKBACK_DAYS = 360
MIN_TRAIN = 1000        # ~41 days of hourly bars
FEE = 0.0010            # not used in live sizing, but useful for logs
SLIPPAGE = 0.0005
TARGET_DAILY_VOL = 0.02 # 2% daily
BARS_PER_DAY = 24
TARGET_VOL_PER_HOUR = TARGET_DAILY_VOL / sqrt(BARS_PER_DAY)
MAX_LEVERAGE = 0.1
POLL_SECONDS = 30
MIN_TRADE_USD = 25.0    # skip dust adjustments
# ----------------------------------------------------


def fetch_bars(symbol: str = SYMBOL, timeframe: TimeFrame = TIMEFRAME, days: int = LOOKBACK_DAYS) -> pd.DataFrame:
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    req = CryptoBarsRequest(symbol_or_symbols=symbol, timeframe=timeframe, start=start, end=end)
    bars = data_client.get_crypto_bars(req, feed=CryptoFeed.US).df
    df = bars.xs(symbol).tz_convert("UTC").sort_index()
    return df[["open","high","low","close","volume"]]


def latest_closed_ts(df: pd.DataFrame) -> pd.Timestamp:
    return df.index[-1]


def forecast_sigma_next(df: pd.DataFrame) -> float:
    """Fit GARCH(1,1,t) on the last MIN_TRAIN returns and forecast next sigma. Return np.nan if insufficient data."""
    r = df["close"].pct_change().dropna()
    if len(r) < MIN_TRAIN + 2:
        return np.nan
    r_train = r.iloc[-MIN_TRAIN:]
    am = arch_model(r_train, mean="Zero", vol="GARCH", p=1, q=1, dist="t").fit(disp="off")
    sigma_next = float(am.forecast(horizon=1).variance.iloc[-1, 0]) ** 0.5
    return sigma_next


def get_account_equity_and_bp() -> tuple[float, float]:
    acct = trading_client.get_account()
    equity = float(acct.equity)
    buying_power = float(getattr(acct, "buying_power", equity))  # crypto may reflect cash balance
    return equity, buying_power


def get_open_qty(symbol: str) -> float:
    try:
        for p in trading_client.get_all_positions():
            if p.symbol == symbol:
                return float(p.qty)
    except Exception as e:
        print(f"[WARN] positions: {e}")
    return 0.0


def submit_buy_notional(symbol: str, notional: float):
    try:
        order = MarketOrderRequest(symbol=symbol, notional=round(notional, 2), side=OrderSide.BUY, time_in_force=TimeInForce.GTC)
        resp = trading_client.submit_order(order)
        print(f"[BUY] ${notional:.2f} submitted. id={resp.id}")
    except Exception as e:
        print(f"[ERROR] BUY failed: {e}")


def submit_sell_qty(symbol: str, qty: float):
    if qty <= 0: 
        return
    try:
        order = MarketOrderRequest(symbol=symbol, qty=str(qty), side=OrderSide.SELL, time_in_force=TimeInForce.GTC)
        resp = trading_client.submit_order(order)
        print(f"[SELL] {qty:.8f} {symbol.split('/')[0]} submitted. id={resp.id}")
    except Exception as e:
        print(f"[ERROR] SELL failed: {e}")


def run_live():
    print(f"[START] Live hourly GARCH(1,1) sizing for {SYMBOL}")
    df = fetch_bars()
    last_ts = latest_closed_ts(df)
    print(f"[INIT] Last closed bar: {last_ts}")

    while True:
        try:
            # Wait for a new closed hourly bar
            time.sleep(POLL_SECONDS)
            df = fetch_bars()
            ts = latest_closed_ts(df)
            if ts <= last_ts:
                continue  # no new bar yet

            # New bar closed -> compute sigma for next hour and target size
            sigma = forecast_sigma_next(df)
            if np.isnan(sigma) or sigma <= 0:
                print(f"[INFO] sigma not ready (len<{MIN_TRAIN}) or invalid. HOLD.")
                last_ts = ts
                continue

            frac = np.clip(TARGET_VOL_PER_HOUR / (sigma + 1e-12), 0, MAX_LEVERAGE)

            equity, buying_power = get_account_equity_and_bp()
            px = float(df["close"].iloc[-1])
            desired_notional = min(equity * frac, buying_power * 0.98)  # safety margin
            qty = get_open_qty(SYMBOL)
            current_notional = qty * px

            delta = desired_notional - current_notional

            print(f"[DEBUG] ts={ts} sigma={sigma:.6f} frac={frac:.3f} price={px:.2f} "
                  f"equity={equity:.2f} desired=${desired_notional:.2f} current=${current_notional:.2f} "
                  f"delta=${delta:.2f}")

            if abs(delta) < MIN_TRADE_USD:
                print("[ACTION] delta below MIN_TRADE_USD -> HOLD")
            elif delta > 0:
                submit_buy_notional(SYMBOL, delta)
            else:
                sell_qty = abs(delta) / px
                submit_sell_qty(SYMBOL, sell_qty)

            last_ts = ts

        except KeyboardInterrupt:
            print("[STOP] interrupted by user")
            break
        except Exception as e:
            print(f"[ERROR] {e}")
            time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    run_live()
