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
import requests
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
MAX_LEVERAGE = 1.0
POLL_SECONDS = 30
MIN_TRADE_USD = 25.0    # skip dust adjustments
SCALE = 100.0
# ----------------------------------------------------


def fetch_bars(symbol: str = SYMBOL, timeframe: TimeFrame = TIMEFRAME, days: int = LOOKBACK_DAYS, retries: int = 3, delay: int = 10) -> pd.DataFrame:
    for attempt in range(retries):
        try:
            end = datetime.now(timezone.utc)
            start = end - timedelta(days=days)
            req = CryptoBarsRequest(symbol_or_symbols=symbol, timeframe=timeframe, start=start, end=end)
            bars = data_client.get_crypto_bars(req, feed=CryptoFeed.US).df
            df = bars.xs(symbol).tz_convert("UTC").sort_index()
            return df[["open","high","low","close","volume"]]
        except requests.exceptions.ReadTimeout:
            print(f"[WARN] Read timeout, retrying ({attempt+1}/{retries})...")
            time.sleep(delay)
        except Exception as e:
            print(f"[ERROR] fetch_bars: {e}")
            time.sleep(delay)
    raise RuntimeError("Failed to fetch bars after retries")


def latest_closed_ts(df: pd.DataFrame) -> pd.Timestamp:
    return df.index[-1]


def forecast_sigma_next(df: pd.DataFrame) -> float:
    """Fit GARCH(1,1,t) on the last MIN_TRAIN returns and forecast next sigma. Return np.nan if insufficient data."""
    r = df["close"].pct_change().dropna()
    if len(r) < MIN_TRAIN + 2:
        return np.nan
    r_train = r.iloc[-MIN_TRAIN:] * SCALE 
    am = arch_model(r_train, mean="Zero", vol="GARCH", p=1, q=1, dist="t", rescale=False).fit(disp="off")
    sigma_next = float(am.forecast(horizon=1).variance.iloc[-1, 0]) ** 0.5
    return sigma_next / SCALE


def get_equity_and_cash() -> tuple[float, float]:
    """
    Return (equity, cash) where `cash` is USD available for crypto.
    For crypto on Alpaca, cash is the correct cap for buys.
    """
    acct = trading_client.get_account()
    equity = float(acct.equity)
    # Prefer explicit cash; fall back to non_marginable_buying_power if present
    cash = float(getattr(acct, "cash", 0.0) or 0.0)
    if cash <= 0 and hasattr(acct, "non_marginable_buying_power"):
        try:
            cash = float(acct.non_marginable_buying_power)
        except Exception:
            pass
    return equity, cash


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

    start_time = time.time()
    max_duration = 180  # 3 minutes in seconds

    while True:
        if time.time() - start_time > max_duration:
            print("[INFO] Max runtime reached, exiting.")
            break
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

            # --- sizing (cash-aware) ---
            frac = float(np.clip(TARGET_VOL_PER_HOUR / max(sigma, 1e-6), 0, MAX_LEVERAGE))

            equity, cash = get_equity_and_cash()
            px = float(df["close"].iloc[-1])
            target_notional = equity * frac

            qty = get_open_qty('BTCUSD')
            current_notional = qty * px
            delta = target_notional - current_notional

            print(f"[DEBUG] ts={ts} sigma={sigma:.6f} frac={frac:.3f} px={px:.2f} "
                f"equity={equity:.2f} cash={cash:.2f} target=${target_notional:.2f} "
                f"current=${current_notional:.2f} delta=${delta:.2f}")

            if abs(delta) < MIN_TRADE_USD:
                print("[ACTION] delta < MIN_TRADE_USD -> HOLD")

            elif delta > 0:
                # BUY: clamp by available cash (leave a few dollars as buffer)
                buy_notional = min(delta, max(cash - 5.0, 0.0))
                if buy_notional < MIN_TRADE_USD:
                    print("[ACTION] insufficient cash -> HOLD")
                else:
                    submit_buy_notional(SYMBOL, buy_notional)

            else:
                # SELL: never sell more than you own
                sell_qty = min(abs(delta) / px, qty)
                if sell_qty * px < MIN_TRADE_USD:
                    print("[ACTION] tiny sell -> HOLD")
                else:
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