#!/usr/bin/env python3
"""
live.py

Minute-by-minute LIVE/PAPER trader for BTC/USD on Alpaca using lag-feature
forecasting (DecisionTreeRegressor if available; AR(1) fallback otherwise).
Decision every minute: BUY to go/keep LONG, SELL to go FLAT, else HOLD.

No files written; prints a small decision dictionary each loop.

ENV
----
ALPACA_API_KEY, ALPACA_SECRET_KEY   (or APCA_API_KEY_ID / APCA_API_SECRET_KEY)

Example
-------
python live.py --symbol BTC/USD --timeframe 1Min --window 600 --lags 30 \
    --qty 0.001 --threshold-bps 0.0 --poll-seconds 60 --paper
"""
from __future__ import annotations

import argparse
import os
import time
import signal
from typing import Tuple

import numpy as np
import pandas as pd

# Optional scikit-learn; fallback to statsmodels if unavailable
try:
    from sklearn.tree import DecisionTreeRegressor
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

try:
    import statsmodels.api as sm
    STATSMODELS_AVAILABLE = True
except Exception:
    STATSMODELS_AVAILABLE = False

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest

from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame


def load_keys() -> Tuple[str, str]:
    key = os.getenv("ALPACA_API_KEY") or os.getenv("APCA_API_KEY_ID")
    secret = os.getenv("ALPACA_SECRET_KEY") or os.getenv("APCA_API_SECRET_KEY")
    if not key or not secret:
        raise RuntimeError("Missing ALPACA_API_KEY / ALPACA_SECRET_KEY in environment.")
    return key, secret


def to_timeframe(tf: str) -> TimeFrame:
    mapping = {
        "1Min": TimeFrame.Minute,
        "5Min": TimeFrame(5, "Min"),
        "15Min": TimeFrame(15, "Min"),
        "1Hour": TimeFrame.Hour,
        "1Day": TimeFrame.Day,
    }
    if tf not in mapping:
        raise ValueError(f"Unsupported timeframe: {tf}")
    return mapping[tf]


def fetch_recent(client: CryptoHistoricalDataClient, window: int, tf: TimeFrame) -> pd.Series:
    req = CryptoBarsRequest(
        symbol_or_symbols="BTC/USD",
        timeframe=tf,
        limit=window,
    )
    bars = client.get_crypto_bars(req)
    df = bars.df
    if df is None or df.empty:
        raise RuntimeError("No bars returned from Alpaca.")
    if isinstance(df.index, pd.MultiIndex):
        df = df.xs("BTC/USD", level=1)
    df = df.sort_index()
    return df["close"].astype(float).rename("y")


def make_lag_frame(y: pd.Series, lags: int):
    X = pd.concat([y.shift(i) for i in range(1, lags + 1)], axis=1)
    X.columns = [f"lag_{i}" for i in range(1, lags + 1)]
    Z = pd.concat([X, y], axis=1).dropna()
    return Z.drop(columns=[y.name]), Z[y.name]


class Forecaster:
    def __init__(self, lags: int, random_state: int = 123):
        self.lags = lags
        self.random_state = random_state
        self.use_sklearn = SKLEARN_AVAILABLE
        self.model = None

    def fit(self, y: pd.Series):
        if self.use_sklearn:
            X, yy = make_lag_frame(y, self.lags)
            self.model = DecisionTreeRegressor(random_state=self.random_state)
            self.model.fit(X, yy)
        else:
            if not STATSMODELS_AVAILABLE:
                raise RuntimeError("Neither scikit-learn nor statsmodels is installed.")
            self.model = sm.tsa.ARIMA(y, order=(1, 0, 0)).fit()

    def predict_next(self, y: pd.Series) -> float:
        if self.use_sklearn:
            X, _ = make_lag_frame(y, self.lags)
            return float(self.model.predict(X.iloc[[-1]])[0])
        else:
            return float(self.model.forecast(steps=1).iloc[-1])


def get_position_qty(trading: TradingClient, symbol: str) -> float:
    try:
        pos = trading.get_open_position(symbol)
        return float(pos.qty)
    except Exception:
        return 0.0


def place_order(trading: TradingClient, symbol: str, side: str, qty: float):
    side_enum = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL
    order = MarketOrderRequest(
        symbol=symbol,
        qty=qty,
        side=side_enum,
        time_in_force=TimeInForce.GTC,
    )
    return trading.submit_order(order)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTC/USD")
    ap.add_argument("--timeframe", default="1Min")
    ap.add_argument("--window", type=int, default=600)
    ap.add_argument("--lags", type=int, default=30)
    ap.add_argument("--qty", type=float, default=0.001)
    ap.add_argument("--threshold-bps", type=float, default=0.0)
    ap.add_argument("--poll-seconds", type=int, default=60)
    ap.add_argument("--paper", action="store_true")
    ap.add_argument("--live", dest="paper", action="store_false")
    args = ap.parse_args()

    key, secret = load_keys()
    trading = TradingClient(api_key=key, secret_key=secret, paper=args.paper)
    data = CryptoHistoricalDataClient(api_key=key, secret_key=secret)
    tf = to_timeframe(args.timeframe)

    stop = {"flag": False}
    def handler(sig, frame):
        stop["flag"] = True
        print("Stopping...")
    signal.signal(signal.SIGINT, handler)

    forecaster = Forecaster(lags=args.lags)

    while not stop["flag"]:
        try:
            y = fetch_recent(data, args.window, tf)
            last_ts = y.index[-1]
            last_close = float(y.iloc[-1])

            # Fit & predict
            forecaster.fit(y)
            y_hat = forecaster.predict_next(y)

            # Decide
            up_bps = (y_hat / last_close - 1.0) * 1e4
            want_long = up_bps >= args.threshold_bps

            current_qty = get_position_qty(trading, args.symbol)
            target_qty = args.qty if want_long else 0.0
            delta = target_qty - current_qty

            action = "hold"
            if delta > 1e-12:
                place_order(trading, args.symbol, "buy", round(delta, 6))
                action = "buy"
            elif delta < -1e-12:
                place_order(trading, args.symbol, "sell", round(-delta, 6))
                action = "sell"

            decision = {
                "ts": str(last_ts),
                "symbol": args.symbol,
                "last_close": last_close,
                "y_hat": y_hat,
                "upside_bps": float(up_bps),
                "decision": action,
                "position_qty": float(target_qty if action != "hold" else current_qty),
            }
            print(decision)
        except Exception as e:
            print({"error": str(e)})

        # Sleep
        for _ in range(args.poll_seconds):
            if stop["flag"]:
                break
            time.sleep(1)


if __name__ == "__main__":
    main()
