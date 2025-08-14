#!/usr/bin/env python3
"""
backtest.py

Minute-by-minute backtester for BTC/USD using Alpaca Crypto API.
Implements a simple scikit-learn style forecasting approach with LAG FEATURES
and walk-forward refits (expanding window, steps=1), per the ideas discussed in
"A Practical Guide on Scikit-learn for Time Series Forecasting":
- Use lagged observations as predictors (tabular ML with scikit-learn).
- Walk-forward / backtesting with refits and increasing train size.

No files written. Prints a single summary dictionary at the end.

ENV
----
ALPACA_API_KEY, ALPACA_SECRET_KEY   (or APCA_API_KEY_ID / APCA_API_SECRET_KEY)

Example
-------
python backtest.py --start 2024-06-01 --end 2024-06-07 --initial-equity 10000 \
    --fee-bps 1.0 --lags 30 --min-train 500 --threshold-bps 0.0
"""
from __future__ import annotations

import argparse
import os
import math
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

from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame


def load_keys() -> Tuple[str, str]:
    key = os.getenv("ALPACA_API_KEY") or os.getenv("APCA_API_KEY_ID")
    secret = os.getenv("ALPACA_SECRET_KEY") or os.getenv("APCA_API_SECRET_KEY")
    if not key or not secret:
        raise RuntimeError("Missing ALPACA_API_KEY / ALPACA_SECRET_KEY in environment.")
    return key, secret


def fetch_bars(start: str, end: str, key: str, secret: str) -> pd.Series:
    client = CryptoHistoricalDataClient(api_key=key, secret_key=secret)
    req = CryptoBarsRequest(
        symbol_or_symbols="BTC/USD",
        timeframe=TimeFrame.Minute,
        start=start,
        end=end,
        limit=10000,
    )
    bars = client.get_crypto_bars(req)
    df = bars.df
    if df is None or df.empty:
        raise RuntimeError("No bars returned from Alpaca.")
    if isinstance(df.index, pd.MultiIndex):
        df = df.xs("BTC/USD", level=1)
    df = df.sort_index()
    y = df["close"].astype(float).rename("y")
    return y


def make_lag_frame(y: pd.Series, lags: int) -> Tuple[pd.DataFrame, pd.Series]:
    X = pd.concat([y.shift(i) for i in range(1, lags + 1)], axis=1)
    X.columns = [f"lag_{i}" for i in range(1, lags + 1)]
    Z = pd.concat([X, y], axis=1).dropna()
    return Z.drop(columns=[y.name]), Z[y.name]


class Forecaster:
    """Lag-feature forecaster using sklearn if present, else AR(1) fallback."""
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


def backtest(
    start: str,
    end: str,
    initial_equity: float = 10_000.0,
    fee_bps: float = 1.0,
    lags: int = 30,
    min_train: int = 500,
    threshold_bps: float = 0.0,
) -> dict:
    key, secret = load_keys()
    y = fetch_bars(start, end, key, secret).copy()

    # Ensure enough data for lags + min_train
    start_idx = max(lags + 1, min_train)
    if len(y) <= start_idx:
        raise RuntimeError(f"Not enough data ({len(y)}) for lags={lags} and min_train={min_train}.")

    # Precompute returns
    ret = y.pct_change().fillna(0.0)

    forecaster = Forecaster(lags=lags)
    equity = initial_equity
    pos = 0  # 0 = flat, 1 = long
    num_trades = 0
    strat_rets = []

    # Walk-forward with refit and increasing train size, forecasting 1 step ahead
    for t in range(start_idx, len(y) - 1):
        y_train = y.iloc[:t]  # up to t-1 inclusive
        last_close = float(y.iloc[t - 1])

        forecaster.fit(y_train)
        y_hat = forecaster.predict_next(y_train)

        # Decision rule: go long if forecast above last close by threshold_bps
        up = (y_hat / last_close - 1.0) * 1e4  # in bps
        want_pos = 1 if up >= threshold_bps else 0

        # Trading cost if position changes at time t
        if want_pos != pos:
            num_trades += 1
            trade_cost = (fee_bps / 1e4) * equity
        else:
            trade_cost = 0.0

        # Realized return from t -> t+1 using position at t
        r = float(ret.iloc[t + 1]) if (t + 1) < len(ret) else 0.0
        strat_ret = pos * r  # daily/minute return in fraction
        strat_ret -= trade_cost / equity if equity > 0 else 0.0

        equity *= (1.0 + strat_ret)
        strat_rets.append(strat_ret)
        pos = want_pos

    # Metrics
    eq_series = pd.Series(np.cumprod(1.0 + np.array(strat_rets)) * initial_equity)
    final_equity = float(eq_series.iloc[-1]) if len(eq_series) else initial_equity
    cum_return_pct = (final_equity / initial_equity - 1.0) * 100.0

    running_max = eq_series.cummax()
    dd = (eq_series / running_max) - 1.0
    max_dd_pct = float(dd.min() * 100.0) if len(dd) else 0.0

    mu = np.mean(strat_rets) if strat_rets else 0.0
    sig = np.std(strat_rets) if strat_rets else 0.0
    if sig > 0:
        sharpe = (mu / sig) * math.sqrt(365 * 24 * 60)  # minute annualization
    else:
        sharpe = 0.0

    return {
        "final_equity": round(final_equity, 6),
        "cumulative_return_%": round(cum_return_pct, 6),
        "max_drawdown_%": round(max_dd_pct, 6),
        "num_trades": int(num_trades),
        "sharpe_ratio": round(float(sharpe), 6),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True, help="YYYY-MM-DD or RFC3339")
    ap.add_argument("--end", required=True, help="YYYY-MM-DD or RFC3339")
    ap.add_argument("--initial-equity", type=float, default=10_000.0)
    ap.add_argument("--fee-bps", type=float, default=1.0)
    ap.add_argument("--lags", type=int, default=30)
    ap.add_argument("--min-train", type=int, default=500, help="Minimum training samples before first decision.")
    ap.add_argument("--threshold-bps", type=float, default=0.0, help="Forecast vs last close threshold to go long.")
    args = ap.parse_args()

    res = backtest(
        start=args.start,
        end=args.end,
        initial_equity=args.initial_equity,
        fee_bps=args.fee_bps,
        lags=args.lags,
        min_train=args.min_train,
        threshold_bps=args.threshold_bps,
    )
    print(res)


if __name__ == "__main__":
    main()
