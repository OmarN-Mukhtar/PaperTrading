import os
from dotenv import load_dotenv
import pandas as pd
from datetime import datetime, timedelta, timezone
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import PortfolioHistoryRequest
import matplotlib.pyplot as plt
import json

load_dotenv()

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_API_SECRET = os.getenv("ALPACA_API_SECRET")

trading_client = TradingClient(ALPACA_API_KEY, ALPACA_API_SECRET, paper=True)

# Positions (kept same as before)
positions = trading_client.get_all_positions()
positions_df = pd.DataFrame([{
    "symbol": p.symbol,
    "qty": float(p.qty),
    "market_value": float(p.market_value),
    "unrealized_pl": float(getattr(p, "unrealized_pl", 0) or 0),
    "unrealized_plpc": float(getattr(p, "unrealized_plpc", 0) or 0),
} for p in positions]) if positions else pd.DataFrame(columns=["symbol","qty","market_value","unrealized_pl","unrealized_plpc"])

# Portfolio history (daily for ~3 months)
req = PortfolioHistoryRequest(period="3M", timeframe="1D", extended_hours=True)
hist = trading_client.get_portfolio_history(req)

# Build equity series
eq = pd.Series(hist.equity, index=pd.to_datetime(hist.timestamp, unit="s", utc=True)).sort_index()
eq = eq.astype(float)

# Current and initial equity
equity = float(eq.iloc[-1]) if len(eq) else 0.0
initial_equity = float(eq.iloc[0]) if len(eq) else equity
last_equity = float(eq.iloc[-2]) if len(eq) > 1 else equity

def calc_return(start_equity, end_equity):
    return (end_equity / start_equity - 1) * 100 if start_equity else None

# Helper: equity on or before a given datetime
def equity_on_or_before(ts: pd.Timestamp):
    if len(eq) == 0:
        return None
    sub = eq.loc[:ts]
    return float(sub.iloc[-1]) if len(sub) else None

now_utc = datetime.now(timezone.utc)
one_day_ago = now_utc - timedelta(days=1)
one_week_ago = now_utc - timedelta(days=7)
one_month_ago = now_utc - timedelta(days=30)

eq_1d = equity_on_or_before(pd.Timestamp(one_day_ago))
eq_1w = equity_on_or_before(pd.Timestamp(one_week_ago))
eq_1m = equity_on_or_before(pd.Timestamp(one_month_ago))

returns = {
    "total_return_%": calc_return(initial_equity, equity),
    "return_1d_%": calc_return(eq_1d, equity) if eq_1d is not None else None,
    "return_1w_%": calc_return(eq_1w, equity) if eq_1w is not None else None,
    "return_1m_%": calc_return(eq_1m, equity) if eq_1m is not None else None,
}

# Simple equity bar chart (kept same behavior)
plt.figure(figsize=(8, 4))
plt.bar(["Initial", "Current"], [initial_equity, equity])
plt.title("Account Equity")
plt.ylabel("USD")
plt.tight_layout()
plt.savefig("equity_bar.png")
plt.close()

# Keep output file structure the same
stats = {
    "equity": equity,
    "initial_equity": initial_equity,
    "last_equity": last_equity,
    "returns": returns,
    "positions": positions_df.to_dict(orient="records"),
    "recent_fills": [],  # keeping key for compatibility; portfolio history used for stats
    "last_updated": now_utc.isoformat(),
}

with open("alpaca_stats.json", "w") as f:
    json.dump(stats, f, indent=2)