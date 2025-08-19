import os
from dotenv import load_dotenv
import pandas as pd
from datetime import datetime, timedelta, timezone
from alpaca.trading.client import TradingClient
import matplotlib.pyplot as plt
import json

load_dotenv()

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_API_SECRET = os.getenv("ALPACA_API_SECRET")

trading_client = TradingClient(ALPACA_API_KEY, ALPACA_API_SECRET, paper=True)

# Positions summary
positions = trading_client.get_all_positions()
positions_df = pd.DataFrame([{
    "symbol": p.symbol,
    "qty": float(p.qty),
    "market_value": float(p.market_value),
    "unrealized_pl": float(getattr(p, "unrealized_pl", 0) or 0),
    "unrealized_plpc": float(getattr(p, "unrealized_plpc", 0) or 0),
} for p in positions]) if positions else pd.DataFrame(columns=["symbol","qty","market_value","unrealized_pl","unrealized_plpc"])

# Portfolio history (use default range supported by your SDK version)
hist = trading_client.get_portfolio_history()

# Extract fields robustly (model or dict-like)
def field(obj, name, default=None):
    if hasattr(obj, name):
        return getattr(obj, name)
    if isinstance(obj, dict):
        return obj.get(name, default)
    return default

timestamps = field(hist, "timestamp", []) or []
equities   = field(hist, "equity",   []) or []

# Equity series
ts = pd.to_datetime(timestamps, unit="s", utc=True) if len(timestamps) else pd.to_datetime([], utc=True)
eq = pd.Series(equities, index=ts).sort_index().astype(float) if len(equities) else pd.Series(dtype=float)

equity = float(eq.iloc[-1]) if len(eq) else 0.0
initial_equity = float(100000)
last_equity = float(eq.iloc[-2]) if len(eq) > 1 else equity

def calc_return(start_equity, end_equity):
    return (end_equity / start_equity - 1) * 100 if start_equity else None

def equity_on_or_before(t_utc: pd.Timestamp):
    if len(eq) == 0:
        return None
    sub = eq.loc[:t_utc]
    return float(sub.iloc[-1]) if len(sub) else None

now_utc = datetime.now(timezone.utc)
eq_1d = equity_on_or_before(now_utc - timedelta(days=1))
eq_1w = equity_on_or_before(now_utc - timedelta(days=7))
eq_1m = equity_on_or_before(now_utc - timedelta(days=30))

returns = {
    "total_return_%": calc_return(initial_equity, equity),
    "return_1d_%": calc_return(eq_1d, equity) if eq_1d is not None else None,
    "return_1w_%": calc_return(eq_1w, equity) if eq_1w is not None else None,
    "return_1m_%": calc_return(eq_1m, equity) if eq_1m is not None else None,
}

# Equity bar chart
plt.figure(figsize=(8, 4))
plt.bar(["Initial", "Current"], [initial_equity, equity])
plt.title("Account Equity")
plt.ylabel("USD")
plt.tight_layout()
plt.savefig("equity_bar.png")
plt.close()

# Output JSON (same shape as before)
stats = {
    "equity": equity,
    "initial_equity": initial_equity,
    "last_equity": last_equity,
    "returns": returns,
    "positions": positions_df.to_dict(orient="records"),
    "last_updated": now_utc.isoformat(),
}

with open("alpaca_stats.json", "w") as f:
    json.dump(stats, f, indent=2)

with open("alpaca_stats.html", "w") as f:
    f.write("""<!DOCTYPE html>
<html lang=\"en\">
<head>
    <meta charset=\"UTF-8\">
    <title>Alpaca Account Stats</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 2em; }
        .stats { margin-bottom: 2em; }
        .positions-table, .fills-table { border-collapse: collapse; width: 100%; }
        .positions-table th, .positions-table td, .fills-table th, .fills-table td { border: 1px solid #ccc; padding: 8px; text-align: center; }
        .positions-table th, .fills-table th { background: #f0f0f0; }
    </style>
</head>
<body>
    <h1>Alpaca Account Stats</h1>
    <div class=\"stats\">
        <h2>Returns</h2>
        <ul id=\"returns-list\"></ul>
        <h2>Account Equity</h2>
        <div>Current Equity: <span id=\"equity\"></span> USD</div>
        <div>Initial Equity: <span id=\"initial-equity\"></span> USD</div>
        <div>Last Updated: <span id=\"last-updated\"></span></div>
    </div>
    <h2>Open Positions</h2>
    <table class=\"positions-table\" id=\"positions-table\">
        <thead>
            <tr>
                <th>Symbol</th>
                <th>Quantity</th>
                <th>Market Value</th>
                <th>Unrealized P/L</th>
                <th>Unrealized P/L %</th>
            </tr>
        </thead>
        <tbody></tbody>
    </table>
    <script>
    fetch('alpaca_stats.json')
        .then(response => response.json())
        .then(data => {
            document.getElementById('equity').textContent = (data.equity ?? 0).toFixed(2);
            document.getElementById('initial-equity').textContent = (data.initial_equity ?? 0).toFixed(2);
            document.getElementById('last-updated').textContent = data.last_updated ? new Date(data.last_updated).toLocaleString() : '';
            // Returns
            const returnsList = document.getElementById('returns-list');
            returnsList.innerHTML = '';
            if (data.returns) {
                for (const [key, value] of Object.entries(data.returns)) {
                    const li = document.createElement('li');
                    li.textContent = `${key.replace(/_/g, ' ')}: ${value !== null && value !== undefined ? value.toFixed(2) + '%' : 'N/A'}`;
                    returnsList.appendChild(li);
                }
            }
            // Positions
            const positionsTbody = document.getElementById('positions-table').querySelector('tbody');
            positionsTbody.innerHTML = '';
            if (Array.isArray(data.positions)) {
                data.positions.forEach(pos => {
                    const tr = document.createElement('tr');
                    tr.innerHTML = `
                        <td>${pos.symbol ?? ''}</td>
                        <td>${pos.qty ?? ''}</td>
                        <td>${pos.market_value !== undefined ? parseFloat(pos.market_value).toFixed(2) : ''}</td>
                        <td>${pos.unrealized_pl !== undefined ? parseFloat(pos.unrealized_pl).toFixed(2) : ''}</td>
                        <td>${pos.unrealized_plpc !== undefined ? (parseFloat(pos.unrealized_plpc) * 100).toFixed(2) + '%' : ''}</td>
                    `;
                    positionsTbody.appendChild(tr);
                });
            }
        });
    </script>
</body>
</html>
""")