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

# Fetch all positions and equity
account = trading_client.get_account()
positions = trading_client.get_all_positions()

# Get equity and important stats
equity = float(account.equity)
last_equity = float(account.last_equity)
initial_equity = float(account.last_equity)  # fallback if no history

# Fetch account activities for returns
activities = trading_client.get_activities()
activity_df = pd.DataFrame([a.__dict__ for a in activities])
activity_df['date'] = pd.to_datetime(activity_df['transaction_time'], errors='coerce')
activity_df = activity_df.sort_values('date')

# Calculate returns for different periods
def calc_return(start_equity, end_equity):
    return (end_equity / start_equity - 1) * 100

today = datetime.now(timezone.utc).date()
week_ago = today - timedelta(days=7)
month_ago = today - timedelta(days=30)

# Fallback: use last_equity for previous periods if no data
returns = {}
returns['total_return_%'] = calc_return(initial_equity, equity)

# Calculate returns for last day, week, month
for label, since in [('1d', today - timedelta(days=1)), ('1w', week_ago), ('1m', month_ago)]:
    mask = activity_df['date'].dt.date >= since
    if mask.any():
        start = activity_df.loc[mask, 'net_amount'].cumsum().iloc[0] if 'net_amount' in activity_df else initial_equity
        end = activity_df.loc[mask, 'net_amount'].cumsum().iloc[-1] if 'net_amount' in activity_df else equity
        returns[f'return_{label}_%'] = calc_return(start, end)
    else:
        returns[f'return_{label}_%'] = None

# Prepare positions summary
positions_df = pd.DataFrame([p.__dict__ for p in positions])
positions_summary = positions_df[['symbol', 'qty', 'market_value', 'unrealized_pl', 'unrealized_plpc']].to_dict(orient='records')

# Visualize equity (dummy example, as live equity history may not be available)
plt.figure(figsize=(8,4))
plt.bar(['Initial', 'Current'], [initial_equity, equity], color=['gray', 'green'])
plt.title('Account Equity')
plt.ylabel('USD')
plt.savefig('equity_bar.png')
plt.close()

# Save stats to JSON for GitHub Pages
stats = {
    'equity': equity,
    'initial_equity': initial_equity,
    'returns': returns,
    'positions': positions_summary,
    'last_updated': datetime.now(timezone.utc).isoformat()
}

with open('alpaca_stats.json', 'w') as f:
    json.dump(stats, f, indent=2)

print("Stats and chart saved. You can use 'alpaca_stats.json' and 'equity_bar.png' for your GitHub Pages site.")
