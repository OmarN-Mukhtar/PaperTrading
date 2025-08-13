import os
from dotenv import load_dotenv
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.trading.client import TradingClient

load_dotenv()

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_API_SECRET = os.getenv("ALPACA_API_SECRET")


data_client = CryptoHistoricalDataClient()

trading_client = TradingClient(ALPACA_API_KEY, ALPACA_API_SECRET, paper=True)