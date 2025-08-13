from dotenv import load_dotenv
import os
import pandas as pd
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetAssetsRequest
from alpaca.trading.enums import AssetClass

load_dotenv()

alpaca_key = os.environ.get("ALPACA_API_KEY")
alpaca_secret = os.environ.get("ALPACA_SECRET_KEY")

trading_client = TradingClient(alpaca_key, alpaca_secret, paper=True)

class Util:
    @staticmethod
    def to_dataframe(data):
        if isinstance(data, list):
            return pd.DataFrame([item.__dict__ for item in data])
        return pd.DataFrame(data, columns=['tag', 'value']).set_index('tag')


positions = trading_client.get_all_positions()
positions_df = Util.to_dataframe(positions)

search_params = GetAssetsRequest(asset_class=AssetClass.CRYPTO, status='active')
assets = trading_client.get_all_assets(search_params)
assets_df = Util.to_dataframe(assets)


