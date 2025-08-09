import streamlit as st
import os
from alpaca_trade_api.rest import REST, TimeFrame
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from arch import arch_model
from statsmodels.tsa.arima.model import ARIMA

# Load environment variables
load_dotenv()
ALPACA_API_KEY = os.getenv('ALPACA_API_KEY')
ALPACA_SECRET_KEY = os.getenv('ALPACA_SECRET_KEY')
ALPACA_BASE_URL = 'https://paper-api.alpaca.markets'

# Initialize Alpaca API
api = REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL)

st.title('Alpaca Paper Trading & Econometrics Platform')

symbol = st.text_input('Enter Stock Symbol', 'AAPL')

if st.button('Get Historical Data'):
    barset = api.get_bars(symbol, TimeFrame.Day, limit=100)
    df = pd.DataFrame([bar._raw for bar in barset])
    st.write(df.tail())
    
    st.subheader('ARMA Model')
    returns = df['close'].pct_change().dropna()
    arma_model = ARIMA(returns, order=(2,0,2)).fit()
    st.write(arma_model.summary())
    
    st.subheader('GARCH Model')
    garch_model = arch_model(returns, vol='Garch', p=1, q=1)
    garch_fit = garch_model.fit(disp='off')
    st.write(garch_fit.summary())

st.info('Set your Alpaca API keys in a .env file or as environment variables.')
