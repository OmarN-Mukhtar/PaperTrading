# Alpaca Paper Trading Platform

This project is a Python-based platform for paper trading using the Alpaca API. It features a web interface (built with Streamlit) and includes econometric models such as GARCH and ARMA for financial analysis. All APIs and libraries used are free.

## Features
- Paper trading with Alpaca API
- Web interface (Streamlit)
- Econometric models: GARCH, ARMA, and more
- Ready for deployment on GitHub (Streamlit Community Cloud recommended)

## Setup

1. Clone the repository and create the conda environment:
   ```sh
   conda create -n papertrading python=3.9
   conda activate papertrading
   pip install -r requirements.txt
   ```
2. Set your Alpaca API keys as environment variables or in a `.env` file.
3. Run the app:
   ```sh
   streamlit run app.py
   ```

## Econometric Libraries Used
- `arch` for GARCH
- `statsmodels` for ARMA and other models

## License
This project uses only free APIs and libraries.
