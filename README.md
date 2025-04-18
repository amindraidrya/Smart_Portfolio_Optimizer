# Smart Portfolio Optimizer

![Python](https://img.shields.io/badge/python-3.7%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![PyPortfolioOpt](https://img.shields.io/badge/PyPortfolioOpt-1.4.1-green)

A web-based portfolio optimization tool that applies Modern Portfolio Theory (MPT) to help investors find optimal asset allocations based on their risk and return preferences.

## Features

- **Modern Portfolio Theory Implementation**: Uses mean-variance optimization to find efficient portfolios
- **Interactive Interface**: User-friendly Streamlit web interface with sliders for risk/return preferences
- **Visualization**: Plots the efficient frontier with random portfolios and highlights the optimal portfolio
- **Performance Metrics**: Calculates expected return, volatility, and Sharpe ratio
- **Data Integration**: Fetches real-time stock data using Yahoo Finance API
- **Monte Carlo Simulation**: Shows 5,000 random portfolios for comparison

## Usage

1. Run the Streamlit app:
   ```bash
   streamlit run portfolio_optimizer.py
2. In the sidebar:
 - Enter stock tickers (space separated)
 - Adjust risk preference (1-10 scale)
 - Adjust expected return preference (1-10 scale)

3. View the results:

 - Efficient frontier plot with optimal portfolio
 - Optimal asset allocation weights
 - Portfolio performance metrics

## How It Works

1. Data Collection: Fetches 5 years of historical closing prices for the selected tickers
2. Optimization: Uses PyPortfolioOpt to:
 - Calculate expected returns and covariance matrix
 - Find the optimal portfolio based on user preferences
 - Fall back to max Sharpe ratio if target return is not achievable
3. Visualization: Plots the efficient frontier with:
 - Random portfolios from Monte Carlo simulation
 - Individual assets
 - Optimal portfolio
4. Results Display: Shows the optimal weights and performance metrics
