import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pypfopt import expected_returns, risk_models, EfficientFrontier
from pypfopt import plotting 
import yfinance as yf
import streamlit as st

# Set up Streamlit app
st.title("Smart Portfolio Optimizer")
st.write("""
This tool applies Modern Portfolio Theory to optimize your asset allocation 
based on your risk and return preferences.
""")

# Sidebar for user inputs
st.sidebar.header("User Input Parameters")

# Date range selection
end_date = pd.Timestamp.today().strftime('%Y-%m-%d')
start_date = (pd.Timestamp.today() - pd.Timedelta(days=5*365)).strftime('%Y-%m-%d')

# Default tickers
default_tickers = "AAPL MSFT GOOG AMZN TSLA JPM JNJ PG WMT DIS"
tickers = st.sidebar.text_area("Enter tickers (space separated)", value=default_tickers).split()

# Get risk preference
risk_preference = st.sidebar.slider("Risk Preference (1=Conservative, 10=Aggressive)", 1, 10, 5)

# Get expected return preference
return_preference = st.sidebar.slider("Expected Return Preference (1=Low, 10=High)", 1, 10, 5)

# Convert preferences to actual values
risk_tolerance = risk_preference / 20  # Convert to 0.05 to 0.5 range
target_return = return_preference / 50  # Convert to 0.02 to 0.2 range (2% to 20%)

# Download data function
@st.cache_data
def download_data(tickers, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date)['Close']
    return data

# Main function
def main():
    if len(tickers) < 2:
        st.warning("Please enter at least 2 tickers")
        return
    
    with st.spinner("Downloading data and performing optimization..."):
        try:
            # Download data
            data = download_data(tickers, start_date, end_date)
            
            if data.isnull().values.any():
                st.warning("Some tickers may be invalid or have missing data. Removing problematic tickers.")
                data = data.dropna(axis=1)
                if len(data.columns) < 2:
                    st.error("Not enough valid tickers remaining. Please check your inputs.")
                    return
            
            # Calculate expected returns and covariance matrix
            mu = expected_returns.mean_historical_return(data)
            S = risk_models.sample_cov(data)
            
            # Perform optimization using PyPortfolioOpt
            try:
                # First attempt: Efficient return
                ef_return = EfficientFrontier(mu, S)
                ef_return.add_constraint(lambda w: w >= 0)  # Long-only constraint
                ef_return.efficient_return(target_return)
                weights = ef_return.clean_weights()
                ret, vol, sharpe = ef_return.portfolio_performance()
            except:
                # Fallback: Max Sharpe
                st.warning("Could not find optimal portfolio for given parameters. Using max Sharpe instead.")
                ef_sharpe = EfficientFrontier(mu, S)
                ef_sharpe.add_constraint(lambda w: w >= 0)  # Long-only constraint
                ef_sharpe.max_sharpe()
                weights = ef_sharpe.clean_weights()
                ret, vol, sharpe = ef_sharpe.portfolio_performance()

            # Monte Carlo Simulation
            n_portfolios = 5000
            results = np.zeros((3, n_portfolios))
            for i in range(n_portfolios):
                w = np.random.random(len(mu))
                w /= np.sum(w)
                port_return = np.dot(w, mu)
                port_vol = np.sqrt(np.dot(w.T, np.dot(S, w)))
                results[0,i] = port_return
                results[1,i] = port_vol
                results[2,i] = port_return / port_vol
            
            # Plotting
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot efficient frontier
            ef_plot = EfficientFrontier(mu, S)
            plotting.plot_efficient_frontier(ef_plot, ax=ax, show_assets=True)
            
            # Plot random portfolios
            ax.scatter(results[1,:], results[0,:], c=results[2,:], cmap='YlGnBu', marker='o', s=10, alpha=0.3)
            ax.scatter(vol, ret, marker='*', color='r', s=500, label='Optimal Portfolio')
            
            # Formatting
            ax.set_title('Efficient Frontier with Random Portfolios')
            ax.set_xlabel('Volatility')
            ax.set_ylabel('Expected Return')
            ax.legend()
            plt.colorbar(ax.collections[1], ax=ax, label='Sharpe Ratio')
            
            st.pyplot(fig)
            plt.close()
            
            # Display results
            st.subheader("Optimal Portfolio Allocation")
            weights_df = pd.DataFrame.from_dict(weights, orient='index', columns=['Weight'])
            weights_df['Weight'] = weights_df['Weight'].apply(lambda x: f"{x:.2%}")
            st.dataframe(weights_df)
            
            st.subheader("Portfolio Performance")
            col1, col2, col3 = st.columns(3)
            col1.metric("Expected Return", f"{ret:.2%}")
            col2.metric("Volatility", f"{vol:.2%}")
            col3.metric("Sharpe Ratio", f"{sharpe:.2f}")
            
            # Additional metrics
            st.subheader("Additional Information")
            
            if st.checkbox("Show Covariance Matrix"):
                st.dataframe(S)
                
            if st.checkbox("Show Expected Returns"):
                st.dataframe(pd.DataFrame(mu, columns=['Expected Return']))
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()