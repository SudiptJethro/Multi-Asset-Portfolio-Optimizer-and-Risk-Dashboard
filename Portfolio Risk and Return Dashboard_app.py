# Footer inport
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Portfolio Risk and Return Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    /* Make st.metric cards theme-aware */
    div[data-testid="stMetric"] {
        background-color: rgba(240, 248, 255, 0.05); /* light transparent bg */
        padding: 1rem;
        border-radius: 0.75rem;
        border-left: 4px solid #1f77b4;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    /* Optional: force label and value to be readable in dark mode */
    div[data-testid="stMetric"] > label, 
    div[data-testid="stMetric"] > div {
        color: inherit;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">📊 Portfolio Risk and Multi-Asset Allocation Dashboard</h1>', unsafe_allow_html=True)

# LinkedIn link
st.sidebar.markdown(
    """
    <a href="https://www.linkedin.com/in/sudipt-jethro-4ab707241/" target="_blank">
        <img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="25"/>
        Sudipt Jethro
    </a>
    """,
    unsafe_allow_html=True
)

# Sidebar for inputs
st.sidebar.header("Portfolio Configuration")

st.sidebar.markdown("""
**✅ Supported Ticker Examples**  
- **Stocks**: `AAPL`, `MSFT`, `GOOGL`                      
- **Commodities**: `GLD` (Gold), `USO` (Oil), `SLV` (Silver)  
- **Bonds**: `TLT` (Long-Term Treasury), `LQD` (Investment Grade)  
- **ETFs**: `SPY` (S&P 500), `QQQ` (Nasdaq), `HYG` (High Yield Bonds)  
- **Crypto**: `BTC-USD`, `ETH-USD`
""")

# Time period selection
period_options = {
        "1 Year": "1y",
        "2 Years": "2y",
        "5 Years": "5y",
        "Max": "max"
    }

date_mode = st.sidebar.radio("Select Time Range Mode", ["Custom Dates", "Predefined Period"])

if date_mode == "Predefined Period":
    selected_period = st.sidebar.selectbox("Select Time Period", list(period_options.keys()))
    start_date = None
    end_date = None
else:
    start_date = st.date_input("Start date", value=datetime(2023, 1, 1))
    end_date = datetime.today().strftime('%Y-%m-%d')
    selected_period = None

# Risk-free rate input
risk_free_rate = st.sidebar.number_input("Risk-Free Rate (%)", min_value=0.0, max_value=10.0, value=2.0, step=0.1) / 100

# Number of assets
num_assets = st.sidebar.number_input("Number of Assets", min_value=2, max_value=10, value=3, step=1)

# Function to Detect asset type
def get_ticker_info(tickers):
    results = {}

    for ticker in tickers:
        ticker_upper = ticker.upper()

        try:
            info = yf.Ticker(ticker).info
            name = info.get("shortName", "N/A")
            quote_type = info.get("quoteType", "").lower()
            sector = info.get("sector", "").lower()
            category = info.get("category", "").lower()
            fund_family = info.get("fundFamily", "").lower()
            name_lower = name.lower()
        except Exception:
            info = {}
            name = "N/A"
            quote_type = ""
            sector = ""
            category = ""
            fund_family = ""
            name_lower = ""
        
        # 🔍 Asset Type Detection
        if ticker_upper.endswith("-USD"):  # e.g., BTC-USD
            asset_type = "Crypto"

        elif any(x in name_lower for x in ["bond", "treasury", "fixed income", "yield", "debt"]) \
             or "bond" in category or sector == "fixed income":
            asset_type = "Bond"

        elif "etf" in quote_type or "etf" in category or "etf" in name_lower or "etf" in fund_family:
            asset_type = "ETF"

        elif any(x in name_lower for x in ["gold", "silver", "oil", "natural gas", "commodity", "precious metal", "energy"]) \
             or "commodity" in category:
            asset_type = "Commodity"

        elif quote_type in ["equity", "stock"]:
            asset_type = "Stock"

        else:
            asset_type = "Unknown"

        # 🧠 Fallback name for crypto if missing
        if name == "N/A" and ticker_upper.endswith("-USD"):
            name = ticker_upper.replace("-USD", "").capitalize() + " (Crypto)"

        results[ticker] = {
            "name": name,
            "type": asset_type
        }

    return results

# Dynamic input for tickers and weights
tickers = []
weights = []

st.sidebar.subheader("Asset Allocation")
for i in range(num_assets):
    col1, col2 = st.sidebar.columns([2, 1])
    with col1:
        ticker = st.text_input(f"Ticker {i+1}", value=f"{'AAPL' if i==0 else 'GOOGL' if i==1 else 'MSFT' if i==2 else ''}", key=f"ticker_{i}")
    with col2:
        weight = st.number_input(f"Weight {i+1}", min_value=0.0, max_value=1.0, value=1.0/num_assets, step=0.01, key=f"weight_{i}")
    
    if ticker:
        tickers.append(ticker.upper())
        weights.append(weight)

# Get name + asset type for all tickers
ticker_info = get_ticker_info(tickers)

# Display in sidebar
st.sidebar.subheader("🧾 Ticker Details")
for ticker in tickers:
     st.sidebar.write(f"**{ticker}** — {ticker_info[ticker]['name']}  \n*Type:* {ticker_info[ticker]['type']}")

# Normalize weights
if weights and sum(weights) > 0:
    weights = [w/sum(weights) for w in weights]
    
# Display normalized weights
if len(weights) == len(tickers):
    st.sidebar.subheader("Normalized Weights")
    for i, (ticker, weight) in enumerate(zip(tickers, weights)):
        st.sidebar.write(f"{ticker}: {weight:.2%}")

# Function to fetch data
@st.cache_data
def fetch_data(tickers, period=None, start=None, end=None):
    try:
        if period:
            raw = yf.download(tickers, period=period, auto_adjust=False)
        else:
            raw = yf.download(tickers, start=start, end=end, auto_adjust=False)
        
        # Ensure 'Adj Close' exists
        if 'Adj Close' not in raw.columns:
            st.error("❌ 'Adj Close' data not found. Please check the tickers or selected date range.")
            return None
        
        adj_close = raw['Adj Close']

        # If only one ticker is returned, it may come as a Series
        if isinstance(adj_close, pd.Series):
            adj_close = adj_close.to_frame()
            adj_close.columns = tickers

        # Drop rows with any missing data
        adj_close = adj_close.dropna()

        if adj_close.empty:
            st.error("❌ Fetched data is empty after cleaning. Try different tickers or a wider date range.")
            return None

        return adj_close
    
    except Exception as e:
        st.error(f"❌ Error fetching data: {str(e)}")
        return None

# Function to calculate portfolio metrics
def calculate_portfolio_metrics(returns, weights, risk_free_rate):
    portfolio_returns = (returns * weights).sum(axis=1)

    #Basic metrics
    annual_return = portfolio_returns.mean() * 252
    annual_volatility = portfolio_returns.std() * np.sqrt(252)
    sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility
    
    # Drawdown calculation
    cumulative_returns = (1 + portfolio_returns).cumprod()
    rolling_max = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - rolling_max) / rolling_max
    max_drawdown = drawdown.min()
    
    # VaR and CVaR (95% confidence)
    var_95 = np.percentile(portfolio_returns, 5)
    cvar_95 = portfolio_returns[portfolio_returns <= var_95].mean()
    
    return {
        'Annual Return': annual_return,
        'Annual Volatility': annual_volatility,
        'Sharpe Ratio': sharpe_ratio,
        'Max Drawdown': max_drawdown,
        'VaR (95%)': var_95,
        'CVaR (95%)': cvar_95,
        'Portfolio Returns': portfolio_returns,
        'Cumulative Returns': cumulative_returns,
        'Drawdown': drawdown
    }

# Function to calculate benchmark metrics
def calculate_benchmark_metrics(benchmark_returns, risk_free_rate):
    # Basic metrics
    annual_return = benchmark_returns.mean() * 252
    annual_volatility = benchmark_returns.std() * np.sqrt(252)
    sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility
    
    # Drawdown calculation
    cumulative_returns = (1 + benchmark_returns).cumprod()
    rolling_max = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - rolling_max) / rolling_max
    max_drawdown = drawdown.min()
    
    return {
        'Annual Return': annual_return,
        'Annual Volatility': annual_volatility,
        'Sharpe Ratio': sharpe_ratio,
        'Max Drawdown': max_drawdown,
        'Benchmark Returns': benchmark_returns,
        'Cumulative Returns': cumulative_returns,
        'Drawdown': drawdown
    }

# Function to calculate relative performance metrics
def calculate_relative_metrics(portfolio_returns, benchmark_returns, risk_free_rate):
    # Alpha and Beta calculation
    excess_portfolio = portfolio_returns - risk_free_rate / 252
    excess_benchmark = benchmark_returns - risk_free_rate / 252
    
    # Beta (portfolio sensitivity to benchmark)
    covariance = np.cov(excess_portfolio, excess_benchmark)[0, 1]
    benchmark_variance = np.var(excess_benchmark)
    beta = covariance / benchmark_variance if benchmark_variance != 0 else 0
    
    # Alpha (excess return over CAPM expected return)
    portfolio_annual_return = portfolio_returns.mean() * 252
    benchmark_annual_return = benchmark_returns.mean() * 252
    alpha = portfolio_annual_return - (risk_free_rate + beta * (benchmark_annual_return - risk_free_rate))
    
    # Information Ratio (active return / tracking error)
    active_returns = portfolio_returns - benchmark_returns
    tracking_error = active_returns.std() * np.sqrt(252)
    information_ratio = (active_returns.mean() * 252) / tracking_error if tracking_error != 0 else 0
    
    # Correlation
    correlation = np.corrcoef(portfolio_returns, benchmark_returns)[0, 1]
    
    return {
        'Alpha': alpha,
        'Beta': beta,
        'Information Ratio': information_ratio,
        'Correlation': correlation,
        'Tracking Error': tracking_error,
        'Active Returns': active_returns
    }

# Function to calculate correlation matrix
def calculate_correlation_matrix(returns):
    return returns.corr()

# Main analysis
if st.sidebar.button("Analyze Portfolio", type="primary"):
    if len(tickers) >= 2 and len(weights) == len(tickers):
        with st.spinner("Fetching data and analyzing portfolio..."):

            # Fetch portfolio and benchmark data
            if date_mode == "Predefined Period" and selected_period in period_options:
             period_value = period_options[selected_period]
             data = fetch_data(tickers, period=period_value)
             benchmark_data = fetch_data(['^GSPC'], period=period_value)
            elif date_mode == "Custom Dates" and start_date and end_date:
             data = fetch_data(tickers, start=start_date, end=end_date)
             benchmark_data = fetch_data(['^GSPC'], start=start_date, end=end_date)
            else:
             st.error("Please select a valid time range before running the analysis.")
             st.stop()  # ✅ ONLY stop if the else block run
            
            if data is not None and not data.empty and benchmark_data is not None and not benchmark_data.empty:
                # Calculate returns
                returns = data.pct_change().dropna()
                benchmark_returns = benchmark_data.pct_change().dropna().iloc[:, 0]
                
                # Align dates between portfolio and benchmark
                common_dates = returns.index.intersection(benchmark_returns.index)
                returns = returns.loc[common_dates]
                benchmark_returns = benchmark_returns.loc[common_dates]
                
                # Calculate portfolio metrics
                portfolio_metrics = calculate_portfolio_metrics(returns, weights, risk_free_rate)
                benchmark_metrics = calculate_benchmark_metrics(benchmark_returns, risk_free_rate)
                relative_metrics = calculate_relative_metrics(
                    portfolio_metrics['Portfolio Returns'], 
                    benchmark_returns, 
                    risk_free_rate
                )
                
                # Display key metrics comparison
                st.subheader("📈 Portfolio vs S&P 500 Performance")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown("**Portfolio Metrics**")
                    st.metric("Annual Return", f"{portfolio_metrics['Annual Return']:.2%}")
                    st.metric("Annual Volatility", f"{portfolio_metrics['Annual Volatility']:.2%}")
                    st.metric("Sharpe Ratio", f"{portfolio_metrics['Sharpe Ratio']:.3f}")
                    st.metric("Max Drawdown", f"{portfolio_metrics['Max Drawdown']:.2%}")
                
                with col2:
                    st.markdown("**S&P 500 Benchmark**")
                    st.metric("Annual Return", f"{benchmark_metrics['Annual Return']:.2%}")
                    st.metric("Annual Volatility", f"{benchmark_metrics['Annual Volatility']:.2%}")
                    st.metric("Sharpe Ratio", f"{benchmark_metrics['Sharpe Ratio']:.3f}")
                    st.metric("Max Drawdown", f"{benchmark_metrics['Max Drawdown']:.2%}")
                
                with col3:
                    st.markdown("**Relative Performance**")
                    return_diff = portfolio_metrics['Annual Return'] - benchmark_metrics['Annual Return']
                    st.metric("Excess Return", f"{return_diff:.2%}", delta=return_diff)
                    st.metric("Alpha", f"{relative_metrics['Alpha']:.2%}")
                    st.metric("Beta", f"{relative_metrics['Beta']:.3f}")
                    st.metric("Information Ratio", f"{relative_metrics['Information Ratio']:.3f}")
                
                # Additional relative metrics
                col4, col5 = st.columns(2)
                with col4:
                    st.metric("Correlation with S&P 500", f"{relative_metrics['Correlation']:.3f}")
                with col5:
                    st.metric("Tracking Error", f"{relative_metrics['Tracking Error']:.2%}")

                # Portfolio composition pie chart
                st.subheader("🥧 Portfolio Composition")
                fig_pie = px.pie(values=weights, names=tickers, title="Asset Allocation")
                fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig_pie, use_container_width=True)

                # Performance comparison charts
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📊 Cumulative Returns Comparison")
                    fig_cum = go.Figure()
                    
                    # Portfolio cumulative returns
                    fig_cum.add_trace(go.Scatter(
                        x=portfolio_metrics['Cumulative Returns'].index,
                        y=(portfolio_metrics['Cumulative Returns'] - 1) * 100,
                        mode='lines',
                        name='Portfolio',
                        line=dict(color='blue', width=2)
                    ))
                    
                    # S&P 500 cumulative returns
                    fig_cum.add_trace(go.Scatter(
                        x=benchmark_metrics['Cumulative Returns'].index,
                        y=(benchmark_metrics['Cumulative Returns'] - 1) * 100,
                        mode='lines',
                        name='S&P 500',
                        line=dict(color='red', width=2)
                    ))
                    
                    fig_cum.update_layout(
                        title="Portfolio vs S&P 500 Cumulative Returns (%)",
                        xaxis_title="Date",
                        yaxis_title="Cumulative Return (%)",
                        hovermode='x unified'
                    )
                    st.plotly_chart(fig_cum, use_container_width=True)
                
                with col2:
                    st.subheader("📉 Drawdown Comparison")
                    fig_dd = go.Figure()
                    
                    # Portfolio drawdown
                    fig_dd.add_trace(go.Scatter(
                        x=portfolio_metrics['Drawdown'].index,
                        y=portfolio_metrics['Drawdown'] * 100,
                        mode='lines',
                        name='Portfolio',
                        line=dict(color='blue', width=2)
                    ))
                    
                    # S&P 500 drawdown
                    fig_dd.add_trace(go.Scatter(
                        x=benchmark_metrics['Drawdown'].index,
                        y=benchmark_metrics['Drawdown'] * 100,
                        mode='lines',
                        name='S&P 500',
                        line=dict(color='red', width=2)
                    ))
                    
                    fig_dd.update_layout(
                        title="Portfolio vs S&P 500 Drawdown (%)",
                        xaxis_title="Date",
                        yaxis_title="Drawdown (%)",
                        hovermode='x unified'
                    )
                    st.plotly_chart(fig_dd, use_container_width=True)
                
                # Active returns (Portfolio - S&P 500)
                st.subheader("🎯 Active Returns Analysis")
                fig_active = go.Figure()
                
                cumulative_active = (1 + relative_metrics['Active Returns']).cumprod()
                fig_active.add_trace(go.Scatter(
                    x=cumulative_active.index,
                    y=(cumulative_active - 1) * 100,
                    mode='lines',
                    name='Active Returns',
                    line=dict(color='green', width=2),
                    fill='tonexty'
                ))
                
                fig_active.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)
                fig_active.update_layout(
                    title="Cumulative Active Returns (Portfolio - S&P 500)",
                    xaxis_title="Date",
                    yaxis_title="Active Return (%)",
                    hovermode='x unified'
                )
                st.plotly_chart(fig_active, use_container_width=True)
                
                # Individual asset performance
                st.subheader("🔍 Individual Asset Performance")
                individual_returns = returns.mean() * 252
                individual_volatility = returns.std() * np.sqrt(252)
                individual_sharpe = (individual_returns - risk_free_rate) / individual_volatility
                
                performance_df = pd.DataFrame({
                    'Asset': tickers,
                    'Weight': [f"{w:.2%}" for w in weights],
                    'Annual Return': [f"{r:.2%}" for r in individual_returns],
                    'Annual Volatility': [f"{v:.2%}" for v in individual_volatility],
                    'Sharpe Ratio': [f"{s:.3f}" for s in individual_sharpe]
                })
                
                # Add S&P 500 benchmark row
                sp500_row = pd.DataFrame({
                    'Asset': ['S&P 500'],
                    'Weight': ['Benchmark'],
                    'Annual Return': [f"{benchmark_metrics['Annual Return']:.2%}"],
                    'Annual Volatility': [f"{benchmark_metrics['Annual Volatility']:.2%}"],
                    'Sharpe Ratio': [f"{benchmark_metrics['Sharpe Ratio']:.3f}"]
                })
                
                performance_df = pd.concat([performance_df, sp500_row], ignore_index=True)
                st.dataframe(performance_df, use_container_width=True)
                
                # Risk-Return scatter plot with benchmark
                st.subheader("🎯 Risk-Return Profile vs S&P 500")
                fig_scatter = go.Figure()
                
                # Add individual assets
                fig_scatter.add_trace(go.Scatter(
                    x=individual_volatility * 100,
                    y=individual_returns * 100,
                    mode='markers+text',
                    text=tickers,
                    textposition='top center',
                    marker=dict(size=12, color='lightblue', line=dict(width=2, color='blue')),
                    name='Individual Assets'
                ))
                
                # Add portfolio
                fig_scatter.add_trace(go.Scatter(
                    x=[portfolio_metrics['Annual Volatility'] * 100],
                    y=[portfolio_metrics['Annual Return'] * 100],
                    mode='markers+text',
                    text=['Portfolio'],
                    textposition='top center',
                    marker=dict(size=15, color='red', symbol='diamond'),
                    name='Portfolio'
                ))
                
                # Add S&P 500 benchmark
                fig_scatter.add_trace(go.Scatter(
                    x=[benchmark_metrics['Annual Volatility'] * 100],
                    y=[benchmark_metrics['Annual Return'] * 100],
                    mode='markers+text',
                    text=['S&P 500'],
                    textposition='top center',
                    marker=dict(size=15, color='green', symbol='star'),
                    name='S&P 500'
                ))
                
                fig_scatter.update_layout(
                    title="Risk-Return Profile with S&P 500 Benchmark",
                    xaxis_title="Annual Volatility (%)",
                    yaxis_title="Annual Return (%)",
                    hovermode='closest'
                )
                st.plotly_chart(fig_scatter, use_container_width=True)
                
                # Performance summary table
                st.subheader("📊 Performance Summary")
                summary_data = {
                    'Metric': ['Annual Return', 'Annual Volatility', 'Sharpe Ratio', 'Max Drawdown', 'Alpha', 'Beta', 'Information Ratio'],
                    'Portfolio': [
                        f"{portfolio_metrics['Annual Return']:.2%}",
                        f"{portfolio_metrics['Annual Volatility']:.2%}",
                        f"{portfolio_metrics['Sharpe Ratio']:.3f}",
                        f"{portfolio_metrics['Max Drawdown']:.2%}",
                        f"{relative_metrics['Alpha']:.2%}",
                        f"{relative_metrics['Beta']:.3f}",
                        f"{relative_metrics['Information Ratio']:.3f}"
                    ],
                    'S&P 500': [
                        f"{benchmark_metrics['Annual Return']:.2%}",
                        f"{benchmark_metrics['Annual Volatility']:.2%}",
                        f"{benchmark_metrics['Sharpe Ratio']:.3f}",
                        f"{benchmark_metrics['Max Drawdown']:.2%}",
                        "0.00%",  # Alpha of benchmark is 0
                        "1.000",  # Beta of benchmark is 1
                        "N/A"     # Information ratio not applicable for benchmark
                    ],
                    'Difference': [
                        f"{portfolio_metrics['Annual Return'] - benchmark_metrics['Annual Return']:+.2%}",
                        f"{portfolio_metrics['Annual Volatility'] - benchmark_metrics['Annual Volatility']:+.2%}",
                        f"{portfolio_metrics['Sharpe Ratio'] - benchmark_metrics['Sharpe Ratio']:+.3f}",
                        f"{portfolio_metrics['Max Drawdown'] - benchmark_metrics['Max Drawdown']:+.2%}",
                        f"{relative_metrics['Alpha']:+.2%}",
                        f"{relative_metrics['Beta'] - 1:+.3f}",
                        f"{relative_metrics['Information Ratio']:+.3f}"
                    ]
                }
                
                summary_df = pd.DataFrame(summary_data)
                st.dataframe(summary_df, use_container_width=True)
                
                # Correlation matrix
                st.subheader("🔗 Asset Correlation Matrix")
                corr_matrix = calculate_correlation_matrix(returns)
                fig_corr = px.imshow(
                    corr_matrix,
                    text_auto=True,
                    aspect="auto",
                    color_continuous_scale='RdBu_r',
                    title="Correlation Matrix"
                )
                st.plotly_chart(fig_corr, use_container_width=True)
                
                # Rolling correlation with S&P 500
                st.subheader("📈 Rolling Correlation with S&P 500")
                window_size = min(60, len(portfolio_metrics['Portfolio Returns']) // 4)  # 60 days or 1/4 of data
                rolling_corr = portfolio_metrics['Portfolio Returns'].rolling(window=window_size).corr(benchmark_returns)
                
                fig_rolling_corr = go.Figure()
                fig_rolling_corr.add_trace(go.Scatter(
                    x=rolling_corr.index,
                    y=rolling_corr,
                    mode='lines',
                    name=f'{window_size}-Day Rolling Correlation',
                    line=dict(color='purple', width=2)
                ))
                
                fig_rolling_corr.update_layout(
                    title=f"Portfolio Correlation with S&P 500 ({window_size}-Day Rolling Window)",
                    xaxis_title="Date",
                    yaxis_title="Correlation",
                    yaxis=dict(range=[-1, 1]),
                    hovermode='x unified'
                )
                st.plotly_chart(fig_rolling_corr, use_container_width=True)
                
                # Historical returns distribution comparison
                st.subheader("📊 Returns Distribution Comparison")
                fig_hist = go.Figure()
                
                # Portfolio returns distribution
                fig_hist.add_trace(go.Histogram(
                    x=portfolio_metrics['Portfolio Returns'] * 100,
                    name='Portfolio',
                    opacity=0.7,
                    nbinsx=30
                ))
                
                # S&P 500 returns distribution
                fig_hist.add_trace(go.Histogram(
                    x=benchmark_returns * 100,
                    name='S&P 500',
                    opacity=0.7,
                    nbinsx=30
                ))
                
                fig_hist.update_layout(
                    title="Daily Returns Distribution Comparison (%)",
                    xaxis_title="Daily Return (%)",
                    yaxis_title="Frequency",
                    barmode='overlay'
                )
                st.plotly_chart(fig_hist, use_container_width=True)
                
            else:
                st.error("Unable to fetch data for the specified tickers or S&P 500 benchmark. Please check the ticker symbols.")
    else:
        st.warning("Please enter at least 2 valid tickers with corresponding weights.")

# Information section
with st.expander("ℹ️ About This Dashboard"):
    st.markdown("""
    This Portfolio Risk and Multi-Asset Allocation Dashboard provides comprehensive analysis of your investment portfolio:
    
    **Key Features:**
    - **Portfolio Performance Metrics**: Annual return, volatility, Sharpe ratio, and maximum drawdown
    - **Risk Metrics**: Value at Risk (VaR) and Conditional Value at Risk (CVaR) at 95% confidence level
    - **Visual Analytics**: Cumulative returns, drawdown charts, and risk-return profiles
    - **Correlation Analysis**: Understanding relationships between assets
    - **Individual Asset Analysis**: Performance comparison of individual holdings
    
    **How to Use:**
    1. Set your desired time period and risk-free rate
    2. Enter the number of assets in your portfolio
    3. Input ticker symbols and their respective weights
    4. Click "Analyze Portfolio" to generate comprehensive analysis
    
    **Metrics Explained:**
    - **Sharpe Ratio**: Risk-adjusted return measure (higher is better)
    - **Max Drawdown**: Largest peak-to-trough decline
    - **VaR**: Potential loss at given confidence level
    - **CVaR**: Expected loss beyond VaR threshold
    """)

# Footer
st.markdown("---")
st.markdown("**Note**: This tool is for educational and analytical purposes. Always consult with financial professionals before making investment decisions.")
