def analyze_stock_risk(symbol, lookback_years=5, benchmark_symbol="SPY", risk_free_rate=0.03):
    import numpy as np
    import pandas as pd
    import yfinance as yf
    from scipy import stats
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import r2_score
    
    # Calculate date ranges
    end_date = pd.Timestamp.now()
    start_date = end_date - pd.Timedelta(days=lookback_years*365)
    
    # Download data
    stock_data = yf.download(symbol, start=start_date, end=end_date, progress=False)
    benchmark_data = yf.download(benchmark_symbol, start=start_date, end=end_date, progress=False)
    
    if len(stock_data) < 100 or len(benchmark_data) < 100:
        print(f"❌ Insufficient data for {symbol}")
        return None
    
    # Calculate returns
    stock_data['daily_return'] = stock_data['Close'].pct_change()
    benchmark_data['daily_return'] = benchmark_data['Close'].pct_change()

    # Merge datasets
    merged_data = pd.DataFrame({
        'stock_return': stock_data['daily_return'],
        'benchmark_return': benchmark_data['daily_return']
    }).dropna()
    
    # --- Basic metrics ---
    daily_risk_free = risk_free_rate / 252
    mean_daily_return = merged_data['stock_return'].mean()
    std_daily_return = merged_data['stock_return'].std()
    annualized_return = (1 + mean_daily_return) ** 252 - 1
    annualized_volatility = std_daily_return * np.sqrt(252)
    
    # --- Value at Risk and Expected Shortfall ---
    var_95 = np.percentile(merged_data['stock_return'], 5)
    var_99 = np.percentile(merged_data['stock_return'], 1)
    cvar_95 = merged_data['stock_return'][merged_data['stock_return'] <= var_95].mean()
    cvar_99 = merged_data['stock_return'][merged_data['stock_return'] <= var_99].mean()
    
    # --- Drawdown analysis ---
    cum_returns = (1 + merged_data['stock_return']).cumprod()
    running_max = cum_returns.cummax()
    drawdown = (cum_returns / running_max - 1)
    max_drawdown = drawdown.min()
    
    # --- Performance ratios ---
    excess_return = mean_daily_return - daily_risk_free
    
    # Sharpe ratio
    sharpe_ratio = (excess_return / std_daily_return) * np.sqrt(252) if std_daily_return != 0 else 0
    
    # Sortino ratio
    downside_returns = merged_data['stock_return'][merged_data['stock_return'] < 0]
    downside_deviation = downside_returns.std()
    sortino_ratio = (excess_return / downside_deviation) * np.sqrt(252) if downside_deviation != 0 else 0
    
    # --- Advanced model for Beta and Alpha ---
    # Create lagged features
    for lag in [1, 2, 3, 5]:
        merged_data[f'benchmark_lag_{lag}'] = merged_data['benchmark_return'].shift(lag)
    
    merged_data['rolling_correlation'] = merged_data['stock_return'].rolling(20).corr(merged_data['benchmark_return'])
    merged_data['benchmark_volatility'] = merged_data['benchmark_return'].rolling(20).std()
    merged_data['rel_volatility'] = merged_data['stock_return'].rolling(20).std() / merged_data['benchmark_volatility']
    
    model_data = merged_data.dropna()
    X = model_data[[col for col in model_data.columns if col != 'stock_return']]
    y = model_data['stock_return']
    
    # Train Random Forest model
    rf_model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
    rf_model.fit(X, y)
    
    # Calculate metrics
    y_pred = rf_model.predict(X)
    r_squared = r2_score(y, y_pred)
    
    # Calculate traditional beta for comparison
    covariance = merged_data['stock_return'].cov(merged_data['benchmark_return'])
    benchmark_variance = merged_data['benchmark_return'].var()
    beta = covariance / benchmark_variance if benchmark_variance != 0 else 1
    
    # Feature importance as a proxy for factor exposures
    feature_importance = dict(zip(X.columns, rf_model.feature_importances_))
    
    # Alpha - excess return not explained by the model
    alpha = annualized_return - (beta * (merged_data['benchmark_return'].mean() * 252 - risk_free_rate))
    
    # --- Advanced risk metrics ---
    # Information Ratio
    active_return = merged_data['stock_return'] - merged_data['benchmark_return']
    tracking_error = active_return.std()
    information_ratio = (active_return.mean() / tracking_error) * np.sqrt(252) if tracking_error != 0 else 0
    
    # Distribution statistics
    skewness = stats.skew(merged_data['stock_return'])
    kurtosis = stats.kurtosis(merged_data['stock_return'])
    jarque_bera = stats.jarque_bera(merged_data['stock_return'].dropna())
    
    # Market capture ratios
    up_market = merged_data[merged_data['benchmark_return'] > 0]
    down_market = merged_data[merged_data['benchmark_return'] < 0]
    upside_capture = (up_market['stock_return'].mean() / up_market['benchmark_return'].mean()) if len(up_market) > 0 and up_market['benchmark_return'].mean() != 0 else 0
    downside_capture = (down_market['stock_return'].mean() / down_market['benchmark_return'].mean()) if len(down_market) > 0 and down_market['benchmark_return'].mean() != 0 else 0
    
    # Calmar ratio
    calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown < 0 else np.nan

    risk_score = 0
    risk_score += (annualized_volatility / 0.20) * 30  # Volatility component
    risk_score += (abs(max_drawdown) / 0.30) * 30      # Drawdown component
    risk_score += (beta / 1.5) * 20                    # Beta component
    risk_score += (abs(downside_deviation) / 0.15) * 20 # Downside risk component
    
    risk_rating = "Low"
    if risk_score > 80:
        risk_rating = "Very High"
    elif risk_score > 60:
        risk_rating = "High"
    elif risk_score > 40:
        risk_rating = "Moderate"
    
    risk_metrics = {
        'symbol': symbol,
        'analysis_period': f"{start_date.date()} to {end_date.date()}",
        'trading_days': len(merged_data),
        
        # Return metrics
        'annualized_return': annualized_return,
        'annualized_volatility': annualized_volatility,
        
        # VaR and CVaR
        'var_95': var_95,
        'var_99': var_99,
        'cvar_95': cvar_95,
        'cvar_99': cvar_99,
        
        # Drawdown
        'max_drawdown': max_drawdown,
        
        # Ratios
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'information_ratio': information_ratio,
        'calmar_ratio': calmar_ratio,
        
        # CAPM and model metrics
        'beta': beta,
        'alpha': alpha,
        'r_squared': r_squared,
        'model_feature_importance': feature_importance,
        
        # Distribution properties
        'skewness': skewness,
        'kurtosis': kurtosis,
        'jarque_bera_stat': jarque_bera[0],
        'jarque_bera_pval': jarque_bera[1],
        'is_normal': jarque_bera[1] > 0.05,
        
        # Market capture
        'upside_capture': upside_capture,
        'downside_capture': downside_capture,
        
        'tracking_error': tracking_error * np.sqrt(252),
        'risk_score': risk_score,
        'risk_rating': risk_rating
    }
    
    return risk_metrics

if __name__ == "__main__":
    metrics  = analyze_stock_risk("GOOG", lookback_years=2)
    print(metrics)