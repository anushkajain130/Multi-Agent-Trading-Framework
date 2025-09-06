def analyze_technical_indicators(symbol, lookback_days=180):
    import numpy as np
    import pandas as pd
    import yfinance as yf
    from scipy import stats
    end_date = pd.Timestamp.now()
    start_date = end_date - pd.Timedelta(days=lookback_days+100)
    
    stock_data = yf.download(symbol, start=start_date, end=end_date, progress=False)
    
    if len(stock_data) < 100:
        print(f"Insufficient data for {symbol}")
        return None
    
    df = stock_data.copy()
    
    # Price-based indicators
    df['sma_20'] = df['Close'].rolling(window=20).mean()
    df['sma_50'] = df['Close'].rolling(window=50).mean()
    df['sma_200'] = df['Close'].rolling(window=200).mean()
    df['ema_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['ema_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    
    # MACD
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # Bollinger Bands
    df['bb_middle'] = df['Close'].rolling(window=20).mean()
    df['bb_std'] = df['Close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
    df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
    
    # RSI
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
    rs = gain / loss.replace(0, np.nan).fillna(1)  # Avoid division by zero
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Stochastic Oscillator
    low_14 = df['Low'].rolling(window=14).min()
    high_14 = df['High'].rolling(window=14).max()
    df['stoch_k'] = 100 * ((df['Close'] - low_14) / (high_14 - low_14).replace(0, np.nan).fillna(1))
    df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()
    
    # Average Directional Index (ADX) - FIX HERE
    try:
        # Calculate True Range first
        df['tr1'] = df['High'] - df['Low']
        df['tr2'] = abs(df['High'] - df['Close'].shift(1))
        df['tr3'] = abs(df['Low'] - df['Close'].shift(1))
        df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
        
        # Directional Movement
        df['up_move'] = df['High'] - df['High'].shift(1)
        df['down_move'] = df['Low'].shift(1) - df['Low']
        
        # Positive and Negative Directional Movement
        df['plus_dm'] = np.where((df['up_move'] > df['down_move']) & (df['up_move'] > 0), df['up_move'], 0)
        df['minus_dm'] = np.where((df['down_move'] > df['up_move']) & (df['down_move'] > 0), df['down_move'], 0)
        
        # Calculate ATR
        df['atr'] = df['tr'].rolling(window=14).mean()
        
        # Calculate +DI and -DI
        df['plus_di'] = 100 * df['plus_dm'].rolling(window=14).mean() / df['atr']
        df['minus_di'] = 100 * df['minus_dm'].rolling(window=14).mean() / df['atr']
        
        # Calculate DX and ADX
        df['dx'] = 100 * abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di']).replace(0, np.nan).fillna(1)
        df['adx'] = df['dx'].rolling(window=14).mean()
    except Exception as e:
        print(f"Error calculating ADX: {e}")
        df['adx'] = 25  # Default neutral value
        df['plus_di'] = 20
        df['minus_di'] = 20
    
    # On-Balance Volume (OBV)
    df['obv'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
    
    # Ichimoku Cloud
    df['tenkan_sen'] = (df['High'].rolling(window=9).max() + df['Low'].rolling(window=9).min()) / 2
    df['kijun_sen'] = (df['High'].rolling(window=26).max() + df['Low'].rolling(window=26).min()) / 2
    df['senkou_span_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(26)
    df['senkou_span_b'] = ((df['High'].rolling(window=52).max() + df['Low'].rolling(window=52).min()) / 2).shift(26)
    df['chikou_span'] = df['Close'].shift(-26)
    
    # Fibonacci Retracement
    max_price = df['Close'][-lookback_days:].max()
    min_price = df['Close'][-lookback_days:].min()
    diff = max_price - min_price
    
    # Momentum Indicators
    df['roc'] = df['Close'].pct_change(10) * 100
    df['cci'] = (df['Close'] - df['Close'].rolling(window=20).mean()) / (0.015 * df['Close'].rolling(window=20).std().replace(0, np.nan).fillna(1))
    df['williams_r'] = -100 * (high_14 - df['Close']) / (high_14 - low_14).replace(0, np.nan).fillna(1)
    
    # Trend Strength
    df['adx_trend'] = np.where(df['adx'] > 25, 'Strong', 'Weak')
    
    
    
    # Fill NaN values for calculations
    df = df.fillna(method='ffill').fillna(method='bfill')
    
    # Get the current values (last row)
    latest_data = df.iloc[-1].copy()
    

    rsi_signal = "Oversold" if latest_data['rsi'].iloc[-1] < 30 else "Overbought" if latest_data['rsi'].iloc[-1] > 70 else "Neutral"
    macd_signal = "Bullish" if latest_data['macd'].iloc[-1] > latest_data['macd_signal'].iloc[-1] else "Bearish"
    stoch_signal = "Oversold" if latest_data['stoch_k'].iloc[-1] < 20 else "Overbought" if latest_data['stoch_k'].iloc[-1] > 80 else "Neutral"
    
    technical_indicators = {
        'symbol': symbol,
        'last_price': latest_data['Close'],
        'analysis_date': str(end_date.date()),
        
        # Moving Averages
        'sma_20': latest_data['sma_20'],
        'sma_50': latest_data['sma_50'],
        'sma_200': latest_data['sma_200'],
        'ema_12': latest_data['ema_12'],
        'ema_26': latest_data['ema_26'],
        
        # MACD
        'macd': latest_data['macd'],
        'macd_signal': latest_data['macd_signal'],
        'macd_histogram': latest_data['macd_hist'],
        'macd_indicator': macd_signal,
        
        # Bollinger Bands
        'bollinger_upper': latest_data['bb_upper'],
        'bollinger_middle': latest_data['bb_middle'],
        'bollinger_lower': latest_data['bb_lower'],
        'bollinger_width': latest_data['bb_width'],
        
        # RSI
        'rsi': latest_data['rsi'],
        'rsi_signal': rsi_signal,
        
        # Stochastic
        'stochastic_k': latest_data['stoch_k'],
        'stochastic_d': latest_data['stoch_d'],
        'stochastic_signal': stoch_signal,
        
        # ADX
        'adx': latest_data['adx'],
        'plus_di': latest_data['plus_di'],
        'minus_di': latest_data['minus_di'],
        'adx_trend': latest_data['adx_trend'],
        
 
        
        # Ichimoku
        'tenkan_sen': latest_data['tenkan_sen'],
        'kijun_sen': latest_data['kijun_sen'],
        'senkou_span_a': latest_data.get('senkou_span_a', np.nan),
        'senkou_span_b': latest_data.get('senkou_span_b', np.nan),
        
        # Fibonacci
        'fib_100': max_price,
        'fib_61.8': min_price + 0.618 * diff,
        'fib_50': min_price + 0.5 * diff,
        'fib_38.2': min_price + 0.382 * diff,
        'fib_23.6': min_price + 0.236 * diff,
        'fib_0': min_price,
        
        # Additional indicators
        'roc': latest_data['roc'],
        'cci': latest_data['cci'],
        'williams_r': latest_data['williams_r'],

    }
    
    # Signal calculation
    plus_di_gt_minus = latest_data['plus_di'] > latest_data['minus_di']
    
    bullish_signals = sum([

        technical_indicators['macd_indicator'] == 'Bullish',
        technical_indicators['rsi_signal'] == 'Oversold',
    ])
    
    bearish_signals = sum([

        technical_indicators['macd_indicator'] == 'Bearish',
        technical_indicators['rsi_signal'] == 'Overbought',
    ])
    
    if bullish_signals >= 4:
        technical_indicators['overall_signal'] = 'Strongly Bullish'
    elif bullish_signals >= 3:
        technical_indicators['overall_signal'] = 'Bullish'
    elif bearish_signals >= 4:
        technical_indicators['overall_signal'] = 'Strongly Bearish'
    elif bearish_signals >= 3:
        technical_indicators['overall_signal'] = 'Bearish'
    else:
        technical_indicators['overall_signal'] = 'Neutral'
    
    print(f"Technical analysis completed for {symbol}")
    return technical_indicators

if __name__ == "__main__":
    symbol = "GOOG"
    indicators = analyze_technical_indicators(symbol, lookback_days=180)
    
    if indicators:
        print(f"\nTechnical Analysis for {symbol}")
        print(f"Overall Signal: {indicators['overall_signal']}")
        print(f"Current Price: ${indicators['last_price'].iloc[-1]:.2f}")
        print(f"RSI: {indicators['rsi'].values[-1]:.2f} ({indicators['rsi_signal']})")
        print(f"MACD: {indicators['macd'].values[-1]:.4f} ({indicators['macd_indicator']})")