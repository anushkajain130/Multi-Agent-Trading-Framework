def predict_stock_prices(symbol, days_to_predict=30, training_years=5, seq_length=30):
    import numpy as np
    import pandas as pd
    import yfinance as yf
    import matplotlib.pyplot as plt
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    import os
    
    class LSTMRegressor(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers=2):
            super(LSTMRegressor, self).__init__()
            self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                                num_layers=num_layers, batch_first=True, dropout=0.1)
            self.fc = nn.Linear(hidden_size, 1)

        def forward(self, x):
            out, _ = self.lstm(x)
            out = out[:, -1, :]  
            out = self.fc(out)
            return out
    
    def create_sequences(data, seq_length):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i+seq_length])
            y.append(data[i+seq_length, 0])  
        return np.array(X), np.array(y)
    
    print(f"🔍 Fetching {training_years} years of historical data for {symbol}...")
    
    end_date = pd.Timestamp.now().strftime('%Y-%m-%d')
    start_date = (pd.Timestamp.now() - pd.Timedelta(days=training_years*365)).strftime('%Y-%m-%d')
    
    try:
        stock_data = yf.download(symbol, start=start_date, end=end_date)
        print(f"✅ Downloaded {len(stock_data)} trading days of data")
        
        if len(stock_data) < 100:
            print("❌ Not enough historical data to train a reliable model")
            return None
    
    except Exception as e:
        print(f"❌ Error downloading data: {str(e)}")
        return None
    
    df = stock_data[['Close', 'High', 'Low', 'Open', 'Volume']].copy()
    
    for lag in [1, 2, 3, 7]:
        df[f'Close_lag_{lag}'] = df['Close'].shift(lag)
    
    df['Close_rolling_mean_7'] = df['Close'].rolling(7).mean()
    df['Close_rolling_mean_21'] = df['Close'].rolling(21).mean()
    df['Volume_rolling_mean_7'] = df['Volume'].rolling(7).mean()
    df['High_Low_diff'] = df['High'] - df['Low']
    df['Close_Open_diff'] = df['Close'] - df['Open']
    
    df = df.fillna(method='bfill').fillna(0)
    
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df.astype(float))
    
    X, y = create_sequences(scaled_data, seq_length)
    
    split_index = int(0.8 * len(X))
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
    
    BATCH_SIZE = 64
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    input_size = X_train.shape[2]  # Number of features
    model = LSTMRegressor(input_size=input_size, hidden_size=64)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    EPOCHS = 50
    print(f"⏳ Training model for {EPOCHS} epochs...")
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * X_batch.size(0)
        
        avg_loss = total_loss / len(train_loader.dataset)
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{EPOCHS} - Loss: {avg_loss:.6f}")
    
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for X_batch, _ in test_loader:
            preds = model(X_batch)
            predictions.append(preds.numpy())
    
    y_pred = np.concatenate(predictions).flatten()
    y_true = y_test_tensor.numpy().flatten()
    
    min_c, max_c = scaler.data_min_[0], scaler.data_max_[0]
    y_pred_inv = y_pred * (max_c - min_c) + min_c
    y_true_inv = y_true * (max_c - min_c) + min_c
    
    rmse = np.sqrt(mean_squared_error(y_true_inv, y_pred_inv))
    mae = mean_absolute_error(y_true_inv, y_pred_inv)
    r2 = r2_score(y_true_inv, y_pred_inv)
    
    print(f"📏 Model performance:")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  MAE: {mae:.2f}")
    print(f"  R²: {r2:.4f}")
    
    print(f"🔮 Predicting {days_to_predict} days into the future...")
    
    # Get the most recent sequence of data
    last_sequence = scaled_data[-seq_length:]
    
    # Make iterative predictions
    predictions = []
    current_sequence = last_sequence.copy()
    
    with torch.no_grad():
        for day in range(days_to_predict):
            input_tensor = torch.tensor(current_sequence, dtype=torch.float32).unsqueeze(0)
            
            pred = model(input_tensor).item()
            predictions.append(pred)
            
            new_row = np.zeros(input_size)
            new_row[0] = pred  # Set predicted Close price
            
            current_sequence = np.roll(current_sequence, -1, axis=0)
            current_sequence[-1] = new_row
    
    predictions_inv = np.array(predictions) * (max_c - min_c) + min_c
    
    last_date = df.index[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days_to_predict, freq='B')  # Business days
    
    last_actual_price = df['Close'].iloc[-1]
    if isinstance(last_actual_price, pd.Series):
        last_actual_price = last_actual_price.iloc[-1]
    results = {
        'symbol': symbol,
        'predictions': predictions_inv,
        'dates': future_dates,
        'last_actual_price': last_actual_price,
        'predicted_change': predictions_inv[-1] - last_actual_price,
        'predicted_change_pct': ((predictions_inv[-1] - last_actual_price) / last_actual_price) * 100,
        'model_rmse': rmse,
        'model_mae': mae,
        'model_r2': r2
    }
    
    print(f"\n📈 {symbol} Stock Price Predictions")
    print("=" * 50)
    print(f"Last Actual Price: ${last_actual_price:.2f}")
    print(f"Predicted Price ({future_dates[-1].strftime('%Y-%m-%d')}): ${predictions_inv[-1]:.2f}")
    print(f"Predicted Change: ${results['predicted_change']:.2f} ({results['predicted_change_pct']:.2f}%)")
    
    print("\n📅 Detailed Predictions:")
    print("-" * 40)
    for date, price in zip(future_dates[:10], predictions_inv[:10]):
        print(f"{date.strftime('%Y-%m-%d')}: ${price:.2f}")
    
    if len(predictions_inv) > 10:
        print("...")
        print(f"{future_dates[-1].strftime('%Y-%m-%d')}: ${predictions_inv[-1]:.2f}")
        
    return results

if __name__ == "__main__":
    symbol = "GOOG"  
    results = predict_stock_prices(symbol, days_to_predict=10, training_years=5, seq_length=30)
    
    if results:
        print("\n📊 Prediction Summary:")
        print(f"Symbol: {results['symbol']}")
        print(f"Last Actual Price: ${results['last_actual_price']:.2f}")
        print(f"Predicted Price: ${results['predictions'][-1]:.2f}")
        print(f"Predicted Change: ${results['predicted_change']:.2f} ({results['predicted_change_pct']:.2f}%)")
        print(f"Model RMSE: {results['model_rmse']:.2f}")
        print(f"Model MAE: {results['model_mae']:.2f}")
        print(f"Model R²: {results['model_r2']:.4f}")