import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVR
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, r2_score
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.figure_factory as ff
import datetime
import joblib
import os

# Set page config
st.set_page_config(page_title="Stock Analysis Dashboard", layout="wide")

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stAlert {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Functions from your original code
def fetch_stock_data(stock_name, start_date, end_date):
    try:
        data = yf.download(stock_name, start=start_date, end=end_date)
        if data.empty:
            return None
        return data[['Close', 'High', 'Low', 'Open', 'Volume']]
    except Exception as e:
        return None

def add_technical_features(data):
    # Your original technical features code
    data['SMA_15'] = data['Close'].rolling(window=15).mean()
    data['EMA_15'] = data['Close'].ewm(span=15, adjust=False).mean()
    data['Daily_Return'] = data['Close'].pct_change()
    data['High_Low'] = data['High'] - data['Low']
    data['Open_Close'] = data['Open'] - data['Close']
    data['RSI'] = 100 - (100 / (1 + (data['Close'].diff().clip(lower=0).rolling(window=14).mean() / 
                                    data['Close'].diff().abs().rolling(window=14).mean())))
    
    for lag in range(1, 4):
        data[f'Lag_{lag}'] = data['Close'].shift(lag)
    
    data.dropna(inplace=True)
    return data

def prepare_time_series_data(data, lookback=15):
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i - lookback:i, :])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

def assign_risk_levels_quantile(data):
    return_quantiles = data['Daily_Return'].quantile([0.25, 0.75])
    volatility_quantiles = data['High_Low'].quantile([0.25, 0.75])
    
    conditions = [
        (data['Daily_Return'] > return_quantiles[0.75]) | (data['High_Low'] > volatility_quantiles[0.75]),
        (data['Daily_Return'] <= return_quantiles[0.75]) & (data['Daily_Return'] > return_quantiles[0.25]) & 
        (data['High_Low'] <= volatility_quantiles[0.75]) & (data['High_Low'] > volatility_quantiles[0.25]),
        (data['Daily_Return'] <= return_quantiles[0.25]) & (data['High_Low'] <= volatility_quantiles[0.25])
    ]
    choices = ['High', 'Medium', 'Low']
    data['Risk_Level'] = np.select(conditions, choices, default='Medium')
    return data

def train_lstm_svr(stock_name,scaler):
    start_date = '2015-01-01'
    today = datetime.date.today()
    end_date = today.strftime("%Y-%m-%d")
    # Scale all features

    raw_data = fetch_stock_data(stock_name, start_date, end_date)
    data_with_features = add_technical_features(raw_data)

    scaled_data = scaler.fit_transform(data_with_features)
    
    # Split data
    split_index = int(len(scaled_data) * 0.7)
    train_data = scaled_data[:split_index]
    test_data = scaled_data[split_index:]

    train_data = np.array(train_data)
    test_data = np.array(test_data)
    
    # Prepare data
    lookback = 60
    X_train, y_train = prepare_time_series_data(train_data, lookback)
    X_test, y_test = prepare_time_series_data(test_data, lookback)
    
    num_features = scaled_data.shape[1]
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], num_features))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], num_features))
    
    # Build LSTM model
    lstm_model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(lookback, num_features)),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(1)
    ])
    
    lstm_model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    lstm_model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_test, y_test), verbose=1)
    
    # Feature extractor
    feature_extractor = Sequential([
        LSTM(50, return_sequences=False, input_shape=(lookback, num_features))
    ])
    
    train_features = feature_extractor.predict(X_train, verbose=0)
    test_features = feature_extractor.predict(X_test, verbose=0)
    
    # Train SVR
    svr_model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.01)
    svr_model.fit(train_features, y_train)
    
    # Calculate R2 score
    svr_predictions = svr_model.predict(test_features)
    y_pred_rescaled = scaler.inverse_transform(
        np.concatenate((svr_predictions.reshape(-1, 1), np.zeros((len(svr_predictions), num_features - 1))), axis=1)
    )[:, 0]
    y_test_rescaled = scaler.inverse_transform(
        np.concatenate((y_test.reshape(-1, 1), np.zeros((y_test.shape[0], num_features - 1))), axis=1)
    )[:, 0]
    
    r2 = r2_score(y_test_rescaled, y_pred_rescaled)
    test_dates = data_with_features.index[split_index + lookback:]
    
    return lstm_model, feature_extractor, svr_model, r2, test_dates, y_pred_rescaled,y_test_rescaled

def train_naive_bayes(data):
    # Prepare features and labels
    features = ['Close', 'High_Low', 'Daily_Return', 'RSI']
    X = data[features]
    y = data['Risk_Level']
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    split_index = int(0.8 * len(X_scaled))
    X_train = X_scaled[:split_index]
    X_test = X_scaled[split_index:]
    y_train = y[:split_index]
    y_test = y[split_index:]
    
    # Train model
    nb_model = GaussianNB()
    nb_model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = nb_model.predict(X_test)
    report = classification_report(y_test, y_pred,output_dict=True)
    accuracy = accuracy_score(y_test, y_pred)
    
    return nb_model, scaler, report, accuracy

def train_linear(stock_name):
    start_date = '2015-01-01'
    today = datetime.date.today()
    end_date = today.strftime("%Y-%m-%d")

    # Fetch stock data
    raw_data = fetch_stock_data(stock_name, start_date, end_date)
    data_with_features = add_technical_features(raw_data)

    # Create feature matrix X and target variable y
    feature_columns = ['High', 'Low', 'Open', 'Volume', 'SMA_15', 'EMA_15', 
                      'Daily_Return', 'High_Low', 'Open_Close', 'RSI', 
                      'Lag_1', 'Lag_2', 'Lag_3']
    
    X = data_with_features[feature_columns]
    y = data_with_features['Close'].values.reshape(-1, 1)

    # Handle missing values properly
    X = X.ffill().bfill()
    
    # Scale features (X)
    X_scaler = StandardScaler()
    X_scaled = X_scaler.fit_transform(X)

    # Scale target (y)
    y_scaler = StandardScaler()
    y_scaled = y_scaler.fit_transform(y)

    # Apply PCA
    pca = PCA(n_components=0.75)
    X_pca = pca.fit_transform(X_scaled)

    # Split the data into training and testing sets (80-20)
    split_index = int(len(X_pca) * 0.8)
    X_train = X_pca[:split_index]
    X_test = X_pca[split_index:]
    y_train = y_scaled[:split_index]
    y_test = y_scaled[split_index:]

    # Train Linear Regression model
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)

    # Make predictions on test set
    y_pred_scaled = lr_model.predict(X_test)

    # Inverse transform predictions and actual values
    y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    y_test_original = y_scaler.inverse_transform(y_test).flatten()

    # Calculate R2 score
    r2 = r2_score(y_test_original, y_pred)

    # Get test dates for plotting
    test_dates = data_with_features.index[split_index:]

    # Predict next 7 days
    predictions = []
    last_data = data_with_features.iloc[-1].copy()
    
    for i in range(7):
        # Create a new row of features
        new_features = pd.DataFrame([{
            'High': last_data['High'],
            'Low': last_data['Low'],
            'Open': last_data['Open'],
            'Volume': last_data['Volume'],
            'SMA_15': last_data['Close'] if i == 0 else predictions[-1],
            'EMA_15': last_data['Close'] if i == 0 else predictions[-1],
            'Daily_Return': 0 if i == 0 else (predictions[-1] - last_data['Close']) / last_data['Close'],
            'High_Low': last_data['High'] - last_data['Low'],
            'Open_Close': last_data['Open'] - (last_data['Close'] if i == 0 else predictions[-1]),
            'RSI': last_data['RSI'],
            'Lag_1': last_data['Close'] if i == 0 else predictions[-1],
            'Lag_2': last_data['Lag_1'] if i == 0 else (predictions[-2] if len(predictions) > 1 else last_data['Lag_1']),
            'Lag_3': last_data['Lag_2'] if i == 0 else (predictions[-3] if len(predictions) > 2 else last_data['Lag_2'])
        }])

        # Scale the features
        scaled_features = X_scaler.transform(new_features)
        
        # Transform with PCA
        pca_features = pca.transform(scaled_features)
        
        # Make prediction
        pred_scaled = lr_model.predict(pca_features)
        pred = y_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()[0]
        predictions.append(pred)
        
        # Update last_data for next iteration
        last_data['Close'] = pred

    # Create future dates
    forecast_dates = [pd.to_datetime(end_date) + pd.Timedelta(days=i+1) for i in range(7)]

    return lr_model, pca, r2, test_dates, y_pred, y_test_original, X_scaler, y_scaler, forecast_dates, predictions

def main():
    st.title("📈 Stock Analysis Dashboard")
    st.sidebar.header("Navigation")
    # Sidebar options
    predefined_stocks = {
        "Apple": "AAPL",
        "Microsoft": "MSFT",
        "Google": "GOOGL",
        "Amazon": "AMZN",
        "Tesla": "TSLA",
        "Others": "CUSTOM"
    }
    
    selected_stock_name = st.sidebar.selectbox(
        "Select Stock",
        ["Select a stock"] + list(predefined_stocks.keys())
    )
    
    if selected_stock_name == "Select a stock":
        st.warning("Please select a stock to proceed.")
        return
    
    if selected_stock_name == "Others":
        custom_ticker = st.sidebar.text_input("Enter Stock Ticker Symbol:", "").upper()
        if custom_ticker:
            stock_ticker = custom_ticker
        else:
            st.warning("Please enter a valid stock ticker")
            return
    else:
        stock_ticker = predefined_stocks[selected_stock_name]
    
    # Date range
    start_date = '2019-01-01'
    today = datetime.date.today()
    end_date = today.strftime("%Y-%m-%d")
    
    # Fetch data
    data = fetch_stock_data(stock_ticker, start_date, end_date)
    
    if data is None:
        st.error("Invalid stock ticker or no data available. Please check the symbol and try again.")
        return
    
    # Process data
    data_with_features = add_technical_features(data)

    # Tabbed navigation for interactive layout
    tab1, tab2, tab3 = st.tabs(["📊 LSTM + SVR Predictions", "⚖ Risk Assessment" , "📊 LR Predictions"])
    
    # Tab 1: LSTM + SVR Predictions
    with tab1:
        st.header("LSTM + SVR Price Prediction")
        
        # Initialize scalers
        price_scaler = MinMaxScaler()
        
        # Check if models exist for predefined stocks
        if selected_stock_name != "Others" and os.path.exists(f"models/{stock_ticker}_lstm.h5"):
            # Load pre-trained models
            lstm_model = load_model(f"models/{stock_ticker}_lstm.h5")
            feature_extractor = load_model(f"models/{stock_ticker}_feature_extractor.h5")
            svr_model = joblib.load(f"models/{stock_ticker}_svr.joblib")
            r2 = joblib.load(f"models/{stock_ticker}_r2.joblib")
            evaluation_data = joblib.load(f"models/{stock_ticker}_evaluation_data.joblib")
            test_dates = evaluation_data["test_dates"]
            y_pred_rescaled = evaluation_data["y_pred_rescaled"]
            y_test_rescaled = evaluation_data["y_test_rescaled"]
        else:
            # Train new models
            with st.spinner("Training LSTM+SVR model..."):
                lstm_model, feature_extractor, svr_model, r2, test_dates, y_pred_rescaled, y_test_rescaled = train_lstm_svr(stock_ticker, price_scaler)
                if lstm_model is not None:
                    os.makedirs("models", exist_ok=True)
                    lstm_model.save(f"models/{stock_ticker}_lstm.h5")
                    feature_extractor.save(f"models/{stock_ticker}_feature_extractor.h5")
                    joblib.dump(svr_model, f"models/{stock_ticker}_svr.joblib")
                    joblib.dump(r2, f"models/{stock_ticker}_r2.joblib")
                    joblib.dump({
                        "test_dates": test_dates,
                        "y_pred_rescaled": y_pred_rescaled,
                        "y_test_rescaled": y_test_rescaled
                    }, f"models/{stock_ticker}_evaluation_data.joblib")
        
        # Display R2 score
        # st.metric("Model R² Score", f"{r2:.4f}")
        st.markdown(f"""
<div style='
    text-align: center; 
    margin: 1rem 0;'>
    <h3 style='color: #1565c0; font-weight: bold; margin: 0;'>Model R² Score</h3>
    <div style='
        font-size: 2.5rem; 
        font-weight: bold; 
        color: #43a047; 
        margin: 0.5rem 0;'>
        {r2:.4f}
    </div>
    <div style='
        height: 10px; 
        width: 80%; 
        margin: 0 auto; 
        background: #e0e0e0; 
        border-radius: 5px; 
        position: relative;'>
        <div style='
            width: {r2 * 100}%; 
            height: 100%; 
            background: linear-gradient(to right, #1e88e5, #43a047); 
            border-radius: 5px;'>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

        
        # Predict next 7 days
        lookback = 60
        scaled_data = price_scaler.fit_transform(data_with_features)
        recent_features = scaled_data[-lookback:].reshape((1, lookback, scaled_data.shape[1]))
        
        # Make predictions
        predictions = []
        forecast_features = recent_features.copy()
        
        for _ in range(7):
            lstm_features = feature_extractor.predict(forecast_features, verbose=0)
            next_day_prediction = svr_model.predict(lstm_features)
            next_day_rescaled = price_scaler.inverse_transform(
                np.concatenate((next_day_prediction.reshape(-1, 1), 
                              np.zeros((1, scaled_data.shape[1] - 1))), axis=1)
            )[:, 0]
            predictions.append(next_day_rescaled[0])
            
            new_features = np.concatenate(
                (next_day_prediction.reshape(-1, 1), forecast_features[0, -1, 1:].reshape(1, -1)), 
                axis=1
            )
            forecast_features = np.append(forecast_features[:, 1:, :], [new_features], axis=1)
        
        # Create future dates
        forecast_dates = [pd.to_datetime(end_date) + pd.Timedelta(days=i+1) for i in range(7)]
        
        

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=test_dates, y=y_test_rescaled, mode='lines', name='Actual', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=test_dates, y=y_pred_rescaled, mode='lines', name='Predictions', line=dict(color='red')))
        fig.add_trace(go.Scatter(x=forecast_dates, y=predictions, mode='lines', name='Forecast', line=dict(color='green', dash='dash')))

        fig.update_layout(
            title=f"{stock_ticker} Stock Price Forecast",
            xaxis_title="Date",
            yaxis_title="Price",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            template="plotly_white"
        )
        st.plotly_chart(fig)

        # Display predictions table
        forecast_df = pd.DataFrame({
            'Date': forecast_dates,
            'Predicted Price': predictions
        })
        # st.dataframe(forecast_df.style.format({'Predicted Price': '{:.2f}'}))
        forecast_df['Date'] = pd.to_datetime(forecast_df['Date']).dt.date

# Reset the index to start from 1
        forecast_df.index = range(1, len(forecast_df) + 1)
        # Create a visually enhanced forecast table section
        st.markdown("### 📊 Stock Price Forecast Table")

        # Style forecast data with conditional formatting
        styled_forecast_df = forecast_df.style.format({'Predicted Price': '{:.2f}'}).apply(
            lambda x: ['background-color: #d1ffd6' if v > forecast_df['Predicted Price'].mean() else '' for v in x],
            subset=['Predicted Price']
        )

        # Display the styled forecast table
        st.dataframe(styled_forecast_df, use_container_width=True)

    # Tab 2: Risk Assessment
    with tab2:
        st.header("Risk Assessment")
        risk_data = assign_risk_levels_quantile(data_with_features)

        with st.spinner("Training Naive Bayes model..."):
            nb_model, nb_scaler, report, accuracy = train_naive_bayes(risk_data)

            # Display model performance
            # st.metric("Model Accuracy", f"{accuracy:.2%}")
            st.markdown(f"""
<div style='
    text-align: center; 
    margin: 2rem 0;'>
    <h3 style='color: #1565c0; font-weight: bold; margin: 0;'>Model Accuracy</h3>
    <div style='
        font-size: 2.5rem; 
        font-weight: bold; 
        color: #1e88e5; 
        margin: 0.5rem 0;'>
        {accuracy:.2f}
    </div>
    <div style='
        height: 10px; 
        width: 80%; 
        margin: 0 auto; 
        background: #e0e0e0; 
        border-radius: 5px; 
        position: relative;'>
        <div style='
            width: {accuracy * 100}%; 
            height: 100%; 
            background: linear-gradient(to right, #43a047, #1e88e5); 
            border-radius: 5px;'>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

            


# Convert classification report to DataFrame

            classification_report_df = pd.DataFrame.from_dict(report).T
            classification_report_df = classification_report_df.applymap(
                lambda x: f"{x:.2f}" if isinstance(x, (float, int)) else x
            )

            # Display classification report as a plotly table with better styling
            st.markdown("### 📋 Classification Report")
            fig = ff.create_table(classification_report_df, index=True)

            # Update the layout to customize the table's appearance
            fig.update_layout(
                height=300,
                margin=dict(t=0, b=0, l=0, r=0),
                font=dict(family="Arial", size=14, color="black"),
            )

            # Display the figure
            st.plotly_chart(fig)

            # Predict next day's risk
            recent_features = nb_scaler.transform(risk_data[['Close', 'High_Low', 'Daily_Return', 'RSI']].iloc[-1:])
            next_day_risk = nb_model.predict(recent_features)

            # Display next day's risk prediction
            risk_color = {
                'High': 'red',
                'Medium': 'yellow',
                'Low': 'green'
            }

            st.markdown(f"""
### Next Day Risk Prediction
<div style='position: relative; width: 150px; height: 150px; margin: auto;'>
    <svg viewBox='0 0 36 36' width='100%' height='100%' style='transform: rotate(-90deg);'>
        <path d='M18 2 a16 16 0 1 1 0 32 a16 16 0 1 1 0 -32' 
              fill='none' stroke='{risk_color[next_day_risk[0]]}' 
              stroke-width='4' stroke-dasharray='100, 100'></path>
    </svg>
    <div style='position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); 
                font-size: 1.2rem; font-weight: bold; color: black; text-align: center;'>
        {next_day_risk[0]}<br>Risk
    </div>
</div>
""", unsafe_allow_html=True)



            # Add numeric representation for risk levels
            risk_mapping = {'High': 3, 'Medium': 2, 'Low': 1}
            risk_data['Risk_Numeric'] = risk_data['Risk_Level'].map(risk_mapping)

            # Ensure all dates are included (fill missing dates)
            full_date_range = pd.date_range(risk_data.index[-30].normalize(), pd.Timestamp.today().normalize())
            risk_data = risk_data.reindex(full_date_range)
            risk_data['Risk_Numeric'] = risk_data['Risk_Numeric'].fillna(method='ffill')  # Fill missing risk levels
            risk_data.index.name = 'Date'

            # Prepare data for Plotly graph
            risk_dates = risk_data.index
            risk_values = risk_data['Risk_Numeric'].tolist()

            # Correctly calculate the next day's date (today + 1 day)
            next_day_date = pd.Timestamp.today().normalize() + pd.Timedelta(days=1)

            # Append the next day risk prediction
            risk_dates = list(risk_dates) + [next_day_date]
            risk_values = risk_values + [risk_mapping[next_day_risk[0]]]

            # Create Plotly graph
            fig2 = go.Figure()

            # Plot historical risk levels
            fig2.add_trace(go.Scatter(
                x=risk_dates[:-1],
                y=risk_values[:-1],
                mode='lines+markers',
                name='Risk History',
                marker=dict(color='blue'),
                line=dict(color='blue')
            ))

            # Highlight next day's risk prediction
            fig2.add_trace(go.Scatter(
                x=[risk_dates[-1]],
                y=[risk_values[-1]],
                mode='markers+text',
                name="Next Day Risk",
                marker=dict(size=10, color=risk_color[next_day_risk[0]], symbol='star'),
                text=[next_day_risk[0]],
                textposition="top center"
            ))

            fig2.update_layout(
                title=f"{stock_ticker} Risk Level History and Prediction",
                xaxis_title="Date",
                yaxis_title="Risk Level",
                yaxis=dict(
                    tickvals=[1, 2, 3],
                    ticktext=['Low', 'Medium', 'High']
                ),
                template="plotly_white"
            )

            st.plotly_chart(fig2)
        pass


    with tab3:
        st.header("Linear Regression Time Series")
        
        if selected_stock_name != "Others" and os.path.exists(f"models/{stock_ticker}_lr.joblib"):
            lr_model = joblib.load(f"models/{stock_ticker}_lr.joblib")
            pca = joblib.load(f"models/{stock_ticker}_pca.joblib")
            r2 = joblib.load(f"models/{stock_ticker}_lr_r2.joblib")
            evaluation_data = joblib.load(f"models/{stock_ticker}_lr_evaluation_data.joblib")
            test_dates = evaluation_data["test_dates"]
            y_pred = evaluation_data["y_pred"]
            y_test = evaluation_data["y_test"]
            X_scaler = joblib.load(f"models/{stock_ticker}_X_scaler.joblib")
            y_scaler = joblib.load(f"models/{stock_ticker}_y_scaler.joblib")
            forecast_dates = evaluation_data["forecast_dates"]
            predictions = evaluation_data["predictions"]
        else:
            with st.spinner("Training Linear Regression model..."):
                lr_model, pca, r2, test_dates, y_pred, y_test, X_scaler, y_scaler, forecast_dates, predictions = train_linear(stock_ticker)
                if lr_model is not None:
                    os.makedirs("models", exist_ok=True)
                    joblib.dump(lr_model, f"models/{stock_ticker}_lr.joblib")
                    joblib.dump(pca, f"models/{stock_ticker}_pca.joblib")
                    joblib.dump(r2, f"models/{stock_ticker}_lr_r2.joblib")
                    joblib.dump({
                        "test_dates": test_dates,
                        "y_pred": y_pred,
                        "y_test": y_test,
                        "forecast_dates": forecast_dates,
                        "predictions": predictions
                    }, f"models/{stock_ticker}_lr_evaluation_data.joblib")
                    joblib.dump(X_scaler, f"models/{stock_ticker}_X_scaler.joblib")
                    joblib.dump(y_scaler, f"models/{stock_ticker}_y_scaler.joblib")

        # Display R2 score (existing code)
        st.markdown(f"""
        <div style='
            text-align: center; 
            margin: 1rem 0;'>
            <h3 style='color: #1565c0; font-weight: bold; margin: 0;'>Model R² Score</h3>
            <div style='
                font-size: 2.5rem; 
                font-weight: bold; 
                color: #43a047; 
                margin: 0.5rem 0;'>
                {r2:.4f}
            </div>
            <div style='
                height: 10px; 
                width: 80%; 
                margin: 0 auto; 
                background: #e0e0e0; 
                border-radius: 5px; 
                position: relative;'>
                <div style='
                    width: {r2 * 100}%; 
                    height: 100%; 
                    background: linear-gradient(to right, #1e88e5, #43a047); 
                    border-radius: 5px;'>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Create and display the plot with forecast
        fig = go.Figure()
        
        # Plot actual values
        fig.add_trace(go.Scatter(
            x=test_dates, 
            y=y_test, 
            mode='lines', 
            name='Actual', 
            line=dict(color='blue')
        ))
        
        # Plot predictions
        fig.add_trace(go.Scatter(
            x=test_dates, 
            y=y_pred, 
            mode='lines', 
            name='Predictions', 
            line=dict(color='red')
        ))

        # Plot forecast
        fig.add_trace(go.Scatter(
            x=forecast_dates,
            y=predictions,
            mode='lines',
            name='Forecast',
            line=dict(color='green', dash='dash')
        ))

        # Update layout
        fig.update_layout(
            title=f"{stock_ticker} Stock Price Prediction",
            xaxis_title="Date",
            yaxis_title="Price",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            template="plotly_white"
        )
        
        st.plotly_chart(fig)


        forecast_df = pd.DataFrame({
            'Date': forecast_dates,
            'Predicted Price': predictions
        })
        forecast_df['Date'] = pd.to_datetime(forecast_df['Date']).dt.date
        forecast_df.index = range(1, len(forecast_df) + 1)

        st.markdown("### 📊 Stock Price Forecast Table")
        
        def highlight_above_mean(x):
            return ['background-color: #d1ffd6' if x > forecast_df['Predicted Price'].mean() else '' for _ in [0]][0]
        
        styled_forecast_df = forecast_df.style.format({
            'Predicted Price': '{:.2f}'
        }).map(
            highlight_above_mean,
            subset=['Predicted Price']
        )
        
        st.dataframe(styled_forecast_df, use_container_width=True)

if __name__== "__main__":
     main()
