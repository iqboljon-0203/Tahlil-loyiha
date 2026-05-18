"""
AI Engine - LSTM Model & Data Processing for Predictive Maintenance
Multivariate LSTM architecture for industrial equipment health forecasting
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# Phase 1: Data Engineering
# ============================================================

def clean_data(df):
    """Handle missing values with interpolation and trend-based filling"""
    df = df.copy()
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].interpolate(method='linear', limit_direction='both')
            df[col] = df[col].ffill().bfill()
    return df


def generate_synthetic_data(df, target_days=3650, degradation_factor=0.0003):
    """
    Extend historical data with synthetic degradation trends.
    Simulates mechanical wear and tear over time.
    """
    n_existing = len(df)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    n_needed = max(0, target_days - n_existing)
    if n_needed == 0:
        return df

    last_vals = df[numeric_cols].iloc[-1].values
    trends = np.zeros(len(numeric_cols))
    if n_existing > 10:
        window = min(50, n_existing // 2)
        early = df[numeric_cols].iloc[:window].mean().values
        late = df[numeric_cols].iloc[-window:].mean().values
        trends = (late - early) / n_existing

    synthetic_rows = []
    for i in range(1, n_needed + 1):
        deg = 1 + degradation_factor * (i / 365)
        noise = np.random.normal(0, 0.02, len(numeric_cols))
        new_vals = last_vals + trends * i * deg + noise * np.abs(last_vals + 0.01)
        synthetic_rows.append(new_vals)

    syn_df = pd.DataFrame(synthetic_rows, columns=numeric_cols)

    if 'Time' in df.columns:
        last_time = pd.to_datetime(df['Time'].iloc[-1])
        syn_df['Time'] = pd.date_range(start=last_time + pd.Timedelta(days=1), periods=n_needed, freq='D')
    elif df.index.name == 'Time':
        last_time = df.index[-1]
        syn_df.index = pd.date_range(start=last_time + pd.Timedelta(days=1), periods=n_needed, freq='D')
        syn_df.index.name = 'Time'

    return pd.concat([df, syn_df], ignore_index=True)


def create_sample_data():
    """Generate sample vibration analytics data for demo purposes"""
    np.random.seed(42)
    days = 730  # 2 years of daily data
    time_idx = pd.date_range(start='2023-01-01', periods=days, freq='D')
    t = np.arange(days)

    # Simulate gradual degradation with seasonal variation
    base_vib_y = 2.5 + 0.002 * t + 0.3 * np.sin(2 * np.pi * t / 365)
    base_vib_x = 2.0 + 0.0015 * t + 0.25 * np.sin(2 * np.pi * t / 365 + 0.5)
    speed = 3000 - 0.5 * t + np.random.normal(0, 15, days)
    temp = 45 + 0.01 * t + 5 * np.sin(2 * np.pi * t / 365) + np.random.normal(0, 1.5, days)
    current = 15 + 0.005 * t + np.random.normal(0, 0.5, days)

    df = pd.DataFrame({
        'Time': time_idx,
        'VYI': base_vib_y + np.random.normal(0, 0.15, days),
        'VXI': base_vib_x + np.random.normal(0, 0.12, days),
        'Speed': speed,
        'T': temp,
        'Current': current
    })
    return df


# ============================================================
# Phase 2: LSTM Model
# ============================================================

def build_lstm_model(n_features, seq_length=30):
    """Build Multivariate LSTM for predictive maintenance"""
    try:
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
        from tensorflow.keras.optimizers import Adam

        model = Sequential([
            Bidirectional(LSTM(64, return_sequences=True, input_shape=(seq_length, n_features))),
            Dropout(0.2),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(n_features)
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        return model
    except Exception:
        return None


def prepare_sequences(data, seq_length=30):
    """Convert time series data to supervised learning sequences"""
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i - seq_length:i])
        y.append(data[i])
    return np.array(X), np.array(y)


def train_and_forecast(df, feature_cols, forecast_years=10, seq_length=30, epochs=30):
    """
    Train LSTM model and generate multi-year forecast.
    Returns: forecast_df, scaler, training_history, is_fallback
    """
    data = df[feature_cols].values
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)

    X, y = prepare_sequences(scaled, seq_length)
    if len(X) < 10:
        return None, scaler, None, False

    model = build_lstm_model(len(feature_cols), seq_length)

    # Fallback to simple trend if TensorFlow unavailable
    if model is None:
        return _fallback_forecast(df, feature_cols, forecast_years, scaler), scaler, None, True

    history = model.fit(X, y, epochs=epochs, batch_size=16, validation_split=0.15, verbose=0)

    # Generate forecast
    forecast_days = forecast_years * 365
    current_seq = scaled[-seq_length:].copy()
    predictions = []

    for _ in range(forecast_days):
        pred = model.predict(current_seq.reshape(1, seq_length, len(feature_cols)), verbose=0)
        predictions.append(pred[0])
        current_seq = np.vstack([current_seq[1:], pred])

    forecast_scaled = np.array(predictions)
    forecast_vals = scaler.inverse_transform(forecast_scaled)

    if 'Time' in df.columns:
        last_date = pd.to_datetime(df['Time'].iloc[-1])
    else:
        last_date = pd.Timestamp.now()

    forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days, freq='D')
    forecast_df = pd.DataFrame(forecast_vals, columns=feature_cols, index=forecast_dates)
    forecast_df.index.name = 'Time'

    return forecast_df, scaler, history, False


def _fallback_forecast(df, feature_cols, forecast_years, scaler):
    """Simple linear trend fallback when TensorFlow is unavailable"""
    forecast_days = forecast_years * 365
    n = len(df)
    t = np.arange(n)
    forecasts = {}

    for col in feature_cols:
        vals = df[col].values
        coeffs = np.polyfit(t, vals, 2)
        future_t = np.arange(n, n + forecast_days)
        forecasts[col] = np.polyval(coeffs, future_t)

    if 'Time' in df.columns:
        last_date = pd.to_datetime(df['Time'].iloc[-1])
    else:
        last_date = pd.Timestamp.now()

    forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days, freq='D')
    forecast_df = pd.DataFrame(forecasts, index=forecast_dates)
    forecast_df.index.name = 'Time'
    return forecast_df


# ============================================================
# RUL & Health Calculations
# ============================================================

def calculate_rul(forecast_df, col, threshold):
    """Calculate Remaining Useful Life - when signal crosses threshold"""
    if forecast_df is None:
        return None, None
    vals = forecast_df[col].values
    crossings = np.where(vals >= threshold)[0]
    if len(crossings) > 0:
        rul_days = crossings[0]
        rul_date = forecast_df.index[crossings[0]]
        return rul_days, rul_date
    return None, None


def calculate_health_score(current_vals, thresholds):
    """
    Calculate equipment health score (0-100%).
    Based on how close current readings are to critical thresholds.
    """
    scores = []
    for val, thresh in zip(current_vals, thresholds):
        if thresh > 0:
            ratio = val / thresh
            score = max(0, min(100, (1 - ratio) * 100 + 20))
            scores.append(score)
    return np.mean(scores) if scores else 50.0


def generate_ai_insights(health_score, rul_days_vy, rul_days_vx, forecast_df, feature_cols):
    """Generate human-readable AI diagnostic report"""
    insights = []

    # Health status
    if health_score >= 80:
        insights.append("✅ **Overall Status: GOOD** — Equipment operating within normal parameters.")
    elif health_score >= 60:
        insights.append("⚠️ **Overall Status: CAUTION** — Early signs of degradation detected.")
    elif health_score >= 40:
        insights.append("🔶 **Overall Status: WARNING** — Significant wear detected. Plan maintenance.")
    else:
        insights.append("🔴 **Overall Status: CRITICAL** — Immediate inspection required!")

    # RUL insights
    if rul_days_vy is not None:
        years = rul_days_vy / 365
        if years < 1:
            insights.append(f"🚨 **Y-Axis Vibration** will hit critical limit in **{rul_days_vy} days** ({years:.1f} years). Urgent bearing inspection needed.")
        elif years < 3:
            q = ((rul_days_vy % 365) // 91) + 1
            y = pd.Timestamp.now().year + int(years)
            insights.append(f"⚠️ **Y-Axis Vibration** projected to reach limit by **Q{q} {y}**. Schedule bearing replacement.")
        else:
            insights.append(f"📊 **Y-Axis Vibration** expected to remain within limits for **{years:.1f} years**.")
    else:
        insights.append("📊 **Y-Axis Vibration** — No critical threshold breach predicted in forecast window.")

    if rul_days_vx is not None:
        years = rul_days_vx / 365
        if years < 2:
            insights.append(f"⚠️ **X-Axis Vibration** approaching limit in **{years:.1f} years**. Monitor closely.")

    # Trend analysis
    if forecast_df is not None and 'T' in feature_cols and 'T' in forecast_df.columns:
        temp_trend = forecast_df['T'].iloc[-1] - forecast_df['T'].iloc[0]
        if temp_trend > 10:
            insights.append("🌡️ **Temperature** shows rising trend. Check cooling system and lubrication.")

    if forecast_df is not None and 'Current' in feature_cols and 'Current' in forecast_df.columns:
        curr_trend = forecast_df['Current'].iloc[-1] - forecast_df['Current'].iloc[0]
        if curr_trend > 5:
            insights.append("⚡ **Current draw** increasing — possible winding degradation or mechanical load increase.")

    # Maintenance recommendations
    if health_score < 70:
        insights.append("\n**🔧 Recommended Actions:**")
        insights.append("1. Perform detailed vibration spectrum analysis")
        insights.append("2. Check bearing condition and lubrication")
        insights.append("3. Inspect shaft alignment and balance")
        insights.append("4. Review motor current signature for anomalies")

    return insights
