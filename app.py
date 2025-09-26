"""
Streamlit ARIMA Traffic Forecasting App
======================================

This single-file Streamlit app lets you:
 - Upload your daily toll/traffic CSV (Date, Total_Vehicles, Toll_Revenue, etc.)
 - Preview and clean data (parse dates, set index)
 - Fit an ARIMA model (auto selection with pmdarima or manual p,d,q)
 - Forecast next N days (traffic or revenue)
 - View evaluation metrics, interactive plots, and download forecast CSV

Deployment
----------
1. Create a new GitHub repo and push this file (streamlit_arima_app.py) to the repo root.
2. Add a requirements.txt file (example below) to the repo root.
3. Sign in to https://streamlit.io/cloud and connect your GitHub repo. Create a new app, choose the branch and file (streamlit_arima_app.py). Streamlit Cloud will install requirements and deploy automatically.

Example requirements.txt
------------------------
streamlit
pandas
numpy
matplotlib
pmdarima
statsmodels
scikit-learn
openpyxl

(Optional) If you prefer deploying elsewhere (Heroku, Docker), ask and I will provide a Dockerfile / Procfile / GitHub Actions workflow.

Notes
-----
- pmdarima (auto_arima) is convenient; if you run into build issues on the host, use statsmodels.ARIMA/SARIMAX with manual selection.
- This app expects one row per day. If you have multiple timestamps per day, aggregate to daily sums.

"""

import streamlit as st
import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tempfile

# Try imports that may not be present in some environments
try:
    from pmdarima import auto_arima
    PMDARIMA_AVAILABLE = True
except Exception:
    PMDARIMA_AVAILABLE = False

st.set_page_config(page_title="Traffic Forecasting (ARIMA)", layout="wide")

st.title("🚦 Traffic & Toll Forecasting — ARIMA (daily)")
st.markdown(
    """
    Upload a daily CSV (Date column + one numeric target column like Total_Vehicles or Toll_Revenue).

    Features:
    - Auto-ARIMA (if pmdarima is installed) or manual ARIMA
    - Forecast horizon selection
    - Model evaluation on a holdout
    - Downloadable forecast CSV
    """
)

# -----------------
# Sidebar controls
# -----------------
st.sidebar.header("Data & Model Settings")
uploaded = st.sidebar.file_uploader("Upload CSV (or leave blank to use sample data)", type=["csv", "xlsx"])
use_sample = False
if uploaded is None:
    use_sample = st.sidebar.checkbox("Use sample synthetic data (5 years daily)", value=True)

date_col = st.sidebar.text_input("Date column name", value="Date")

# Forecast settings
horizon = st.sidebar.number_input("Forecast horizon (days)", min_value=1, max_value=365, value=7)
train_frac = st.sidebar.slider("Train fraction (rest used as test)", min_value=0.5, max_value=0.95, value=0.9)

st.sidebar.markdown("---")
if PMDARIMA_AVAILABLE:
    st.sidebar.success("pmdarima available: auto_arima enabled")
else:
    st.sidebar.warning("pmdarima not available: auto selection disabled — use manual p,d,q")

# Manual ARIMA parameters (if user chooses)
use_auto = st.sidebar.checkbox("Use auto_arima (if available)", value=PMDARIMA_AVAILABLE)
st.sidebar.markdown("**Manual ARIMA (ignored if auto used)**")
p = st.sidebar.number_input("p (AR order)", min_value=0, max_value=10, value=1)
d = st.sidebar.number_input("d (difference)", min_value=0, max_value=3, value=1)
q = st.sidebar.number_input("q (MA order)", min_value=0, max_value=10, value=1)

# -----------------
# Load data
# -----------------
@st.cache_data
def load_sample_data():
    rng = pd.date_range(end=pd.Timestamp.today(), periods=5*365, freq="D")
    # Create a seasonal + trend + noise series
    seasonal = 2000 + 300 * np.sin(2 * np.pi * rng.dayofyear / 365)
    trend = np.linspace(0, 800, len(rng))
    noise = np.random.normal(0, 150, len(rng))
    series = np.round(seasonal + trend + noise).astype(int)
    df = pd.DataFrame({"Date": rng, "Total_Vehicles": series})
    return df

@st.cache_data
def read_csv(/home/it/Desktop/rutuja pore/toll ):
    # Supports CSV or Excel via uploaded object
    try:
        if uploaded_file.name.endswith('.xlsx') or uploaded_file.name.endswith('.xls'):
            df = pd.read_excel(uploaded_file)
        else:
            df = pd.read_csv(uploaded_file)
    except Exception:
        uploaded_file.seek(0)
        df = pd.read_csv(io.StringIO(uploaded_file.getvalue().decode("utf-8")))

    # Ensure date column exists
    if date_col not in df.columns:
        # Try common alternatives
        for alt in ["date", "DATE", "Day"]:
            if alt in df.columns:
                df = df.rename(columns={alt: date_col})
                break
    df[date_col] = pd.to_datetime(df[date_col])
    return df

if use_sample and uploaded is None:
    df = load_sample_data()
else:
    if uploaded is None:
        st.info("Please upload a CSV or use sample data.")
        st.stop()
    df = read_csv(uploaded, date_col)

# Show dataframe and let user pick target column
st.subheader("Preview data")
with st.expander("Data snapshot (first 10 rows)"):
    st.dataframe(df.head(10))

# Ensure date parse
if date_col not in df.columns:
    st.error(f"Date column '{date_col}' not found after parsing. Check your file and column name.")
    st.stop()

# Convert to daily index
df[date_col] = pd.to_datetime(df[date_col])
df = df.sort_values(date_col).drop_duplicates(subset=[date_col])
df.set_index(date_col, inplace=True)

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if not numeric_cols:
    st.error("No numeric columns found to forecast. Ensure your CSV contains at least one numeric column (e.g., Total_Vehicles).")
    st.stop()

target_col = st.selectbox("Select target column to forecast", numeric_cols)

# Optional aggregation if multiple entries per day
if df.index.has_duplicates:
    st.warning("Multiple rows share the same date — aggregating by sum for each day.")
    df = df.groupby(df.index).sum()

series = df[target_col].asfreq('D')  # fill missing days with NaN

st.write(f"Data from **{series.index.min().date()}** to **{series.index.max().date()}** — {len(series)} days")

# Fill small gaps by interpolation (user can modify this part as needed)
series_filled = series.interpolate()

# Train/test split
split_idx = int(len(series_filled) * train_frac)
train = series_filled.iloc[:split_idx]
test = series_filled.iloc[split_idx:]

st.subheader("Modeling")
st.write(f"Using {len(train)} days for training and {len(test)} days for testing (if any).")

# Fit model
model = None
fitted = False

if use_auto and PMDARIMA_AVAILABLE:
    st.write("Fitting auto_arima (this may take a few moments)...")
    with st.spinner("Running auto_arima..."):
        try:
            model = auto_arima(train, seasonal=False, error_action='ignore', suppress_warnings=True, stepwise=True)
            fitted = True
            st.success("auto_arima finished")
            st.text(str(model.summary()))
        except Exception as e:
            st.error(f"auto_arima failed: {e}")
            fitted = False

if not fitted:
    st.write("Fitting manual ARIMA with statsmodels (p,d,q) = (%d,%d,%d)" % (p, d, q))
    try:
        from statsmodels.tsa.arima.model import ARIMA
        with st.spinner("Training ARIMA..."):
            sm_model = ARIMA(train, order=(p, d, q)).fit()
            model = sm_model
            fitted = True
            st.success("ARIMA fitted")
            st.text(sm_model.summary().as_text())
    except Exception as e:
        st.error(f"Manual ARIMA failed: {e}")
        st.stop()

# Forecasting
st.subheader("Forecast")
if fitted:
    # Forecast horizon includes test length + requested horizon if user wants to evaluate
    n_forecast = int(horizon)

    # If model is pmdarima, .predict returns a numpy array; if statsmodels, use get_forecast
    if PMDARIMA_AVAILABLE and use_auto and hasattr(model, 'predict') and not hasattr(model, 'forecast'):
        # pmdarima model
        fc = model.predict(n_periods=n_forecast)
        index = pd.date_range(start=series_filled.index[-1] + pd.Timedelta(days=1), periods=n_forecast, freq='D')
        forecast_series = pd.Series(fc, index=index)
    else:
        # statsmodels ARIMAResults
        try:
            last_date = series_filled.index[-1]
            pred = model.get_forecast(steps=n_forecast)
            fc = pred.predicted_mean
            index = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=n_forecast, freq='D')
            forecast_series = pd.Series(fc.values, index=index)
        except Exception:
            # fallback: use model.forecast
            try:
                fc = model.forecast(steps=n_forecast)
                index = pd.date_range(start=series_filled.index[-1] + pd.Timedelta(days=1), periods=n_forecast, freq='D')
                forecast_series = pd.Series(fc, index=index)
            except Exception as e:
                st.error(f"Forecast failed: {e}")
                st.stop()

    st.write(forecast_series.to_frame(name='Forecast'))

    # Plot historical + forecast
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(series_filled.index, series_filled.values, label='History')
    ax.plot(forecast_series.index, forecast_series.values, linestyle='--', marker='o', label='Forecast')
    ax.set_xlabel('Date')
    ax.set_ylabel(target_col)
    ax.legend()
    st.pyplot(fig)

    # Evaluate on test if available and horizon overlaps
    if len(test) > 0:
        # Compare predictions for the length of test if horizon >= len(test)
        eval_steps = min(len(test), n_forecast)
        # If forecast covers the test period (forecast starts after last train date)
        # we can only evaluate if we produced forecasts into the test range.
        # Create a predictions index matching the first eval_steps days of forecast.
        try:
            y_true = test.iloc[:eval_steps].values
            y_pred = forecast_series.iloc[:eval_steps].values
            mae = mean_absolute_error(y_true, y_pred)
            rmse = mean_squared_error(y_true, y_pred, squared=False)
            st.markdown(f"**Evaluation on test (first {eval_steps} days)** — MAE: {mae:.2f}, RMSE: {rmse:.2f}")
        except Exception as e:
            st.info("Could not evaluate on test set: %s" % e)

    # Provide download
    csv_buffer = io.StringIO()
    out_df = pd.DataFrame({"Date": forecast_series.index, "Forecast": forecast_series.values})
    out_df.to_csv(csv_buffer, index=False)
    st.download_button("Download forecast CSV", data=csv_buffer.getvalue(), file_name="forecast.csv", mime="text/csv")

    # Save model (optional) - we will create a temporary file and offer to download the pickled model
    if st.button("Download fitted model (pickle)"):
        try:
            import pickle
            tmpf = tempfile.NamedTemporaryFile(delete=False, suffix='.pkl')
            with open(tmpf.name, 'wb') as f:
                pickle.dump(model, f)
            with open(tmpf.name, 'rb') as f:
                st.download_button("Download model file", data=f, file_name='fitted_model.pkl')
        except Exception as e:
            st.error(f"Could not pickle model for download: {e}")

else:
    st.error("Model is not fitted — check logs above.")

st.sidebar.markdown("---")
st.sidebar.info("Questions or want a Dockerfile / GitHub Actions workflow? Ask me and I'll add it.")

# Footer
st.markdown("---")
st.caption("Built with ❤️ — modify the code to suit seasonal modeling or multivariate forecasting (SARIMA, Prophet, or deep-learning models).")

