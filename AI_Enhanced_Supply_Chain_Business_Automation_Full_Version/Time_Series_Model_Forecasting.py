import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet

def clean_data(data):
  num_data=data.dtypes(includes=['int64','float64'])
  cat_data=data.dtypes(includes=['objects'])
  num_impute=SimpleImputer(strategy='mean')
  cat_impute=SimpleImputer(strategy='most_frequent')
  data[num_data]=num_fit.transform(data[num_data])
  data[cat_data]=cat.fit_transform(data[cat_data])
  return data

def preprocess_data(data):
  lab=LabelEncoder()
  std=StandardScaler()
  data[num_data]=std.fit_transform(data[num_data])
  data[cat_data]=lab.fit_transform(data[cat_data])
  return data

def split_time_series_for_forecasting(data, date_col, value_col, test_ratio=0.2, freq='D'):

    # Preparing and clean the dataset
    data = data.copy()
    data[date_col] = pd.to_datetime(data[date_col])
    data = data.set_index(date_col)
    data = data.asfreq(freq)
    data[value_col] = data[value_col].interpolate()

    # splitting
    split_index = int(len(data) * (1 - test_ratio))
    train = data.iloc[:split_index]
    test = data.iloc[split_index:]

    # series
    train_series = train[value_col]
    test_series = test[value_col]

    # For Prophet (DataFrame with 'ds' and 'y')
    prophet_df = data[[value_col]].reset_index().rename(columns={date_col: 'ds', value_col: 'y'})
    prophet_train = prophet_df.iloc[:split_index]
    prophet_test = prophet_df.iloc[split_index:]

    return train_series, test_series, prophet_train, prophet_test

arima_model = ARIMA(train_series, order=(5, 1, 2))
arima_fit = arima_model.fit()

# Forecast
arima_forecast = arima_fit.forecast(steps=len(test_series))
arima_forecast.index = test_series.index

# Plot results
plt.figure(figsize=(10, 4))
plt.plot(train_series, label='Train')
plt.plot(test_series, label='Test')
plt.plot(arima_forecast, label='ARIMA Forecast')
plt.legend()
plt.title("ARIMA Forecast")
plt.show()

sarima_model = SARIMAX(train_series, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))  # (p,d,q)(P,D,Q,s)
sarima_fit = sarima_model.fit()

# Forecast
sarima_forecast = sarima_fit.forecast(steps=len(test_series))
sarima_forecast.index = test_series.index

# Plot results
plt.figure(figsize=(10, 4))
plt.plot(train_series, label='Train')
plt.plot(test_series, label='Test')
plt.plot(sarima_forecast, label='SARIMA Forecast')
plt.legend()
plt.title("SARIMA Forecast")
plt.show()

prophet_model = Prophet(daily_seasonality=True)
prophet_model.fit(prophet_train)  # Prophet requires columns: ds (date), y (value)

# Forecast future
future = prophet_model.make_future_dataframe(periods=len(prophet_test))
forecast = prophet_model.predict(future)

# Plot forecast
fig1 = prophet_model.plot(forecast)
plt.title("Prophet Forecast")
plt.show()

# Compare yhat with actual test values
forecast_with_actual = forecast[['ds', 'yhat']].set_index('ds').join(
    prophet_test.set_index('ds')
)

forecast_with_actual[['y', 'yhat']].plot(figsize=(10, 4), title="Prophet vs Actual")
plt.show()
