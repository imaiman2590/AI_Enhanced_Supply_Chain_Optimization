import os
import re
import pytesseract
import cv2
import pdfplumber
import pandas as pd
import numpy as np
from email import policy
from email.parser import BytesParser
from transformers import pipeline
from joblib import dump, load
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from keras.models import Sequential
from keras.layers import Dense, Dropout, GRU, LSTM
from keras.optimizers import Adam
from keras.wrappers.scikit_learn import KerasClassifier
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
import matplotlib.pyplot as plt
import warnings
import streamlit as st
import os
import tempfile
import plotly.graph_objects as go
warnings.filterwarnings("ignore")

# Define supported file types
SUPPORTED_FILE_TYPES = ['.pdf', '.jpg', '.jpeg', '.png', '.eml']

# Load transformer-based NER model
ner_pipeline = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english",
                        tokenizer="dbmdz/bert-large-cased-finetuned-conll03-english")

# Base Extractor Class
class BaseExtractor:
    def extract_text(self, file_path):
        raise NotImplementedError

class PDFExtractor(BaseExtractor):
    def extract_text(self, file_path):
        with pdfplumber.open(file_path) as pdf:
            return "\n".join([page.extract_text() or "" for page in pdf.pages])

class ImageExtractor(BaseExtractor):
    def extract_text(self, file_path):
        img = cv2.imread(file_path)
        return pytesseract.image_to_string(img)

class EmailExtractor(BaseExtractor):
    def extract_text(self, file_path):
        with open(file_path, 'rb') as f:
            msg = BytesParser(policy=policy.default).parse(f)
        return msg.get_body(preferencelist=('plain')).get_content()

def classify_document(text):
    text_lower = text.lower()
    if "invoice" in text_lower:
        return "invoice"
    elif "purchase order" in text_lower or "po number" in text_lower:
        return "purchase_order"
    elif "bill of lading" in text_lower:
        return "shipping_doc"
    return "unknown"

def extract_order_data(text):
    supplier_name = extract_supplier_name(text)
    patterns = {
        "order_number": r"(PO Number|Order ID):?\s*(\w+)",
        "total": r"Total Amount:?\s*\$?([\d,]+\.\d{2})"
    }
    data = {
        "order_number": match_pattern(text, patterns["order_number"], group=2),
        "supplier": supplier_name,
        "total": match_pattern(text, patterns["total"])
    }
    return data

def match_pattern(text, pattern, group=1):
    match = re.search(pattern, text)
    return match.group(group) if match else None

def extract_supplier_name(text):
    ner_results = ner_pipeline(text)
    for entity in ner_results:
        if entity['entity'] in ['B-ORG', 'I-ORG']:
            return entity['word']
    return None

def validate_data(data):
    errors = []
    if not data.get("order_number"):
        errors.append("Missing order number")
    if not data.get("supplier"):
        errors.append("Missing supplier name")
    if not data.get("total"):
        errors.append("Missing total amount")
    return errors

def split_time_series_for_forecasting(data, date_col, value_col, test_ratio=0.2, freq='D'):

    # Prepare and clean
    data = data.copy()
    data[date_col] = pd.to_datetime(data[date_col])
    data = data.set_index(date_col)
    data = data.asfreq(freq)
    data[value_col] = data[value_col].interpolate()

    # Chronological split
    split_index = int(len(data) * (1 - test_ratio))
    train = data.iloc[:split_index]
    test = data.iloc[split_index:]

    # For ARIMA/SARIMA (pandas Series)
    train_series = train[value_col]
    test_series = test[value_col]

    # For Prophet (DataFrame with 'ds' and 'y')
    prophet_df = data[[value_col]].reset_index().rename(columns={date_col: 'ds', value_col: 'y'})
    prophet_train = prophet_df.iloc[:split_index]
    prophet_test = prophet_df.iloc[split_index:]

    return train_series, test_series, prophet_train, prophet_test

## Forecasting Techniques

# ARIMA
arima_model = ARIMA(train_series, order=(5, 1, 2))
arima_fit = arima_model.fit()
arima_forecast = arima_fit.forecast(steps=len(test_series))
plt.figure(figsize=(10, 4))
plt.plot(train_series, label='Train')
plt.plot(test_series, label='Test')
plt.plot(test_series.index, arima_forecast, label='ARIMA Forecast')
plt.legend()
plt.title("ARIMA Forecast")
plt.show()

# SARIMA
sarima_model = SARIMAX(train_series, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
sarima_fit = sarima_model.fit()
sarima_forecast = sarima_fit.forecast(steps=len(test_series))
plt.figure(figsize=(10, 4))
plt.plot(train_series, label='Train')
plt.plot(test_series, label='Test')
plt.plot(test_series.index, sarima_forecast, label='SARIMA Forecast')
plt.legend()
plt.title("SARIMA Forecast")
plt.show()

# Prophet
prophet_train = pd.DataFrame({'ds': train_series.index, 'y': train_series.values})
prophet_test = pd.DataFrame({'ds': test_series.index, 'y': test_series.values})
prophet_model = Prophet(daily_seasonality=True)
prophet_model.fit(prophet_train)
future = prophet_model.make_future_dataframe(periods=len(prophet_test))
forecast = prophet_model.predict(future)
prophet_model.plot(forecast)
plt.title("Prophet Forecast")
plt.show()

forecast_with_actual = forecast[['ds', 'yhat']].set_index('ds').join(
    prophet_test.set_index('ds'))
forecast_with_actual[['y', 'yhat']].plot(figsize=(10, 4), title="Prophet vs Actual")
plt.show()

def split_data(data,target):
  x=data.drop(target)
  y=data[target]
  return x,y

def split_train_test_dataset(x,y):
  x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
  return x_train,x_test,y_train,y_test

def tune_ml_model(x_train, y_train,x_test,y_test):
    rf = RandomForestClassifier()
    lr = LogisticRegression()
    xgb = XGBClassifier()

    param_grid_rf = {'n_estimators': [100, 200], 'max_depth': [10, 20]}
    param_grid_lr = {'C': [0.1, 1, 10], 'solver': ['liblinear']}
    param_grid_xgb = {'max_depth': [6, 7], 'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1]}

    grid_rf = GridSearchCV(rf, param_grid_rf, cv=3)
    grid_lr = GridSearchCV(lr, param_grid_lr, cv=3)
    grid_xgb = GridSearchCV(xgb, param_grid_xgb, cv=3)

    grid_rf.fit(x_test, y_test)
    grid_lr.fit(x_test, y_test)
    grid_xgb.fit(x_test, y_test)

    return grid_rf.best_estimator_, grid_lr.best_estimator_, grid_xgb.best_estimator_

def forecasting_neural_model(x_train, y_train, x_test, y_test, input_shape):
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    checkpoint = ModelCheckpoint('best_model.keras', save_best_only=True)
    model = Sequential()
    model.add(GRU(units=128, activation='relu', return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(GRU(units=64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(units=32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(units=1, activation='linear'))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    history = model.fit(x_train, y_train, batch_size=32, epochs=100,
                        validation_split=0.2, callbacks=[early_stop, checkpoint])
    test_loss, test_mae = model.evaluate(x_test, y_test, verbose=0)
    return test_loss, test_mae, model, history

##Inventory Optimization

def create_sequences(data, seq_length=10, target_index=0):
    x, y = [], []
    for i in range(len(data) - seq_length):
        x.append(data[i:i + seq_length, :])
        y.append(data[i + seq_length, target_index])
    return np.array(x), np.array(y)


def split_data(data, target):
    x = data.drop(target, axis=1)
    y = data[target]
    return train_test_split(x, y, test_size=0.2, random_state=42)

def build_nn_model(input_dim=3):
    model = Sequential()
    model.add(Dense(128, input_dim=input_dim, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_model(model, x_train, y_train):
    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    history = model.fit(
        x_train, y_train,
        batch_size=32,
        epochs=50,
        validation_split=0.2,
        callbacks=[early_stop]
    )
    return model, history


def validate_model(model, x_test, y_test):
    loss, accuracy = model.evaluate(x_test, y_test)
    return loss, accuracy

##Anonymal Detection.

def create_sequences(values, time_steps=30):
    output = []
    for i in range(len(values) - time_steps):
        output.append(values[i : i + time_steps])
    return np.stack(output)

TIME_STEPS = 30
sequences = create_sequences(df['scaled'].values, TIME_STEPS)

model_2 = Sequential([
    LSTM(64, activation='relu', input_shape=(TIME_STEPS, 1), return_sequences=False),
    RepeatVector(TIME_STEPS),
    LSTM(64, activation='relu', return_sequences=True),
    TimeDistributed(Dense(1))
])

model_2.compile(optimizer='adam', loss='mse')
X_train = sequences.reshape((sequences.shape[0], sequences.shape[1], 1))
model.fit(X_train, X_train, epochs=20, batch_size=32, validation_split=0.1, verbose=1)
X_pred = model.predict(X_train)
mse = np.mean(np.power(X_train - X_pred, 2), axis=(1, 2))

# Set threshold for anomaly
threshold = np.percentile(mse, 95)
print(f"Anomaly threshold: {threshold:.6f}")

# Anomalies
anomalies = mse > threshold

input_dim = X_scaled.shape[1]
encoding_dim = input_dim // 2

# Define architecture
input_layer = Input(shape=(input_dim,))
encoded = Dense(encoding_dim, activation="relu", activity_regularizer=regularizers.l1(1e-5))(input_layer)
decoded = Dense(input_dim, activation="linear")(encoded)

autoencoder = Model(inputs=input_layer, outputs=decoded)
autoencoder.compile(optimizer='adam', loss='mse')

autoencoder.summary()

history = autoencoder.fit(
    X_scaled, X_scaled,
    epochs=50,
    batch_size=32,
    shuffle=True,
    validation_split=0.1,
    verbose=1
)

# Reconstruct the data
X_pred = autoencoder.predict(X_scaled)
mse = np.mean(np.power(X_scaled - X_pred, 2), axis=1)

# Set threshold (95th percentile of MSE)
threshold = np.percentile(mse, 95)
print(f"Anomaly Threshold: {threshold:.6f}")

# Identify anomalies
df['reconstruction_error'] = mse
df['anomaly'] = df['reconstruction_error'] > threshold

def decision_engine(doc_type, data, errors):
    if errors:
        return "Flag for Review", errors
    return "Auto-Approve", []

def send_to_erp(data, decision):
    print(f"Sending to ERP: {data}, Decision: {decision}")

def get_extractor(file_ext):
    return {
        ".pdf": PDFExtractor(),
        ".jpg": ImageExtractor(),
        ".jpeg": ImageExtractor(),
        ".png": ImageExtractor(),
        ".eml": EmailExtractor()
    }.get(file_ext, None)

def process_file(file_path):
    ext = os.path.splitext(file_path)[-1].lower()
    if ext not in SUPPORTED_FILE_TYPES:
        return
    extractor = get_extractor(ext)
    if not extractor:
        return
    try:
        text = extractor.extract_text(file_path)
        doc_type = classify_document(text)
        data = extract_order_data(text)
        errors = validate_data(data)
        decision, notes = decision_engine(doc_type, data, errors)
        send_to_erp(data, decision)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

## Customer and Product Segmentation
def feature_engineering_and_clustering(df):
    customer_df = df.groupby('customer_id').agg({
        'order_amount': ['mean', 'sum', 'count'],
        'date': lambda x: (pd.to_datetime(x).max() - pd.to_datetime(x).min()).days
    }).reset_index()
    customer_df.columns = ['customer_id', 'avg_order', 'total_order', 'order_count', 'active_days']
    features = customer_df.drop(columns='customer_id')
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)
    customer_df['kmeans_cluster'] = KMeans(n_clusters=4, random_state=42).fit_predict(X_scaled)
    customer_df['dbscan_cluster'] = DBSCAN(eps=1.2, min_samples=5).fit_predict(X_scaled)
    return customer_df

#Streamlit UI
st.title("📄 Intelligent Document Processing & Forecasting")

st.sidebar.header("Upload a File")
uploaded_file = st.sidebar.file_uploader("Choose a file", type=['pdf', 'jpg', 'jpeg', 'png', 'eml'])

if uploaded_file:
    ext = os.path.splitext(uploaded_file.name)[-1].lower()
    if ext not in SUPPORTED_FILE_TYPES:
        st.error("Unsupported file type!")
    else:
        # Save to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(uploaded_file.read())
            file_path = tmp.name

        # Extract text
        extractor = get_extractor(ext)
        if extractor:
            try:
                text = extractor.extract_text(file_path)
                st.subheader("📜 Extracted Text")
                st.text_area("Text Content", value=text, height=300)

                doc_type = classify_document(text)
                data = extract_order_data(text)
                errors = validate_data(data)
                decision, notes = decision_engine(doc_type, data, errors)

                st.subheader("📁 Document Info")
                st.write(f"**Document Type:** {doc_type}")
                st.write("**Extracted Data:**", data)
                st.write("**Validation Errors:**", errors or "None")
                st.write("**Decision:**", decision)
                if notes:
                    st.write("**Notes:**", notes)

                # Simulate ERP push
                send_to_erp(data, decision)

            except Exception as e:
                st.error(f"Error: {e}")

st.markdown("---")
st.subheader("📊 Forecasting (Static Example)")

if st.button("Show Forecast Charts"):
    from your_module import train_series, test_series, arima_forecast, sarima_forecast, forecast_with_actual

    # ARIMA Forecast
    fig_arima = go.Figure()
    fig_arima.add_trace(go.Scatter(x=train_series.index, y=train_series, mode='lines', name='Train'))
    fig_arima.add_trace(go.Scatter(x=test_series.index, y=test_series, mode='lines', name='Test'))
    fig_arima.add_trace(go.Scatter(x=test_series.index, y=arima_forecast, mode='lines', name='ARIMA Forecast'))
    fig_arima.update_layout(title='ARIMA Forecast')
    st.plotly_chart(fig_arima, use_container_width=True)

    # SARIMA Forecast
    fig_sarima = go.Figure()
    fig_sarima.add_trace(go.Scatter(x=train_series.index, y=train_series, mode='lines', name='Train'))
    fig_sarima.add_trace(go.Scatter(x=test_series.index, y=test_series, mode='lines', name='Test'))
    fig_sarima.add_trace(go.Scatter(x=test_series.index, y=sarima_forecast, mode='lines', name='SARIMA Forecast'))
    fig_sarima.update_layout(title='SARIMA Forecast')
    st.plotly_chart(fig_sarima, use_container_width=True)

    # Prophet Forecast
    fig_prophet = go.Figure()
    fig_prophet.add_trace(go.Scatter(x=forecast_with_actual.index, y=forecast_with_actual['y'], mode='lines', name='Actual'))
    fig_prophet.add_trace(go.Scatter(x=forecast_with_actual.index, y=forecast_with_actual['yhat'], mode='lines', name='Prophet Forecast'))
    fig_prophet.update_layout(title='Prophet Forecast vs Actual')
    st.plotly_chart(fig_prophet, use_container_width=True)

st.markdown("---")
st.subheader("🔍 Customer Segmentation (Static Example)")

if st.button("Run Clustering"):
    df = pd.DataFrame({
        'customer_id': [1, 2, 3],
        'order_amount': [100, 200, 300],
        'date': ['2025-01-01', '2025-02-01', '2025-03-01'],
    })
    customer_df = feature_engineering_and_clustering(df)
    st.dataframe(customer_df)
