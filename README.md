***AI Enhanced Supply Chain Automation***

---

## 🧾 **1. Document Parsing & Information Extraction**

### ✅ Supports:

* PDF, Image (`.jpg`, `.jpeg`, `.png`), and Email (`.eml`) files.

### ✅ Extracts:

* Raw text using:

  * `pdfplumber` for PDFs
  * `pytesseract` for images
  * Python `email` module for emails

### ✅ Uses BERT-based NER:

* Extracts supplier names from unstructured text using HuggingFace’s BERT model (`dbmdz/bert-large-cased-finetuned-conll03-english`)

### ✅ Classifies documents as:

* `invoice`, `purchase_order`, `shipping_doc`, or `unknown`

### ✅ Extracts key fields like:

* Order number, total amount, supplier name

### ✅ Validates:

* Checks if any required fields are missing

---

## 📈 **2. Forecasting (Time Series)**

### Implements three time series models:

* **ARIMA**
* **SARIMA**
* **Prophet**

### Forecasts:

* Future values based on training/test split on a date-based dataset.

### Visualizes:

* Forecasts using `matplotlib` and `plotly`.

---

## 🧠 **3. Machine Learning Model Tuning**

### ML Models:

* Random Forest
* Logistic Regression
* XGBoost

### GridSearchCV:

* Used for hyperparameter tuning

---

## 🔢 **4. Deep Learning Forecasting**

### GRU-based Neural Network:

* For time series prediction

### Includes:

* Dropout for regularization
* EarlyStopping and ModelCheckpoint callbacks

---

## 🏭 **5. Inventory Optimization / Demand Forecasting**

### Sequence creation for time series

### Feed into:

* Dense Neural Network for binary classification

---

## 🚨 **6. Anomaly Detection**

### Uses:

* LSTM-based sequence reconstruction (Autoencoder)
* Dense Autoencoder with reconstruction error

### Flags anomalies based on:

* Mean squared error threshold (95th percentile)

---

## 🧍‍♂️👥 **7. Customer Segmentation**

### Features:

* `avg_order`, `total_order`, `order_count`, `active_days`

### Uses:

* KMeans
* DBSCAN

---

## 🌐 **8. Streamlit Web Interface**

### Features:

* Upload documents
* See extracted text and parsed fields
* Display validation results
* Run forecasting visualizations
* View customer segmentation (static example)

---

## 🧠 **9. Decision Engine & ERP Integration (Simulated)**

* Automatically determines if a document can be "Auto-Approved" or needs "Review"
* Simulates sending to ERP

---

### ✅ **In Summary**, this project:

* Automates document understanding from various formats
* Extracts and validates structured data
* Supports forecasting and analytics
* Detects anomalies
* Segments customers
* Offers an interactive web UI

---
