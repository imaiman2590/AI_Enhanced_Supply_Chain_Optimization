import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM,GRU,Dropout,RepeatVector, TimeDistributed
from tensorflow.keras.callbacks import EarlyStopping
import pytesseract
import os
import re
import cv2
import pdfplumber
from email import policy
from email.parser import BytesParser
from transformers import pipeline
from joblib import load, dump
from keras.optimizers import Adam
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import accuracy_score
import re
import nltk
import spacy
from nltk.corpus import stopwords
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

SUPPORTED_FILE_TYPES = ['.pdf', '.jpg', '.jpeg', '.png', '.eml']

def import_dataset(file_path):
  if file_path.endswith('.csv'):
    return pd.read_csv(file_path)
  elif file_path.endswith('.xlsx'):
    return pd.read_excel(file_path)
  elif file_path.endswith('.json'):
    return pd.read_json(file_path)
  else:
    return None

# Load Transformer-based NER model from Hugging Face
ner_pipeline = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english", tokenizer="dbmdz/bert-large-cased-finetuned-conll03-english")

# ------------------------- #
# Abstract Base Extractor   #
# ------------------------- #
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

# ---------------------------- #
# Document Classifier          #
# ---------------------------- #
def classify_document(text):
    text_lower = text.lower()
    if "invoice" in text_lower:
        return "invoice"
    elif "purchase order" in text_lower or "po number" in text_lower:
        return "purchase_order"
    elif "bill of lading" in text_lower:
        return "shipping_doc"
    return "unknown"

# ------------------------ #
# Data Extraction          #
# ------------------------ #
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
        if entity['entity'] == 'B-ORG' or entity['entity'] == 'I-ORG':
            return entity['word']
    return None

# ------------------------ #
# Validation Rules         #
# ------------------------ #
def validate_data(data):
    errors = []
    if not data.get("order_number"):
        errors.append("Missing order number")
    if not data.get("supplier"):
        errors.append("Missing supplier name")
    if not data.get("total"):
        errors.append("Missing total amount")
    return errors

#-----------------------
# Cleaning part
#-----------------------

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
nlp = spacy.load('en_core_web_sm')

def clean_text(text):
    if not isinstance(text, str):
        return ""

    # Lowercase and remove non-alphabetic characters
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    # Tokenize and remove stopwords
    tokens = nltk.word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]

    # Lemmatize using spaCy
    doc = nlp(" ".join(tokens))
    lemmatized = [token.lemma_ for token in doc if not token.is_stop]

    return " ".join(lemmatized)

def clean_data(data):
    data = data.copy()

    # Identify numeric, categorical, and text columns
    num_cols = data.select_dtypes(include=['int64', 'float64']).columns
    cat_cols = data.select_dtypes(include=['object', 'category']).columns

    # Detect long text columns (text-like)
    text_cols = [col for col in cat_cols if data[col].apply(lambda x: isinstance(x, str)).mean() > 0.9 and data[col].str.len().mean() > 20]
    cat_cols = [col for col in cat_cols if col not in text_cols]

    # Impute and scale numeric
    num_imputer = SimpleImputer(strategy='mean')
    data[num_cols] = num_imputer.fit_transform(data[num_cols])

    # Impute and encode categorical
    cat_imputer = SimpleImputer(strategy='most_frequent')
    data[cat_cols] = cat_imputer.fit_transform(data[cat_cols])

    for col in cat_cols:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col].astype(str))

    # Clean text columns
    for col in text_cols:
        data[col] = data[col].apply(clean_text)

    return data

