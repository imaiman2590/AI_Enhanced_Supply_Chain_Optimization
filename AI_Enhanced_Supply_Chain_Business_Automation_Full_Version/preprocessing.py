import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin

# Custom Transformer for converting categorical strings to numerical labels
class LabelEncoderWrapper(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.encoders = {}

    def fit(self, X, y=None):
        for col in X.columns:
            le = LabelEncoder()
            le.fit(X[col].astype(str))
            self.encoders[col] = le
        return self

    def transform(self, X):
        X_out = X.copy()
        for col in X.columns:
            X_out[col] = self.encoders[col].transform(X_out[col].astype(str))
        return X_out

# Main Preprocessing Function
def preprocess_data(data):
    # Identify feature types
    num_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
    text_cols = [col for col in cat_cols if data[col].apply(lambda x: isinstance(x, str)).all() and data[col].str.len().mean() > 20]

    # Remove text_cols from cat_cols
    cat_cols = list(set(cat_cols) - set(text_cols))

    # Pipelines
    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('label_encoder', LabelEncoderWrapper())
    ])

    text_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='')),
        ('tfidf', TfidfVectorizer(max_features=100))  # You can tune max_features
    ])

    # ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_pipeline, num_cols),
            ('cat', categorical_pipeline, cat_cols)
        ],
        remainder='drop'
    )

    # Fit and transform numeric/categorical
    processed_data = preprocessor.fit_transform(data)

    # Handle text separately because TfidfVectorizer returns sparse matrix
    tfidf_features = []
    for col in text_cols:
        vectorizer = text_pipeline.fit(data[col])
        tfidf_matrix = vectorizer.transform(data[col])
        tfidf_features.append(tfidf_matrix)

    # Combine all features
    if tfidf_features:
        from scipy.sparse import hstack
        final_data = hstack([processed_data] + tfidf_features)
    else:
        final_data = processed_data

    return processdata
