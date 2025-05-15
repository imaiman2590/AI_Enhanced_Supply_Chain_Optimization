import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM,GRU,Dropout,RepeatVector, TimeDistributed
from tensorflow.keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import accuracy_score
from preprocessing import processdata

def split_data(data,target):
  x=data.drop(target)
  y=data[target]
  return x,y

def split_train_test_dataset(x,y):
  x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
  return x_train,x_test,y_train,y_test


# ------------------------ #
# Machine Learning Models  #
# ------------------------ #
def build_nn_model(input_dim=3):
    model = Sequential()
    model.add(Dense(128, input_dim=input_dim, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))  # Binary classification
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# ------------------------ #
# Hyperparameter Tuning    #
# ------------------------ #

# ML model hyperparameter tuning using GridSearchCV
def tune_ml_model(X, y):
    rf = RandomForestClassifier()
    lr = LogisticRegression()
    xgb = XGBClassifier()

    param_grid_rf = {'n_estimators': [100, 200], 'max_depth': [10, 20]}
    param_grid_lr = {'C': [0.1, 1, 10], 'solver': ['liblinear']}
    param_grid_xgb = {'max_depth': [6, 7], 'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1]}

    grid_rf = GridSearchCV(rf, param_grid_rf, cv=3)
    grid_lr = GridSearchCV(lr, param_grid_lr, cv=3)
    grid_xgb = GridSearchCV(xgb, param_grid_xgb, cv=3)

    # Tune RandomForest
    grid_rf.fit(X, y)
    print(f"Best RandomForest Params: {grid_rf.best_params_}")
    rf_best = grid_rf.best_estimator_

    # Tune LogisticRegression
    grid_lr.fit(X, y)
    print(f"Best LogisticRegression Params: {grid_lr.best_params_}")
    lr_best = grid_lr.best_estimator_

    # Tune XGBoost
    grid_xgb.fit(X, y)
    print(f"Best XGBoost Params: {grid_xgb.best_params_}")
    xgb_best = grid_xgb.best_estimator_

    return rf_best, lr_best, xgb_best

# Deep Learning Model Hyperparameter Tuning using RandomizedSearchCV
def tune_dl_model(X_train, y_train):
    nn_model = KerasClassifier(build_fn=build_nn_model, input_dim=X_train.shape[1], epochs=10, batch_size=32, verbose=0)

    param_dist = {
        'batch_size': [32, 64],
        'epochs': [10, 20],
        'input_dim': [X_train.shape[1]],
    }
    randomized_search = GridSearchCV(estimator=nn_model, param_grid=param_dist, n_jobs=-1, cv=3)
    randomized_search.fit(X_train, y_train)

    print(f"Best Deep Learning Params: {randomized_search.best_params_}")
    return randomized_search.best_estimator_

# ------------------------ #
# Decision Engine          #
# ------------------------ #
def decision_engine(doc_type, data, errors):
    if errors:
        return "Flag for Review", errors

    ml_decision = ml_model.predict(doc_type, data["supplier"], data["total"])
    if ml_decision is not None:
        return ml_decision, []

    return "Auto-Approve", []

# ------------------------ #
# ERP Integration Stub     #
# ------------------------ #
def send_to_erp(data, decision):
    pass

# ------------------------ #
# File Processor           #
# ------------------------ #
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
        pass
