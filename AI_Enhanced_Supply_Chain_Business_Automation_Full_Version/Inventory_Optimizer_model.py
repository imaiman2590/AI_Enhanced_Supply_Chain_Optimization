import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM,GRU,Dropout,RepeatVector, TimeDistributed
from tensorflow.keras.callbacks import EarlyStopping
from preprocessing import processdata

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

def shape(x_train):
    x_train=x_train.reshape(x_train.shape[0],x_train.shape[1],[1])
    input_shape=x_train
    return input_shape

def build_model(input_shape):
    model = Sequential()
    model.add(LSTM(64, activation='tanh', return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.3))
    model.add(LSTM(64, activation='tanh'))
    model.add(Dropout(0.3))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))  

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
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
