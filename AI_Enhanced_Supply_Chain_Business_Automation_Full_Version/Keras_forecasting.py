import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM,GRU,Dropout,RepeatVector, TimeDistributed
from tensorflow.keras.callbacks import EarlyStopping
from  preprocessing import processdata

def split_data(processdata,target):
  x=processdata.drop(target)
  y=processdata[target]
  return x,y

def split_train_test_dataset(x,y):
  x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
  return x_train,x_test,y_train,y_test

def shape(x_train):
    x_train=x_train.reshape(x_train.shape[0],x_train.shape[1],[1])
    input_shape=x_train
    return input_shape

def forecasting_neural_model(x_train, y_train, x_test, y_test, input_shape):
    # Callbacks for early stopping and best model saving
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Model definition
    model = Sequential()
    model.add(GRU(units=128, activation='relu', return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(GRU(units=64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(units=32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(units=1, activation='linear'))  # Regression output

    # Compile with MSE loss and learning rate tuning
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

    # Fit the model
    history = model.fit(
        x_train,
        y_train,
        batch_size=32,
        epochs=100,
        validation_split=0.2,
        callbacks=[early_stop, checkpoint],
        verbose=1
    )

    # Evaluate on test data
    test_loss, test_mae = model.evaluate(x_test, y_test, verbose=0)
    return test_loss, test_mae, model, history
