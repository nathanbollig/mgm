"""
Module for NN models. Import main function like "make_LSTM", then pass id param to select specific model.

Inputs:
    X_train - used to infer architecture
    y_train
    id

Output: TensorFlow Sequential model object
"""


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten


def make_LSTM(X_train, y_train, N_POS, id=0):
    if id == 0:
        return _LSTM_0(X_train, y_train, N_POS)

def make_CNN(X_train, y_train, N_POS, id=0):
    if id == 0:
        return _CNN_0(X_train, y_train, N_POS)

def _LSTM_0(X_train, y_train, N_POS):
    n = X_train.shape[0]
    X_train = X_train.reshape((n, N_POS, -1))
    num_features = X_train.shape[2]

    model = Sequential()
    model.add(Bidirectional(LSTM(64), input_shape=(N_POS, num_features)))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def _CNN_0(X_train, y_train, N_POS):
    n = X_train.shape[0]
    X_train = X_train.reshape((n, N_POS, -1))
    num_features = X_train.shape[2]

    model = Sequential()
    model.add(Conv1D(filters=32, kernel_size=5, input_shape=(N_POS, num_features)))
    model.add(MaxPooling1D(pool_size=5))
    model.add(Flatten())
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model