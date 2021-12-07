# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 10:08:01 2020

@author: NBOLLIG
"""

from mgm.data.HMM_generator_motif import HMMGenerator
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import Flatten


def encode_generated_sequences(X_raw, aa_vocab):
    """
    Encode sequences in the format returned by the generator.

    Parameters:
        X _raw: list of sequence strings
        aa_vocab: list of aa tokens
    
    Output:
        List of one-hot-encoded sequences
    """
#    # Encode each instance as a list of indices into aa vocabulary [NEEDED FOR CHARACTER SEQ, NOT INDEX SEQ]
#    aa_dict = {}
#    for i, aa in enumerate(aa_vocab):
#        aa_dict[aa] = i
#    
#    X_seq = []
#    for x in X_raw:
#        x = list(x)
#        x = [aa_dict[aa] for aa in x]
#        X_seq.append(x)

    # Bypass character to index conversion above
    X_seq = X_raw

    # Transform to one-hot encoding
    X = to_categorical(X_seq)
    
    return X

def create_LSTM(X_train, X_val, y_train, y_val, n_epochs = 10):
    # define the  model
    model = Sequential()
    model.add(Bidirectional(LSTM(128, input_shape=(60,20))))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    # Train model
    model.fit(X_train, y_train, epochs=n_epochs, batch_size=64, verbose=1)
    
    # Evaluate on train set
    result = {}
    _, train_accuracy = model.evaluate(X_train, y_train, verbose=0)
    result['model_train_accuracy'] = train_accuracy
    print('Train Accuracy: %.2f' % (train_accuracy*100))
    
    # Evaluate on validation set
    _, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
    result['model_val_accuracy'] = val_accuracy
    print('Validation Accuracy: %.2f' % (val_accuracy*100))
    
    return model, result

def create_LR(X_train, X_val, y_train, y_val, n_epochs = 10):
    # define the  model
    model = Sequential()
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    # Train model
    model.fit(X_train, y_train, epochs=n_epochs, batch_size=64, verbose=1)
    
    # Evaluate on train set
    result = {}
    _, train_accuracy = model.evaluate(X_train, y_train, verbose=0)
    result['model_train_accuracy'] = train_accuracy
    print('Train Accuracy: %.2f' % (train_accuracy*100))
    
    # Evaluate on validation set
    _, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
    result['model_val_accuracy'] = val_accuracy
    print('Validation Accuracy: %.2f' % (val_accuracy*100))
    
    return model, result

def big_bang(num_instances=5000, p=0.5, class_signal=10, n_epochs=10, model_type='LSTM'):
    """
    Generates sequence data and trains a model.
    
    Parameters:
        num_instances: the number of total instances to generate
        p: positive (1) class prevalance
        model_type: string designating type of model to be created
    
    
    Returns:
        model: trained TensorFlow Model
        X: list [X_train, X_val, X_test]
        y: list [y_train, y_val, y_test]
        gen: generator object
    """
    # Generate data
    gen = HMMGenerator(p = p, class_signal = class_signal)
    X_raw, y = gen.generate(n_samples=num_instances)
    aa_vocab = gen.aa_list
    
    # Encode X
    X = encode_generated_sequences(X_raw, aa_vocab)
    
    # Split into train, validation, test (80/10/10)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=1)
    X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=0.5, random_state=1)
    
    # Define model
    if model_type == 'LSTM':
        model_fn = create_LSTM
    elif model_type == 'LR':
        model_fn = create_LR
    
    model, result = model_fn(X_train, X_val, y_train, y_val, n_epochs=n_epochs)
    
    return model, result, [X_train, X_val, X_test], [y_train, y_val, y_test], gen, aa_vocab
