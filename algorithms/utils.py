"""
Utilities for mutation algorithms.
"""

import tensorflow as tf
from tensorflow.keras import backend as K


def compute_gradient(seq, model):
    """
    Compute the gradient of the model loss with respect to inputs, evaluated at a particular sequence.

    Inputs:
        seq - a Sequence object
        model - a TensorFlow Model object
    """
    # Compute gradients
    input = tf.Variable(seq.to_predict())
    with tf.GradientTape() as tape:
        prediction = model(input, training=False)  # Logits for this minibatch
        target = tf.constant(0.0)
        target = tf.reshape(target, [1, 1])
        loss_value = K.binary_crossentropy(target, prediction)
        grads = tape.gradient(loss_value, input)

    # return n_positions x n_characters output
    return grads.numpy()[0]
