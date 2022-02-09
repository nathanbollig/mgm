"""
Utilities for mutation algorithms.
"""

import tensorflow as tf
from tensorflow.keras import backend as K


def compute_gradient(seq, model, representation='one-hot', loss=True):
    """
    Compute the gradient of the model with respect to inputs, evaluated at a particular sequence.

    Inputs:
        seq - a Sequence object
        model - a TensorFlow Model object
        loss - If true, the gradient is model loss wrt inputs. If false, the gradient is model output wrt inputs.
    """
    # Compute gradients
    input = tf.Variable(seq.to_predict(representation))
    with tf.GradientTape() as tape:
        prediction = model(input, training=False)  # Logits for this minibatch
        target = tf.constant(0.0)
        target = tf.reshape(target, [1, 1])
        if loss is True:
            target_value = K.binary_crossentropy(target, prediction)
        else:
            target_value = prediction

        grads = tape.gradient(target_value, input)

    # return n_positions x n_characters output
    return grads.numpy()[0]
