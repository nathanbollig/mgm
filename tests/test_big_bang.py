

from mgm.data.HMM_generator_motif import HMMGenerator
from mgm.data.seq_model import big_bang
import numpy as np

def test_big_bang():
    # Test HMMGenerator and predict method
    gen = HMMGenerator()
    count = 0
    total = 100

    for i in range(total):
        seq, y = gen.generate_one_sequence()
        y_inference = gen.predict(seq)
        if y != y_inference:
            count += 1

    inference_error = count /total
    print("Fraction of incorrect inferences: %.2f" % (inference_error,))
    assert(inference_error < 0.1)

    # Big bang
    p=0.5
    n=200
    big_bang_result_tuple = big_bang(class_signal=10,
                                      num_instances=n,
                                      p=p,
                                      model_type="LSTM",
                                      n_epochs=10)

    model, result, X_list, y_list, gen, aa_vocab = big_bang_result_tuple
    X_train, X_val, X_test = X_list
    y_train, y_val, y_test = y_list

    assert(result['model_train_accuracy'] > 0.8)
    assert(result['model_val_accuracy'] > 0.5)
    assert(np.abs(np.mean(y_train) - 0.5) < 0.8)
    assert(X_train.shape[0] + X_val.shape[0] + X_test.shape[0] == n)
    assert(len(y_train) + len(y_val) + len(y_test) == n)

    print("Passed!")

if __name__ == "__main__":
    test_big_bang()
