"""
This script is to compare a local implementation of MGM to the distributed application in Spark. This will read in the same
test set and apply the MGM adversarial mutation algorithm to the negative sequences in that dataset.

Executing this as part of the mgm package should run without any additional data input requirements. The following
files are used by this test:
    model.tf
    aa_vocab.pkl
    data_test.txt
"""
import pickle
import datetime
from tensorflow import keras

from mgm.algorithms.mutations import greedy_mgm
from mgm.common.sequence import Sequence
import pandas as pd

from mgm.common.utils import get_full_path
model_file_name = get_full_path("tests", "sample_data", "model.tf")
aa_vocab_file_name = get_full_path("tests", "sample_data", "aa_vocab.pkl")
data_file_name = get_full_path("tests", "sample_data", "data_test.txt")

def test_744_experiment:
    # Load model
    model = keras.models.load_model(model_file_name)

    # Load aa vocab
    with open(aa_vocab_file_name, 'rb') as f:
        aa_vocab = pickle.load(f)

    # Initialize variables for output
    outputs = []
    dataframes = []
    start_time = datetime.datetime.now()

    # Load data
    linecount = 0
    with open(data_file_name, 'r') as f:
        for line in f:
            # Print linecount
            if linecount % 100 == 0:
                print("Line %i of total" % (linecount,))

            # Increment counter
            linecount += 1

            # Parse line
            items = line.lower().split(',')
            label = int(items[-1])

            # Only proceed if negative
            if label == 1:
                continue

            # Transform into Sequence object
            x = items[0:-1]
            seq = Sequence(x, y=label, aa_vocab=aa_vocab)

            # Apply hotflip using the greedy mgm wrapper
            x_new, data = greedy_mgm(seq, model=model, confidence_threshold = 0.9, type="hotflip", verbose=False)

            # Save mutated sequence
            outputs.append(x_new.integer_encoded)

            # Save mutation trajectory data
            dataframes.append(pd.DataFrame(data))

    # Save data
    data = pd.concat(dataframes)
    data.to_csv("dataframe.csv")
    with open("outputs.pkl", 'wb') as pfile:
        pickle.dump(outputs, pfile, protocol=pickle.HIGHEST_PROTOCOL)

    # record end time
    end_time = datetime.datetime.now()
    time_s = (end_time - start_time).total_seconds()
    print("Total time (sec): %.3f" % (time_s,))

if __name__ == "__main__":
    test_744_experiment()