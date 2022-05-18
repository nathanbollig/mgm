import os
import pickle
import random
import numpy as np

from sklearn.metrics import roc_auc_score

from mgm.common.utils import set_data_directory
from mgm.data.kuzmin_data import load_kuzmin_data, LOOCV
from mgm.models.NN import make_CNN
from mgm.common.utils import get_full_path

from sklearn.model_selection import LeaveOneOut

set_data_directory('NB_Ze_comparison_5')

def read_pickle(filename):
  location_full = get_full_path("data", "Ze_data", filename)
  with open(location_full, 'rb') as f:
    return pickle.load(f)

def write_pickle(filename, data):
  with open(os.path.join(os.getcwd(), filename), 'wb') as f:
    pickle.dump(data,f)

output = ''

# Nathan's one-hot with package data load routine
model_initializer = make_CNN
# X, y, species, deflines, sequences, sp, human_virus_species_list, seqs = load_kuzmin_data(representation_type='one-hot')
# thresh, data = LOOCV(model_initializer, X, y, species, epochs=10, output_string="NB_oh")

# Nathan's one-hot with Ze's data splits
X = read_pickle('X.pkl').reshape((-1, 2396, 25))
y = read_pickle('y.pkl')
species = read_pickle('species.pkl')
deflines = read_pickle('deflines.pkl')
sp = read_pickle('sp.pkl')
human_virus_species_list = read_pickle('human_virus_species_list.pkl')
thresh, data = LOOCV(model_initializer, X, y, species, epochs=10, output_string="NB_oh_Ze_data", batchsize=64)
auc_61 = data['auc_based_on_avg_within_species']
auc_all = data['auc_based_on_all_seq']
output += 'NB - One Hot AUROC: %.3f, %.3f (grouped, all)\n' % (auc_61, auc_all)
with open("results.txt", "w") as text_file:
  text_file.write(output)

# Nathan's kidera with package data load routine
# X_kidera, _, _, _, _, _, _, _ = load_kuzmin_data(representation_type='kidera')
# thresh, data = LOOCV(model_initializer, X_kidera, y, species, epochs=10, output_string="NB_kidera")

# Nathan's Kidera with Ze's data splits
X_kidera =  read_pickle('chemical_physical_representation_map_10_proterties.pkl')
thresh, data = LOOCV(model_initializer, X_kidera, y, species, epochs=10, output_string="NB_kidera_Ze_data", batchsize=64)
auc_61 = data['auc_based_on_avg_within_species']
auc_all = data['auc_based_on_all_seq']
output += 'NB - Kidera AUROC: %.3f, %.3f (grouped, all)\n' % (auc_61, auc_all)
with open("results.txt", "w") as text_file:
  text_file.write(output)

########################################################################################################################
# Ze's Code
########################################################################################################################

def all_species_index_sets(species):
  """
  Builds index of indices for each unique species in the species list.
  Inputs:
      species - list of species labels for each data item (e.g. parallel to deflines)
  Output:
      species_index - dictionary such that sp[species name] is a list of indices corresponding to
              that species in the species list.
  """

  species_index = {}

  for i, species_name in enumerate(species):
    if species_name not in species_index.keys():
      species_index[species_name] = []
    species_index[species_name].append(i)

  return species_index

def LOOCV_Ze(X, y, species, rs):
  """
  Takes in a model initializer and kuzmin data, applies leave-one-out CV (wrt viral species) to determine model performance.
  Strategy:
      1. Iterate over unique viral species (54+7 in total)
      2. For each species, withhold all sequences of that species.
      3. Train a model on the remaining sequences
      4. Apply the trained model to each of the withheld sequences, to get a set of predicted probabilities.
      5. Report average predicted probability for each viral species (average in csv file + distributions shown in violin plot)
      6. Use this set of 61 numbers with corresponding binary targets to generate an ROC curve.
  Model initializer parameter is a function that takes three parameters: X, y, N_POS
  """
  random.seed(rs)
  np.random.seed(rs)
  # Build dictionary of index sets for each species
  species_index = all_species_index_sets(species)

  # Get list of all unique species names
  species_names_list = np.array(list(species_index.keys()))

  # Do LOOCV on species_names_list
  loo = LeaveOneOut()
  num_splits = loo.get_n_splits(species_names_list)
  print("Running LOOCV with %i splits..." % (num_splits,))

  output = []
  i = 0
  Y_avg_proba = []
  Y_pred_lists = []
  Y_targets = []
  Y_species = []
  train_test_list = []

  for train_species_idx, hold_species_idx in loo.split(species_names_list):
    train_species = species_names_list[train_species_idx]  # names of species to be in training set
    hold_species = species_names_list[hold_species_idx]  # name of withheld species

    assert (len(hold_species) == 1)
    hold_species = hold_species[0]

    train = []
    for name in train_species:
      index_set = species_index[name]  # list of indices where this species is
      train.extend(index_set)

    test = species_index[hold_species]
    train_test_list.append([train, test])

    i += 1

  return train_test_list

def initialize_proba(classifiers):
  proba_map = {}
  for model_name in classifiers:
    proba_map[model_name] = []
  return proba_map

def train_model(classifiers, model_name, X_train, y_train, X_test, bs, N_POS=2396):
  # Sklear classifiers
  if classifiers[model_name] != None:
    model = clone(classifiers[model_name])
    history = model.fit(X_train, y_train)
    y_proba = model.predict_proba(X_test)
    assert (y_proba.shape[1] == 2)
    y_proba = y_proba[:, 1]
    y_proba_train = model.predict_proba(X_train)
    y_proba_train = y_proba_train[:, 1]

  # Keras classifiers
  if model_name == "CNN":
    print('before', X_train.shape)
    X_train = X_train.reshape((X_train.shape[0], N_POS, -1))
    print('after', X_train.shape)
    X_test = X_test.reshape((X_test.shape[0], N_POS, -1))
    model = make_CNN(X_train, y_train, N_POS=N_POS)
    history = model.fit(X_train, y_train, epochs=10, batch_size=bs)
    y_proba = model.predict(X_test)
    y_proba_train = model.predict(X_train)

  return y_proba, y_proba_train

def get_predict_result_LOOCV_61(X,y,train_test_list,classifiers,model_list,human_virus_species_list,bs,N_POS):
  histories_all = []
  Y_targets = []
  Y_targets_61=[]
  Y_proba = initialize_proba(model_list)
  Y_proba_61 =  initialize_proba(model_list)
  i=0
  for train,test in train_test_list:
    X_train = X[train]
    y_train = y[train]
    X_test = X[test]
    y_test = y[test]

    print("*******************FOLD %i*******************" % (i, ))
    print("Test size = %i" % (len(y_test),))
    print("Test pos class prevalence: %.3f" % (np.mean(y_test),))
    i+=1
    for model_name in model_list:
      print("Training %s..." % (model_name,))
      y_proba, y_proba_train=train_model(classifiers,model_name, X_train, y_train, X_test,bs,N_POS)
      Y_proba[model_name].extend(y_proba)
      Y_proba_61[model_name].append(np.mean(y_proba))
    Y_targets.extend(y_test)
    Y_targets_61.append(y_test[0])

  return (Y_proba,Y_proba_61,Y_targets,Y_targets_61)

def saveLOOCVResult_61(X,y,species,sp,rs,bs, N_POS,classifiers,model_list,human_virus_species_list,save_name):
  random.seed(rs)
  np.random.seed(rs)
  train_test_list = LOOCV_Ze(X,y,species,rs)
  Y_proba,Y_proba_61,Y_targets,Y_targets_61 = get_predict_result_LOOCV_61(X,y,train_test_list,classifiers,model_list,human_virus_species_list,bs,N_POS)
  return (Y_proba,Y_proba_61,Y_targets,Y_targets_61)

all_models = ["CNN"]
classifiers = {"CNN": None}

Y_proba, Y_proba_61, Y_targets, Y_targets_61 = saveLOOCVResult_61(X,y,species,sp,10,64,2396, classifiers, all_models, human_virus_species_list, 'LOOCV_ALL_onehot_10_64.pkl')
auc_61 = roc_auc_score(Y_targets_61, Y_proba_61['CNN'])
auc_all = roc_auc_score(Y_targets, Y_proba['CNN'])
output += 'Ze - One Hot AUROC: %.3f, %.3f (grouped, all)\n' % (auc_61, auc_all)
with open("results.txt", "w") as text_file:
  text_file.write(output)


Y_proba, Y_proba_61, Y_targets, Y_targets_61 = saveLOOCVResult_61(X_kidera,y,species,sp,10,64,2396,classifiers,all_models,human_virus_species_list,'LOOCV_ALL_kidera_10_64.pkl')
auc_61 = roc_auc_score(Y_targets_61, Y_proba_61['CNN'])
auc_all = roc_auc_score(Y_targets, Y_proba['CNN'])
output += 'Ze - Kidera AUROC: %.3f, %.3f (grouped, all)\n' % (auc_61, auc_all)
with open("results.txt", "w") as text_file:
  text_file.write(output)