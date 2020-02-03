import os
import json
import time
import argparse
import numpy as np
import tensorflow as tf

from src.data.processing.data_loading import loadData
from src.models.networks.core.FC import FullyConnectedNet

from sklearn import metrics
from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split, StratifiedKFold

from sklearn.model_selection import RandomizedSearchCV

# Debugger to check which device is being used (CPU or GPU) - by default GPU is used from TF 2.0
# tf.debugging.set_log_device_placement(True)

# Just disables the warning ("Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA"), doesn't enable AVX/FMA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# fix random seed for reproducibility
seed = 1
np.random.seed(seed)


class TrainModelSelection():

    def __init__(self, args):

        self.patient = int(args.patient)
        self.network = args.network
        self.final_data_path = args.final_data_path
        self.preictal_duration = int(args.preictal_duration) # cast to int - we don't need it in seconds
        self.group_segments_form_input = eval(args.group_segments_form_input)
        self.n_segments_form_input = int(args.n_segments_form_input)
        self.segments_duration = args.segments_duration

        self.trainModelSelection()


    def trainModelSelection(self):

        # Define some hyper-parameters
        data_dir = self.final_data_path + '/chb{:02d}'.format(self.patient)
        segment_files_load = ['interictal_segments.npy', 'preictal_segments.npy']

        print("Loading the data")
        X, Y = loadData(data_dir, segment_files_load, self.group_segments_form_input, self.n_segments_form_input)

        print("X: ", np.shape(X))
        print("Y: ", np.shape(Y))

        # Calculate input dimensionality
        n_features = np.shape(X)[1]  # data features/dimensionality
        input_dim = n_features * self.n_segments_form_input if self.group_segments_form_input == True else n_features

        # if use of LSTM and group segments to form timesteps inputs, reshape the X to [timesteps,features] => timesteps = self.n_segments_form_input
        if(self.group_segments_form_input == True and (self.network == "LSTM" or self.network == "TCN")):
            features_original_size = int(n_features / self.n_segments_form_input)
            X = X.reshape([-1, self.n_segments_form_input, features_original_size])

        print("X after reshape: ", np.shape(X))

        print("Splitting the data")
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, stratify=Y) # , stratify=y

        print("X_train: ", np.shape(X_train))
        print("X_test: ", np.shape(X_test))
        print("y_train: ", np.shape(y_train))
        print("y_test: ", np.shape(y_test))


        # select neural network
        print('Creating the model...')
        if (self.network == "FC"):
            model = tf.keras.wrappers.scikit_learn.KerasClassifier(
                build_fn=FullyConnectedNet.build_network,
                input_dim=input_dim
            )

        print("Started training...")

        # Because of imbalanced data, calculate class weights
        class_weights_calculated = class_weight.compute_class_weight('balanced', np.unique(Y), Y)
        class_weights = {0: class_weights_calculated[0], 1: class_weights_calculated[1]}
        print("Class weights: ", class_weights)

        early_stop = [
            tf.keras.callbacks.EarlyStopping(
                # Stop training when `val_loss` is no longer improving
                monitor='val_loss',
                # "no longer improving" being defined as "no better than 1e-2 less"
                min_delta=1e-2,
                # "no longer improving" being further defined as "for at least 2 epochs"
                patience=5,
                verbose=1)
        ]

        fit_params = {
            'callbacks': early_stop,
            'epochs': 200,
            'batch_size': 16,
            'validation_data': (X_test, y_test),
            'verbose': 0
        }

        # random search's parameter:
        # specify the options and store them inside the dictionary
        # batch size and training method can also be hyperparameters,
        # but it is fixed
        params_dict = {
            'units1': [16, 32, 64, 128],
            'units2': [16, 32, 64, 128],
            'units3': [16, 32, 64, 128],
            'dropout1': [0.0, 0.2, 0.5, 0.8],
            'dropout2': [0.0, 0.2, 0.5, 0.8],
            'dropout3': [0.0, 0.2, 0.5, 0.8],
            'learning_rate': [0.01, 0.005, 0.001, 0.0001],
            'multi_layer': [True, False],
            'l2_1': [0.0001, 0.001, 0.01, 0.1],
            'l2_2': [0.0001, 0.001, 0.01, 0.1],
            'l2_3': [0.0001, 0.001, 0.01, 0.1],
            'kernel_init1': ['glorot_uniform', 'glorot_normal', 'he_uniform', 'he_normal'],
            'kernel_init2': ['glorot_uniform', 'glorot_normal', 'he_uniform', 'he_normal'],
            'kernel_init3': ['glorot_uniform', 'glorot_normal', 'he_uniform', 'he_normal']
        }

        # `verbose` 2 will print the class info for every cross validation,
        # kind of too much
        rs_model = RandomizedSearchCV(
            model,
            param_distributions=params_dict,
            fit_params=fit_params,
            n_iter=20,
            cv=5,
            verbose=1
        )
        rs_model.fit(X_train, y_train)

        # summarize results
        print("Best result: %f using params %s" % (rs_model.best_score_, rs_model.best_params_))
        self.writeBestParams(rs_model.best_params_)
        means = rs_model.cv_results_['mean_test_score']
        stds = rs_model.cv_results_['std_test_score']
        params = rs_model.cv_results_['params']
        for mean, stdev, param in zip(means, stds, params):
            print("%f (%f) with: %r" % (mean, stdev, param))

    def writeBestParams(self, best_params):
        param_dir = "./output/cv-parameters/" + self.network + "/sec_" + self.segments_duration + "/" + 'chb{:02d}'.format(self.patient)
        os.makedirs(param_dir, exist_ok=True)  # create any parent directory that does not exist
        best_model_file = open(param_dir + "/preictal_" + str(self.preictal_duration) + "_best_params.json", "w")
        best_model_file.write(json.dumps(best_params, default=self.default))
        best_model_file.close()

    # write the best model params to a JSON file
    # helper function to cast np.int64 integers to int because JSON does not accept np.int64
    def default(self, o):
        if isinstance(o, np.int64): return int(o)
        raise TypeError

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--patient", help="Patient number")
    parser.add_argument("--network", help="Which neural network to use")
    parser.add_argument("--final_data_path", help="Path to load data from binary files")
    parser.add_argument("--preictal_duration", help="Preictal duration in minutes")
    parser.add_argument("--group_segments_form_input", help="bool: Group segments to form inputs (LSTM,TCN)")
    parser.add_argument("--n_segments_form_input", help="How many segments to use to form inputs for sequential networks (LSTM, TCN) - works iff group_segments_form_input==True.")
    parser.add_argument("--segments_duration", help="Duration used to do segmentation (e.g. 5 or 30 secs)")
    args = parser.parse_args()

    TrainModelSelection(args)

if __name__ == '__main__':
    main()