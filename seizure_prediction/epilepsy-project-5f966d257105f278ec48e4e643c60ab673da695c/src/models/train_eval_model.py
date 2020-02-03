import os
import json
import time
import argparse
import numpy as np
import tensorflow as tf
from scipy import interp
import matplotlib.pyplot as plt

from src.helpers.plots import plot_roc_auc
from src.data.processing.data_loading import loadData
from src.models.networks.core.FC import FullyConnectedNet

from sklearn import metrics
from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split, StratifiedKFold

# Debugger to check which device is being used (CPU or GPU) - by default GPU is used from TF 2.0
# tf.debugging.set_log_device_placement(True)

# Just disables the warning ("Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA"), doesn't enable AVX/FMA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# fix random seed for reproducibility
seed = 1
np.random.seed(seed)


class TrainEval():

    def __init__(self, args):

        self.patient = int(args.patient)
        self.network = args.network
        self.final_data_path = args.final_data_path
        self.preictal_duration = int(args.preictal_duration) # cast to int - we don't need it in seconds
        self.group_segments_form_input = eval(args.group_segments_form_input)
        self.n_segments_form_input = int(args.n_segments_form_input)
        self.segments_duration = args.segments_duration

        self.trainEvalNetwork()


    def trainEvalNetwork(self):

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

        # print("Splitting the data")
        # X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, stratify=Y) # , stratify=y
        #
        #
        # # Samples per class
        # unique_y_train, counts_y_train = np.unique(y_train, return_counts=True)
        # unique_y_test, counts_y_test = np.unique(y_test, return_counts=True)
        # print('Training samples per class: ', dict(zip(unique_y_train, counts_y_train)))
        # print('Test samples per class: ', dict(zip(unique_y_test, counts_y_test)))


        # Load parameters' from JSON
        # No CV parameters for TCN
        # if self.network != "TCN":
        parameters_dir = './output/cv-parameters/' + self.network + '/sec_' + self.segments_duration + '/' + 'chb{:02d}'.format(self.patient) + '/preictal_' + str(self.preictal_duration)  + '_best_params.json'
        with open(parameters_dir, 'r') as json_file:
            parameters = json.load(json_file)
        print("Best selected parameters: \n", parameters)


        # select neural network
        print('Creating the model...')
        if (self.network == "FC"):
            model = FullyConnectedNet.build_network(
                input_dim=input_dim,
                units1=parameters['units1'],
                units2=parameters['units2'],
                units3=parameters['units3'],
                dropout1=parameters['dropout1'],
                dropout2=parameters['dropout2'],
                dropout3=parameters['dropout3'],
                learning_rate=parameters['learning_rate'],
                multi_layer=parameters['multi_layer']
            )
            print("Model architecture: ", model.summary())

        # elif (MODEL == "TCN"):
        #
        #     # learning_rate, dropout
        #     model = TCNNetwork.build_model(
        #         input_dim=FEATURES_ORIGINAL_SIZE,
        #         timesteps=N_SEGMENTS_FORM_INPUT
        #     )
        #     print("Model architecture: ", model.summary())
        # elif (MODEL == "RNN"):
        #     model = RNNNetwork.build_model(
        #         input_dim=FEATURES_ORIGINAL_SIZE,
        #         timesteps=N_SEGMENTS_FORM_INPUT,
        #         units1=PARAMETERS['units1'],
        #         units2=PARAMETERS['units2'],
        #         units3=PARAMETERS['units3'],
        #         dropout1=PARAMETERS['dropout1'],
        #         dropout2=PARAMETERS['dropout2'],
        #         dropout3=PARAMETERS['dropout3'],
        #         learning_rate=PARAMETERS['learning_rate'],
        #         multi_layer=PARAMETERS['multi_layer'],
        #         l2_1=PARAMETERS['l2_1'],
        #         l2_2=PARAMETERS['l2_2'],
        #         l2_3=PARAMETERS['l2_3'],
        #         kernel_init=PARAMETERS['kernel_init']
        #     )
        #     print("Model architecture: ", model.summary())

        print("Started training...")

        start = time.time()

        # Because of imbalanced data, calculate class weights
        class_weights_calculated = class_weight.compute_class_weight('balanced', np.unique(Y), Y)
        class_weights = {0: class_weights_calculated[0], 1: class_weights_calculated[1]}
        print("Class weights: ", class_weights)


        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                # Stop training when `val_loss` is no longer improving
                monitor='val_loss',
                # "no longer improving" being defined as "no better than 1e-2 less"
                min_delta=1e-2,
                # "no longer improving" being further defined as "for at least 2 epochs"
                patience=3,
                verbose=1)
        ]

        # Hyper-parameters
        buffer_size = 500000
        batch_size = 10
        CV = 5

        # Plot the cv
        cv_skfold = StratifiedKFold(n_splits=CV) # shuffle=True, random_state=42

        # hold tprs, aucs and the mean to plot the ROC AUC
        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)

        final_acc = []
        final_balanced_acc = []
        final_precision = []
        final_recall = []
        final_f1 = []
        final_auc = []
        final_fpr = []
        final_fnr = []
        final_tpr = []
        final_tnr = []

        tps = []
        tns = []
        fps = []
        fns = []

        plt.figure(figsize=(8,6))

        kth = 0
        for train, test in cv_skfold.split(X, Y):
            kth += 1
            print("=================== FOLD: ", kth ,"======================================================================================================================================")

            # Samples per class
            unique_y_train, counts_y_train = np.unique(Y[train], return_counts=True)
            unique_y_test, counts_y_test = np.unique(Y[test], return_counts=True)
            print('Training samples per class: ', dict(zip(unique_y_train, counts_y_train)))
            print('Test samples per class: ', dict(zip(unique_y_test, counts_y_test)))

            # Load data to tf.data.Dataset, shuffle and create batches
            train_set = tf.data.Dataset.from_tensor_slices((X[train], Y[train])).shuffle(buffer_size).batch(batch_size,drop_remainder=False)
            test_set = tf.data.Dataset.from_tensor_slices((X[test], Y[test])).batch(batch_size, drop_remainder=False)

            # fit the model
            model.fit(train_set, epochs=50, validation_data=test_set, class_weight=class_weights, callbacks=callbacks)

            # evaluate the model
            loss, acc = model.evaluate(test_set)
            print("Eval Loss: ", loss, "\nEval Accuracy: ", acc)
            print('\n\n')


            # Prediction - metrics calculations

            # predict probabilities for test set
            y_pred_probs = model.predict(X[test]).ravel() #.ravel() # ravel() reduce to 1d array

            # predict crisp classes for test set
            if(self.network == "TCN"):
                y_pred_classes = (y_pred_probs > 0.5) # get classes for Model() since predict_classes() works for Sequential only
            else:
                y_pred_classes = model.predict_classes(X[test]).ravel()  # ravel() reduce to 1d array


            # Calculate the metrics
            # accuracy: (tp + tn) / (p + n)
            accuracy = metrics.accuracy_score(Y[test], y_pred_classes)
            final_acc.append(accuracy)
            print('Accuracy: ', accuracy)

            # balanced accuracy: n_samples / (n_classes * np.bincount(y))
            balanced_accuracy = metrics.balanced_accuracy_score(Y[test], y_pred_classes)
            final_balanced_acc.append(balanced_accuracy)
            print('Balanced accuracy: ', balanced_accuracy)

            # precision tp / (tp + fp)
            precision = metrics.precision_score(Y[test], y_pred_classes)
            final_precision.append(precision)
            print('Precision: ', precision)

            # recall: tp / (tp + fn)
            recall = metrics.recall_score(Y[test], y_pred_classes)
            final_recall.append(recall)
            print('Recall: ', recall)

            # f1: 2 tp / (2 tp + fp + fn)
            f1 = metrics.f1_score(Y[test], y_pred_classes)
            final_f1.append(f1)
            print('F1 score: ', f1)

            # Additional metrics
            # ROC AUC
            auc = metrics.roc_auc_score(Y[test], y_pred_probs)
            final_auc.append(auc)
            print('ROC AUC: ', auc)

            # Confusion matrix
            # true positives (TP): These are cases in which we predicted yes (they have the disease), and they do have the disease.
            # true negatives (TN): We predicted no, and they don't have the disease.
            # false positives (FP): We predicted yes, but they don't actually have the disease. (Also known as a "Type I error.")
            # false negatives (FN): We predicted no, but they actually do have the disease. (Also known as a "Type II error.")
            print("Confusion matrix: \n", metrics.confusion_matrix(Y[test], y_pred_classes))
            tn, fp, fn, tp = metrics.confusion_matrix(Y[test], y_pred_classes).ravel()
            tps.append(tp)
            tns.append(tn)
            fps.append(fp)
            fns.append(fn)
            print({'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp})

            fpr = fp/(fp+tn)
            final_fpr.append(fpr)
            print('FPR: ', fpr)

            fnr = fn/(fn+tp)
            final_fnr.append(fnr)
            print('FPR: ', fnr)

            tpr = tp/(tp+fn)
            final_tpr.append(tpr)
            print('TPR (sensitivity): ', tpr)

            tnr = tn/(fp+tn)
            final_tnr.append(tnr)
            print('TNR (specificity): ', tnr)


            # Plot ROC AUC
            # Compute ROC curve and area the curve
            fpr, tpr, thresholds = metrics.roc_curve(Y[test], y_pred_probs)
            tprs.append(interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
            roc_auc = metrics.auc(fpr, tpr)
            aucs.append(roc_auc)
            plt.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (kth, roc_auc))
            print("=========================================================================================================================================================")

        plot_roc_auc(tprs, mean_fpr, aucs, self.network, self.patient, plt, self.segments_duration, self.preictal_duration)

        print('Report the results:')
        print('Final acc', np.mean(final_acc))
        print('Final balanced_acc', np.mean(final_balanced_acc))
        print('Final precision', np.mean(final_precision))
        print('Final recall', np.mean(final_recall))
        print('Final f1', np.mean(final_f1))
        print('Final AUC', np.mean(final_auc))
        print('Final FPR', np.mean(final_fpr))
        print('Final FNR', np.mean(final_fnr))
        print('Final TPR (sensitivity)', np.mean(final_tpr))
        print('Final TNR (specificity)', np.mean(final_tnr))

        print("-------------------------------------------------------------------------------------------------------------------------------------------------")


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

    TrainEval(args)

if __name__ == '__main__':
    main()