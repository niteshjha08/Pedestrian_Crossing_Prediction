#!/usr/bin/env python3
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import data_utils as du
import config as cfg
import neural_network as nn


def start_pipeline(run_eda=False, run_pca=False):
    train_data_file_path = "../Data/crossing_dataset_train.csv"
    training_data = du.load_data(train_data_file_path)

    test_data_file_path = "../Data/crossing_dataset_test.csv"
    testing_data = du.load_data(test_data_file_path)

    if run_eda:
        # explore the data
        du.explore_data(training_data)

    feature_set = cfg.FEATURE_SET_FULL

    per_frame_data = du.unravel_data(training_data, cfg.WINDOW_SIZE, feature_set=feature_set)
    random.shuffle(per_frame_data)

    test_data = du.unravel_data(testing_data, cfg.WINDOW_SIZE, feature_set=feature_set)

    # split the dataset into training-test and input-output
    train_data, val_data = train_test_split(per_frame_data, train_size=cfg.TRAIN_SPLIT, random_state=10)

    # Split the train data into input and output
    X_train, Y_train = du.split_input_and_output(train_data)
    # Split the validation data into input and output
    X_val, Y_val = du.split_input_and_output(val_data)
    # Split the test data into input and output
    X_test, Y_test = du.split_input_and_output(test_data)

    print("size train: ", len(X_train))
    print("size test: ", len(X_test))
    print("size val: ", len(X_val))

    if run_pca:
        pca = du.get_pca(X_train)
        X_val = du.apply_pca(pca, X_val)
        X_test = du.apply_pca(pca, X_test)

        Y_train, Y_val, Y_test = np.array(Y_train), np.array(Y_val), np.array(Y_test)
    
    # tune the hyperparameters
    nn.tune_hyperparameters(X_train, Y_train, X_val, Y_val)
    
    # get the best model based on validation losses
    nn.find_best_model()

    print("[INFO] From the best model, get the best hyperparameters and train the final model using final_training.py")

if __name__ == "__main__":
    run_eda = True
    run_pca = True
    start_pipeline(run_eda, run_pca)
