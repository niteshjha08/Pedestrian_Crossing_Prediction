import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import neural_network as nn
import config as cfg
import data_utils as du
import random   
import os
from tensorflow import keras
from sklearn.metrics import fbeta_score

def final_train_and_eval():
    # Load the data
    train_data_file_path = "../Data/crossing_dataset_train.csv"
    training_data = du.load_data(train_data_file_path)

    test_data_file_path = "../Data/crossing_dataset_test.csv"
    testing_data = du.load_data(test_data_file_path)

    feature_set = cfg.FEATURE_SET_FULL

    # Unravel the data into frames
    per_frame_data = du.unravel_data(training_data, cfg.WINDOW_SIZE, feature_set=feature_set)
    random.shuffle(per_frame_data)
    test_data = du.unravel_data(testing_data, cfg.WINDOW_SIZE, feature_set=feature_set)

    # Split the data into train and validation
    train_data, val_data = train_test_split(per_frame_data, train_size=cfg.TRAIN_SPLIT, random_state=10)

    # Split the train data into input and output
    X_train, Y_train = du.split_input_and_output(train_data)
    # Split the validation data into input and output
    X_val, Y_val = du.split_input_and_output(val_data)
    # Split the test data into input and output
    X_test, Y_test = du.split_input_and_output(test_data)
    
    # Apply PCA for 8 components , i.e. ~ 95% variance
    pca_trans = du.get_pca(n_components=8)
    X_train = du.apply_pca(pca_trans, X_train)
    X_val = du.apply_pca(pca_trans, X_val)
    X_test = du.apply_pca(pca_trans, X_test)

    Y_train, Y_val, Y_test = np.array(Y_train), np.array(Y_val), np.array(Y_test)

    print("Training set length: ", len(X_train))
    print("Validation set length: ", len(X_val))
    print("Testing set length: ", len(X_test))

    # Form parameter dict based on plots of validation loss and abs(val_loss - train_loss)
    output_path = os.path.join("../Data", 'best_model')
    parameters = {'lr': 0.0001, 'l1': 0, 'l2': 0.001, 'do': 0.4, 'ml': 8, 'o': output_path, 'es': True}

    print('[INFO]: Started model training for best parameters: \n', parameters)
    
    nn.train_model(X_train, Y_train, X_val, Y_val, parameters, epochs = 200)

    print('[INFO]: Completed model training for best parameters: \n')

    model = keras.models.load_model(output_path)

    # Model evaluation format: [loss, accuracy]
    train_metrics = model.evaluate(X_train, Y_train)
    print('Train Metrics - ', train_metrics)
    val_metrics = model.evaluate(X_val, Y_val)
    print('Val Metrics - ', val_metrics)
    test_metrics = model.evaluate(X_test, Y_test)
    print('Test Metrics - ', test_metrics)

    
    print('================================')
    train_pred = model.predict(X_train)
    train_pred = np.array(train_pred >= cfg.THRESHOLD, dtype=np.float32)
    train_f1 = fbeta_score(Y_train, train_pred, beta=1.0)
    print('Train f1 score - ', train_f1)
    val_pred = model.predict(X_val)
    val_pred = np.array(val_pred >= cfg.THRESHOLD, dtype=np.float32)
    val_f1 = fbeta_score(Y_val, val_pred, beta=1.0)
    print('val f1 score - ', val_f1)
    test_pred = model.predict(X_test)
    test_pred = np.array(test_pred >= cfg.THRESHOLD, dtype=np.float32)
    test_f1 = fbeta_score(Y_test, test_pred, beta=1.0)
    print('Test f1 score - ', test_f1)

if __name__ == "__main__":
    final_train_and_eval()