import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import neural_network as nn
import config as cfg 
import classifier as clf
import data_utils as du
import os
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import random

def generate_learning_curve(data_per_frame, parameters, epochs = 50):
    """
    Generates a learning curve of the training and validation losses vs. training sample size
    """
    val_losses=  []
    # Take section of data to generate learning curve
    for i in np.arange(0.02, 0.03, 0.001):
        # Take a subset of the data
        partial_data = data_per_frame[:int(len(data_per_frame)*i)]
        # Split the data into training and validation sets
        train_data, val_data = train_test_split(partial_data, train_size=cfg.TRAIN_SPLIT)

        # Split the train data into input and output
        X_train, Y_train = du.split_input_and_output(train_data)
        # Split the validation data into input and output
        X_val, Y_val = du.split_input_and_output(val_data)
    
        # Apply PCA
        pca_trans = du.get_pca(n_components=8)
        X_train = du.apply_pca(pca_trans, X_train)
        X_val = du.apply_pca(pca_trans, X_val)

        Y_train = np.array(Y_train)
        Y_val = np.array(Y_val)

        print('[INFO]: Started model training for parameters: \n', parameters)
        # Train the model
        min_val_loss = nn.train_model(X_train, Y_train, X_val, Y_val, parameters, epochs)
        # Append the validation loss to the list
        val_losses.append(min_val_loss)
    # Plot the learning curve
    plt.plot(val_losses, label='Validation loss')
    plt.xlabel("Training data percentage")
    plt.ylabel("Loss")
    plt.show()

if __name__ == '__main__':
    train_data_file_path = "../Data/crossing_dataset_train.csv"
    training_data = du.load_data(train_data_file_path)

    test_data_file_path = "../Data/crossing_dataset_test.csv"
    testing_data = du.load_data(test_data_file_path)

    feature_set = cfg.FEATURE_SET_FULL

    data_per_frame = du.unravel_data(training_data, cfg.WINDOW_SIZE, feature_set=feature_set)
    random.shuffle(data_per_frame)

    parameters = {'lr': 0.01, 'l1': 0, 'l2': 0.01, 'do': 0, 'ml': 8, 'o': 'check_learning_curve'}
    generate_learning_curve(data_per_frame, parameters, epochs=50)