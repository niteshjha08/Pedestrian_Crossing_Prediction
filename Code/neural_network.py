#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.python.keras.layers.core import Dropout
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
from ann_visualizer.visualize import ann_viz

import data_utils as du
import config as cfg



def two_hlayer_network(parameters):
    """
    This function returns a two hidden layer neural network model with the given parameters.
    """
    # Get the parameters
    lr = parameters.get('lr', 0.01)
    l1 = parameters.get('l1', 1)
    l2 = parameters.get('l2', 0)
    do = parameters.get('do', 0.2)
    in_dim = parameters.get('in', 12)
    # Create the model
    model = keras.Sequential()
    model.add(layers.Dense(12, input_dim=in_dim, activation='relu'))
    
    if l1!=0:
        model.add(layers.Dense(32, kernel_regularizer=keras.regularizers.l1(l1), activation='relu'))
    elif l2 != 0:
        model.add(layers.Dense(32, kernel_regularizer=keras.regularizers.l2(l2), activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss='binary_crossentropy', metrics=['accuracy'])
    
    return model


def four_hlayer_network(parameters):
    """
    This function returns a four hidden layer neural network model with the given parameters.
    """
    # Get the parameters
    lr = parameters.get('lr', 0.01)
    l1 = parameters.get('l1', 0)
    l2 = parameters.get('l2', 0.1)
    do = parameters.get('do', 0)
    in_dim = parameters.get('in', 12)
    # Create the model
    model = keras.Sequential()
    model.add(layers.Dense(12, input_dim=in_dim, activation='relu'))
    model.add(layers.Dense(8, activation='relu'))
    model.add(layers.Dropout(do))
    if l1 != 0:
        model.add(layers.Dense(16, kernel_regularizer=keras.regularizers.l1(l1), activation='relu'))
        model.add(layers.Dropout(do))
        
        model.add(layers.Dense(16, kernel_regularizer=keras.regularizers.l1(l1), activation='relu'))
    elif l2 != 0:
        model.add(layers.Dense(16, kernel_regularizer=keras.regularizers.l2(l2), activation='relu'))
        model.add(layers.Dropout(do))
        
        model.add(layers.Dense(16, kernel_regularizer=keras.regularizers.l2(l2), activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss='binary_crossentropy', metrics=['accuracy'])
    
    return model


def eight_hlayer_network(parameters):
    """
    This function returns a eight hidden layer neural network model with the given parameters.
    """
    # Get the parameters
    lr = parameters.get('lr', 0.01)
    l1 = parameters.get('l1', 0.1)
    l2 = parameters.get('l2', 0)
    do = parameters.get('do', 0)
    in_dim = parameters.get('in', 12)
    # Create the model
    model = keras.Sequential()
    model.add(layers.Dense(12, input_dim=in_dim, activation='relu'))
    model.add(layers.Dense(8, activation='relu'))
    model.add(layers.Dropout(do))
    if l1 != 0:
        model.add(layers.Dense(16, kernel_regularizer=keras.regularizers.l1(l1), activation='relu'))
        model.add(layers.Dropout(do))
        model.add(layers.Dense(32, kernel_regularizer=keras.regularizers.l1(l1), activation='relu'))
        model.add(layers.Dropout(do))
        model.add(layers.Dense(64, kernel_regularizer=keras.regularizers.l1(l1), activation='relu'))
        model.add(layers.Dropout(do))
        model.add(layers.Dense(64, kernel_regularizer=keras.regularizers.l1(l1), activation='relu'))
        model.add(layers.Dropout(do))
        model.add(layers.Dense(32, kernel_regularizer=keras.regularizers.l1(l1), activation='relu'))
        model.add(layers.Dropout(do))
        model.add(layers.Dense(32, kernel_regularizer=keras.regularizers.l1(l1), activation='relu'))
        model.add(layers.Dropout(do))

    elif l2 != 0:
        model.add(layers.Dense(16, kernel_regularizer=keras.regularizers.l2(l2), activation='relu'))
        model.add(layers.Dropout(do))
        model.add(layers.Dense(32, kernel_regularizer=keras.regularizers.l2(l2), activation='relu'))
        model.add(layers.Dropout(do))
        model.add(layers.Dense(64, kernel_regularizer=keras.regularizers.l2(l2), activation='relu'))
        model.add(layers.Dropout(do))
        model.add(layers.Dense(64, kernel_regularizer=keras.regularizers.l2(l2), activation='relu'))
        model.add(layers.Dropout(do))
        model.add(layers.Dense(32, kernel_regularizer=keras.regularizers.l2(l2), activation='relu'))
        model.add(layers.Dropout(do))
        model.add(layers.Dense(32, kernel_regularizer=keras.regularizers.l2(l2), activation='relu'))
        model.add(layers.Dropout(do))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss='binary_crossentropy', metrics=['accuracy'])

    return model


def train_model(X_train, Y_train, X_val, Y_val, parameters, epochs = 50):
    """
    This function trains the model with the given parameters and returns the minimum validation loss
    """
    # Get the parameters
    output_path = parameters['o']
    lr = parameters['lr']
    l1 = parameters['l1']
    l2 = parameters['l2']
    es = parameters.get('es', False)
    
    layer_count = parameters['ml']
    dropout = parameters['do']
    # Create the output directory if it does not exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    best_val_loss_checkpoint = ModelCheckpoint(filepath=output_path, monitor='val_loss', save_best_only=True, verbose=0,
                                               mode='min')
    model_layer = parameters['ml']
    parameters['in'] = np.array(X_train).shape[1]

    if model_layer == 2:
        model = two_hlayer_network(parameters)
    elif model_layer == 4:
        model = four_hlayer_network(parameters)
    else:
        model = eight_hlayer_network(parameters)
    
    early_stopping = EarlyStopping(monitor='loss', patience=10)
    # Train the model
    if es:
        cb = [best_val_loss_checkpoint, early_stopping]

    else:
        cb = [best_val_loss_checkpoint]
    history = model.fit(X_train, Y_train, epochs=epochs, batch_size=512,
                        validation_data=(X_val, Y_val), workers=4, callbacks=cb)
    # Plot training & validation loss values
    plt.plot(history.history['loss'], 'b', label='train_loss')
    plt.plot(history.history['val_loss'], 'orange', label='val_loss')
    plt.legend(loc='upper center')
    plt.title('lr-{0}, l1-{1}, l2-{2}, layers-{3}, dropout-{4}'.format(lr, l1, l2, layer_count, dropout))
    plt.xlabel('epochs')
    plt.ylabel('binary_crossentropy')
    plt.savefig(os.path.join(output_path, 'bce_loss.png'))
    plt.close()
    # Plot training & validation accuracy values
    plt.plot(history.history['accuracy'], 'b', label='train_accuracy')
    plt.plot(history.history['val_accuracy'], 'orange', label='val_accuracy')
    plt.legend(loc='upper center')
    plt.title('lr-{0}, l1-{1}, l2-{2}, layers-{3}, dropout-{4}'.format(lr, l1, l2, layer_count, dropout))
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    print("output_path: ", output_path)
    plt.savefig(os.path.join(output_path, 'accuracy.png'))
    plt.close()

    hist_df = pd.DataFrame(history.history)

    hist_json = os.path.join(output_path, 'history.json')
    with open(hist_json, mode='w') as f:
        hist_df.to_json(f)

    hist_csv = os.path.join(output_path, 'history.csv')
    with open(hist_csv, mode='w') as f:
        hist_df.to_csv(f)

    min_val_loss = np.min(history.history['val_loss'])

    return min_val_loss

def get_viz():
    """
    This function generates the visualization of the neural network architecture
    """
    parameters = {}
    model = two_hlayer_network(parameters)
    ann_viz(model, view = True, filename="arch1", title="DNN with 2 hidden layers")
    model = four_hlayer_network(parameters)
    ann_viz(model, view = True, filename="arch2", title="DNN with 4 hidden layers")
    model = eight_hlayer_network(parameters)
    ann_viz(model, view = True, filename="arch3", title="DNN with 8 hidden layers")


def tune_hyperparameters(X_train, Y_train, X_val, Y_val):
    
    print('[INFO]: STARTED Hyperparameter tuning')
    # Hyperparameter tuning values
    learning_rates = [0.01, 0.001, 0.0001]
    reg_types = ['l1', 'l2']
    reg_vals = [0.1, 0.01, 0.0001]
    dropouts = [0, 0.4]
    model_layers = [2, 4, 8]

    best_model_count = None
    best_model_val_loss = None
    best_parameters = None

    model_path = "nn_models"
    model_count = 1
    # Tune the hyperparameters for all the combinations
    for lr in learning_rates:
        for reg_type in reg_types:
            for reg_val in reg_vals:
                l1 = 0
                l2 = 0
                if reg_type == 'l1':
                    l1 = reg_val
                else:
                    l2 = reg_val
                for dropout in dropouts:
                    for model_layer in model_layers:
                        model = None
                        output_path = os.path.join(model_path, 'model_count_' + str(model_count))
                        parameters = {'lr': lr, 'l1': l1, 'l2': l2, 'do': dropout, 'ml': model_layer, 'o': output_path}
                        print('[INFO]: Started model training for parameters: \n', parameters)
                        
                        min_val_loss = train_model(X_train, Y_train, X_val, Y_val, parameters, epochs = 50)
                        if best_model_val_loss is None:
                            best_model_val_loss = min_val_loss
                            best_model_count = model_count
                            best_parameters = parameters
                        elif best_model_val_loss > min_val_loss:
                            best_model_val_loss = min_val_loss
                            best_model_count = model_count
                            best_parameters = parameters

                        model_count += 1

    print('[INFO]: COMPLETED Hyperparameter tuning')
    print('[INFO]; best parameters; ', best_parameters)
    print('[INFO]: best val loss: ', best_model_val_loss)
    

def find_best_model():
    min_val_loss = 10000
    best_model_count = -1
    val_losses = []
    model_count = 1
    for i in range(108):
        file_path = "nn_models/model_count_{0}/history.csv".format(str(model_count))
        print(file_path)
        f = open(file_path, "r")
        lines = f.readlines()
        val_loss = float(lines[-1].split(",")[1])
        train_loss = float(lines[-1].split(",")[2])
        diff = np.abs(train_loss - val_loss)

        val_losses.append(diff)
        if(val_loss < min_val_loss):

            min_val_loss = val_loss
            best_model_count = model_count
        model_count+=1
    print("minimum ind", np.argmin(val_losses))
    print("Best model count: {}".format(best_model_count))
    print("Min val loss: {}".format(min_val_loss))
    plt.plot(val_losses)
    plt.xlabel("Models")
    plt.ylabel("abs(Training loss - Validation loss)")
    plt.show() 



if __name__ == "__main__":
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
    tune_hyperparameters(X_train, Y_train, X_val, Y_val)