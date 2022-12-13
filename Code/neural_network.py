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



def model_arch1(parameters):
    lr = parameters.get('lr', 0.01)
    l1 = parameters.get('l1', 1)
    l2 = parameters.get('l2', 0)
    do = parameters.get('do', 0.2)
    in_dim = parameters.get('in', 12)

    model = keras.Sequential()
    model.add(layers.Dense(12, input_dim=in_dim, activation='relu'))
    
    if l1!=0:
        model.add(layers.Dense(32, kernel_regularizer=keras.regularizers.l1(l1), activation='relu'))
    elif l2 != 0:
        model.add(layers.Dense(32, kernel_regularizer=keras.regularizers.l2(l2), activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss='binary_crossentropy')
    
    return model


def model_arch2(parameters):
    lr = parameters.get('lr', 0.01)
    l1 = parameters.get('l1', 0)
    l2 = parameters.get('l2', 0.1)
    do = parameters.get('do', 0)
    in_dim = parameters.get('in', 12)

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
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss='binary_crossentropy')
    
    return model


def model_arch3(parameters):
    lr = parameters.get('lr', 0.01)
    l1 = parameters.get('l1', 0.1)
    l2 = parameters.get('l2', 0)
    do = parameters.get('do', 0)
    in_dim = parameters.get('in', 12)

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
    output_path = parameters['o']
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    best_val_loss_checkpoint = ModelCheckpoint(filepath=output_path, monitor='val_loss', save_best_only=True, verbose=0,
                                               mode='min')
    early_stopping = EarlyStopping(monitor='loss', patience=5)
    model_layer = parameters['ml']

    parameters['in'] = np.array(X_train).shape[1]

    if model_layer == 2:
        model = model_arch1(parameters)
    elif model_layer == 4:
        model = model_arch2(parameters)
    else:
        model = model_arch3(parameters)

    history = model.fit(X_train, Y_train, epochs=epochs, batch_size=60000,
                        validation_data=(X_val, Y_val), workers=4, callbacks=[best_val_loss_checkpoint])

    plt.plot(history.history['loss'], 'b', label='train_loss')
    plt.plot(history.history['val_loss'], 'orange', label='val_loss')
    plt.legend(loc='upper center')
    lr = parameters['lr']
    l1 = parameters['l1']
    l2 = parameters['l2']
    layer_count = parameters['ml']
    dropout = parameters['do']
    plt.title('lr-{0}, l1-{1}, l2-{2}, layers-{3}, dropout-{4}'.format(lr, l1, l2, layer_count, dropout))
    plt.xlabel('epochs')
    plt.ylabel('binary_crossentropy')
    print("output_path: ", output_path)
    plt.savefig(os.path.join(output_path, 'bce_loss.png'))
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
    parameters = {}
    model = model_arch1(parameters)
    ann_viz(model, view = True, filename="arch1", title="DNN with 2 hidden layers")
    model = model_arch2(parameters)
    ann_viz(model, view = True, filename="arch2", title="DNN with 4 hidden layers")
    model = model_arch3(parameters)
    ann_viz(model, view = True, filename="arch3", title="DNN with 8 hidden layers")

if __name__ == "__main__":

    # get_viz()
    # exit()
    train_data_file_path = "../Data/crossing_dataset_train.csv"
    training_data = du.load_data(train_data_file_path)

    test_data_file_path = "../Data/crossing_dataset_test.csv"
    testing_data = du.load_data(test_data_file_path)

    feature_set = cfg.FEATURE_SET_FULL

    # feature_set = ['stopped', 'handwave', 'moving fast', 'looking', 'standing', 'nod', 'speed up', 'bounding_boxes']

    data_per_frame = du.arrange_data_per_frame(training_data, cfg.WINDOW_SIZE, feature_set=feature_set)
    random.shuffle(data_per_frame)

    test_data = du.arrange_data_per_frame(testing_data, cfg.WINDOW_SIZE, feature_set=feature_set)

    # split the dataset into training-test and input-output
    train_data, val_data = train_test_split(data_per_frame, train_size=cfg.TRAIN_SPLIT, random_state=10)

    X_train = []
    Y_train = []
    for data_pt in train_data:
        X_train.append(data_pt[4:])
        Y_train.append(data_pt[3])

    X_val = []
    Y_val = []
    for data_pt in val_data:
        X_val.append(data_pt[4:])
        Y_val.append(data_pt[3])

    X_test = []
    Y_test = []
    for data_pt in test_data:
        X_test.append(data_pt[4:])
        Y_test.append(data_pt[3])

    pca_trans = du.get_pca(n_components=8)
    X_train = du.apply_pca(pca_trans, X_train)
    X_val = du.apply_pca(pca_trans, X_val)
    X_test = du.apply_pca(pca_trans, X_test)
    Y_train, Y_val, Y_test = np.array(Y_train), np.array(Y_val), np.array(Y_test)
    print("Training set length: ", len(X_train))
    print("Validation set length: ", len(X_val))
    print("Testing set length: ", len(X_test))

    print('[INFO]: STARTED Hyperparameter tuning')
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
                        print("output_path: ", output_path)
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
    print('[INFO]: best model count: ', best_model_count)
    print('[INFO]: Train Model Count {0} on full data on parameters {1}'.format(best_model_count, best_parameters))
