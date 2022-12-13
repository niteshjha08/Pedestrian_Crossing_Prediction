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

"""
    This function trains the model and returns the history of the model    
    """
def train_model2(X_train, Y_train, X_val, Y_val, parameters, epochs = 50):
    output_path = parameters['o']
    # Create the output directory if it does not exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    # Create the checkpoint to save the best model
    best_val_loss_checkpoint = ModelCheckpoint(filepath=output_path, monitor='val_loss', save_best_only=True, verbose=0,
                                               mode='min')
    # Create the early stopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=10)
    # Create the model
    model_layer = parameters['ml']
    parameters['in'] = np.array(X_train).shape[1]
    print(parameters['in'])
    
    if model_layer == 2:
        model = nn.model_arch1(parameters)
    elif model_layer == 4:
        model = nn.model_arch2(parameters)
    else:
        model = nn.model_arch3(parameters)

    history = model.fit(X_train, Y_train, epochs=epochs, batch_size=512,
                        validation_data=(X_val, Y_val), workers=16, callbacks=[best_val_loss_checkpoint], verbose=1)

    # history = model.fit(X_train, Y_train, epochs=epochs, batch_size=512)

    print(history.history.keys()) 
    min_val_loss = np.min(history.history['val_accuracy'])
    min_loss = np.min(history.history['accuracy'])
    
    return min_loss, min_val_loss

def generate_learning_curve(data_per_frame, parameters, epochs = 50):
    train_losses=  []
    val_losses=  []

    for i in np.arange(0.02, 0.03, 0.001):
    
        # np.random.shuffle(data_per_frame)
        partial_data = data_per_frame[:int(len(data_per_frame)*i)]
        print("=====================================")  
        print(len(partial_data))
        train_data, val_data = train_test_split(partial_data, train_size=cfg.TRAIN_SPLIT)

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

        print(X_train[0])

        pca_trans = du.get_pca(n_components=8)
        X_train = du.apply_pca(pca_trans, X_train)
        # print(list(X_train)[0])
        X_val = du.apply_pca(pca_trans, X_val)
        # exit()

        Y_train = np.array(Y_train)
        Y_val = np.array(Y_val)

        # X_train, X_val, X_test = du.apply_pca(8, X_train, X_val, X_test)
        # dataset = {'train_x': X_train, 'train_y': Y_train, 'val_x': X_val, 'val_y': Y_val}

        # parameters = {'lr': 0.01, 'l1': 0, 'l2': 0.01, 'do': 0, 'ml': 8, 'o': 'check_learning_curve'}
        print('[INFO]: Started model training for parameters: \n', parameters)
        
        min_training_loss, min_val_loss = train_model2(X_train, Y_train, X_val, Y_val, parameters, epochs)
        train_losses.append(min_training_loss)
        val_losses.append(min_val_loss)

    plt.plot(train_losses, label='Training loss')
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


    data_per_frame = du.arrange_data_per_frame(training_data, cfg.WINDOW_SIZE, feature_set=feature_set)
    random.shuffle(data_per_frame)

    parameters = {'lr': 0.01, 'l1': 0, 'l2': 0.01, 'do': 0, 'ml': 8, 'o': 'check_learning_curve'}
    generate_learning_curve(data_per_frame, parameters, epochs=50)