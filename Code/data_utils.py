import ast
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np

import config as cfg


def load_data(file_path):
    data_frame = pd.read_csv(file_path)
    print("[INFO] Number of pedestrians in the dataset : ", len(data_frame))
    for col_name in ['bounding_boxes', 'frame_numbers', 'moving slow', 'stopped', 'handwave', 'look', 'clear path',
                     'crossing', 'moving fast', 'looking', 'standing', 'slow down', 'nod', 'speed up']:
        data_frame[col_name] = data_frame[col_name].apply(ast.literal_eval)
    return data_frame


def explore_data(data_frame):
    count = 0
    for _, row in data_frame.iterrows():
        count += len(row['frame_numbers'])
    print("Number of Pedestrian-Frames: %d" % count)

    # plot the distribution of crossing
    columns = []
    for col in data_frame.columns:
        if col != 'video_id' and col != 'ped_ind' and col != 'frame_numbers' and col != 'cross_overall' and col != 'crossing':
            columns.append(col)
    print("[INFO] Features : ", columns)
    print("[INFO] Number of features : ", len(columns))

    # select a random row from the data frame
    idx = random.randint(0, len(data_frame))

    print("[INFO] For Video : {} ; Pedestrian {}".format(data_frame.iloc[idx]['video_id'],
                                                         data_frame.iloc[idx]['ped_ind']))

    # plot the features vs crossing across all frames
    fig, axs = plt.subplots(4, 3, figsize=(20, 20))
    img_w = cfg.IMG_W
    for col, ax in zip(columns, axs.ravel()):
        # handle bounding boxes data separately
        if col == 'bounding_boxes':
            xcoords = []
            for bbox in data_frame.iloc[idx][col]:
                # normalize the x-coordinate
                xcoords.append(bbox[0] / img_w)
            ax.plot(data_frame.iloc[idx]['frame_numbers'], xcoords)
        else:
            ax.plot(data_frame.iloc[idx]['frame_numbers'], data_frame.iloc[idx][col])
        ax.plot(data_frame.iloc[idx]['frame_numbers'], data_frame.iloc[idx]['crossing'], linestyle='dashed')
        ax.set_title('Plot : {} vs crossing'.format(col))
        ax.set_xlabel("Frames ==>")
        ax.legend([col, 'crossing'])
    fig.show()
    plt.show()


def normalize_bbox_x(bboxes, img_w):
    """
    Function to normalize the bounding box x-coordinate
    """
    normalized_values = []
    for i in range(len(bboxes)):
        normalized_values.append(
            round(int(float(bboxes[i][0]) + float(bboxes[i][2]) * 0.5) / img_w, 3))
    return normalized_values


def unravel_data(data, window_size, feature_set=cfg.FEATURE_SET_FULL, img_w=cfg.IMG_W):
    per_frame_data = []
    # loop through every pedestrian and unravel to frame-wise data
    for _, row in data.iterrows():
       # get the features to be used for the model
        feature_list = feature_set

        # get the frame values and output for each video and each pedestrian
        frames = row['frame_numbers']
        outputs = row['crossing']

        # get the feature for each pedestrian and normalize the bounding box x-coordinate
        features = {}
        for feat in feature_list[:-1]:
            features[feat] = row[feat]
        features['bounding_boxes'] = normalize_bbox_x(row['bounding_boxes'], img_w)

        # loop through every frame
        for j in range(len(frames)):
            if j >= window_size:
                # initialize the data row with the video id, pedestrian id and frame number
                data_row = [row['video_id'], row['ped_ind'], frames[j], outputs[j]]

                # populate the data row with the features of the current frame
                for k in range(window_size, 0, -1):
                    for feat in feature_list:
                        if feat == 'bounding_boxes':
                            data_row.append(features[feat][j - k])
                        else:
                            data_row.append(int(features[feat][j - k]))
                data_row[3] = int(data_row[3])
                # append the data row to the list
                per_frame_data.append(data_row)
    return per_frame_data


def get_pca(n_components):
    """
    Function to get PCA with n_components
    """
    pca = PCA(n_components=n_components)
    return pca 


def apply_pca(pca, X):
    """
    Function to apply PCA on the features data with n_components
    """
    X_reduced = pca.fit_transform(X)

    return X_reduced

def evaluate_principal_components(X_features):
    """
    Function to evaluate the principal components of the features data with a number of components, 
    calculate the information loss, and plot the feature correlation with the principal components.
    """
    # n_components = [1,2,3,4,5, 6, 7,8,9,10,11,12]
    n_components = [1, 4, 8, 12]

    explained_variances = []
    for n in n_components:
        pca = PCA(n_components=n)
        X_features_reduced = pca.fit_transform(X_features)
        explained_variance = pca.explained_variance_ratio_
        print("explained variance with {} components: ".format(n), sum(explained_variance))
        info_loss = 1 - sum(explained_variance)
        explained_variances.append(sum(explained_variance))
        print("information loss with {} components: ".format(n_components), info_loss)
        component_importance = np.array(pca.components_)
        plt.imshow(np.abs(component_importance), cmap='hot', interpolation='nearest')
        plt.xlabel(cfg.FEATURE_SET_FULL)
        plt.ylabel("Principal components")
        plt.colorbar()
        plt.show()
    # plt.bar(n_components, explained_variances)
    # plt.xlabel("Number of components")
    # plt.ylabel("Explained variance")
    # plt.show()

def find_number_of_components(X_features, variance_threshold=0.95):
    pca = PCA(n_components=variance_threshold)
    X_features_reduced = pca.fit_transform(X_features)
    print("number of features:",X_features_reduced.shape[1])


def split_input_and_output(per_frame_data):
    """
    Function to split the data into input and output
    """
    X = []
    Y = []
    for row in per_frame_data:
        X.append(row[4:])
        Y.append(row[3])
    return X, Y

if __name__=="__main__":

    train_data_file_path = "../Data/crossing_dataset_train.csv"
    training_data = load_data(train_data_file_path)

    test_data_file_path = "../Data/crossing_dataset_test.csv"
    testing_data = load_data(test_data_file_path)

    feature_set = cfg.FEATURE_SET_FULL

    data_per_frame = unravel_data(training_data, cfg.WINDOW_SIZE, feature_set=feature_set)
    random.shuffle(data_per_frame)

    from sklearn.model_selection import train_test_split

    train_data, val_data = train_test_split(data_per_frame, train_size=cfg.TRAIN_SPLIT)

    X_train = []
    Y_train = []
    print(train_data[0])
