import random
import numpy as np
import matplotlib.pyplot as plt
import sklearn.model_selection
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import plot_roc_curve

import data_utils as du
import classifier as clf
import config as cfg


def start_pipeline():
    train_data_file_path = "../Data/crossing_dataset_train.csv"
    training_data = du.load_data(train_data_file_path)

    test_data_file_path = "../Data/crossing_dataset_test.csv"
    testing_data = du.load_data(test_data_file_path)

    # du.explore_data(data)

    # unravel the data for each pedestrian on per frame basis
    # window_size = 10

    # data_per_frame = du.arrange_data_per_frame(data, window_size)
    # print("[INFO] Total number of data points: ", len(data_per_frame))

    # clf.baseline_model(data)

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

    print("size train: ", len(X_train))
    print("size test: ", len(X_test))
    dataset = {'train_x': X_train, 'train_y': Y_train, 'val_x': X_val, 'val_y': Y_val}

    # train logistic regression model
    logistic_regression = clf.Classifier("logistic_reg", max_iter=50)
    logistic_regression.fit(dataset)

    Y_test_pred = logistic_regression.predict({'test_x': X_test})
    print("[INFO] For Logistic Regression classifier, F1 Score : {:.3f}%".format(f1_score(Y_test, Y_test_pred) * 100))
    #
    # plot_roc_curve(logistic_regression.logistic_reg, X_test, Y_test)
    # plt.show()

    # train svm classifier model
    # svm = clf.Classifier("svm")
    # svm.fit(dataset)

    # train deep learning classifier model
    # deep_learning = clf.Classifier("deep_learning")
    # dataset = {'train': train_data, 'val': val_data}
    # deep_learning.fit(dataset)
    #
    # dataset = {'test': test_data}
    # y_gt, y_pred = deep_learning.predict(dataset)
    # print("[INFO] For Deep Learning classifier, F1 Score : {:.3f}%".format(f1_score(y_gt, y_pred) * 100))


if __name__ == "__main__":
    start_pipeline()
