import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import f1_score
import torch.nn
import torch.cuda
import torch.utils.data

import config as cfg
import data_utils as du
import deep_learning_Model as dlm
import deep_learning_DataLoader as dld


def train_ml(model, x, y):
    model.fit(x, y)


def predict_ml(model, x):
    return model.predict(x)


def train_dl(model, dataset, epochs, lr):

    output = cfg.OUTPUT
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_data = dataset['train']
    val_data = dataset['val']

    train_loader = torch.utils.data.DataLoader(dld.CrossingDataset(train_data), batch_size=cfg.BATCH_SIZE, shuffle=True)
    val_loader = torch.utils.data.DataLoader(dld.CrossingDataset(val_data), batch_size=cfg.BATCH_SIZE, shuffle=False)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=cfg.MOMENTUM, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, np.arange(5, epochs, 5), gamma=0.1)
    loss_criteria = torch.nn.BCELoss()

    # carry on with training and validation loop
    for i in range(epochs):
        print("Epoch {:d}".format(i))

        dlm.train(model, loss_criteria, optimizer, train_loader, device)
        scheduler.step(i)

        # evaluate and save checkpoints at only specified intervals
        if i % cfg.VAL_EVERY == 0:
            print("EVALUATING")
            with torch.no_grad():
                dlm.validate(model, loss_criteria, val_loader, device)
            if not os.path.exists(output):
                os.makedirs(output)
            torch.save(model.state_dict(), os.path.join(output, "model_checkpoint.pth"))


def predict_dl(model, dataset):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_data = dataset['test']
    test_loader = torch.utils.data.DataLoader(dld.CrossingDataset(test_data), batch_size=1, shuffle=False)
    y_gt, y_pred = dlm.test(model, test_loader, device)
    return y_gt, y_pred


class Classifier:
    def __init__(self, algo_name, lr=1e-3, max_iter=1000):
        self.algo_name = algo_name
        self.lr = lr
        self.max_iter = max_iter
        self.logistic_reg = None
        self.svm = None
        self.dl = None

    def fit(self, dataset, epochs=100, verbose=False):
        if self.algo_name == "logistic_reg":
            # initialize self.logistic_reg and fit
            self.logistic_reg = LogisticRegression(verbose=verbose, max_iter=self.max_iter, solver='newton-cg')
            train_ml(self.logistic_reg, dataset['train_x'], dataset['train_y'])

        elif self.algo_name == "svm":
            # initialize self.svm and fit
            self.svm = SVC(verbose=verbose, max_iter=self.max_iter)
            train_ml(self.svm, dataset['train_x'], dataset['train_y'])

        elif self.algo_name == "deep_learning":
            # initialize self.dl and fit
            self.dl = dlm.CrossingPredictor(cfg.WINDOW_SIZE, cfg.HIDDEN_SIZE)
            train_dl(self.dl, dataset, epochs, self.lr)

    def predict(self, dataset):
        if self.algo_name == "logistic_reg":
            # use self.logistic_reg and predict
            y = predict_ml(self.logistic_reg, dataset['test_x'])

        elif self.algo_name == "svm":
            # use self.svm and predict
            y = predict_ml(self.svm, dataset['test_x'])

        elif self.algo_name == "deep_learning":
            # use self.dl and predict
            y = predict_dl(self.dl, dataset)
        else:
            y = None

        return y

    def calc_error(self, X, Y):
        if self.algo_name == "logistic_reg":
            # use self.logistic_reg and calculate error
            pass

        elif self.algo_name == "svm":
            # use self.svm and calculate error
            pass

        elif self.algo_name == "deep_learning":
            # use self.dl and calculate error
            pass

    def learning_curve(self, X, Y):
        if self.algo_name == "logistic_reg":
            # use self.logistic_reg and compute learning curves
            pass

        elif self.algo_name == "svm":
            # use self.svm and compute learning curves
            pass

        elif self.algo_name == "deep_learning":
            # use self.dl and compute learning curves
            pass


def baseline_model(data):
    # unravel data for each frame in the dataset
    data_extract_window_size = 0
    data_per_frame = du.arrange_data_per_frame(data, data_extract_window_size)

    # define moving average window and initialize containers to hold predictions and ground truth
    Y_gt = []
    Y_pred = []
    moving_avg_window_size = 15
    prev_frames = []

    # loop through each frame
    for i in range(1, len(data_per_frame)):
        # check if the pedestrian data is same as previous
        if (data_per_frame[i][0] == data_per_frame[i - 1][0]) and (data_per_frame[i][1] == data_per_frame[i - 1][1]):
            # check if sufficient number of frames have been initialized for moving average
            if len(prev_frames) >= moving_avg_window_size:
                # aggregate ground truth
                Y_gt.append(data_per_frame[i][3])

                # aggregate the average prediction of previous frames and aggregrate it in output predictions
                last_frames_pred = int(sum(prev_frames) / len(prev_frames) >= 0.5)
                Y_pred.append(last_frames_pred)

                # move the sliding window
                prev_frames = prev_frames[1:]
                prev_frames.append(last_frames_pred)
            else:
                # re-initialize the previous frame container to store data of new pedestrian
                prev_frames.append(data_per_frame[i][3])
        else:
            prev_frames = []
    print("[INFO] Accuracy for moving average of size {} : {:.3f}% ".format(moving_avg_window_size,
                                                                            100 * f1_score(Y_gt, Y_pred)))