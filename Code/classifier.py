import numpy as np
import sklearn as sk


class Classifier:
    def __init__(self, algo_name):
        self.algo_name = algo_name
        self.lr = None
        self.svm = None
        self.dl = None

    def fit(self, X, Y):
        if self.algo_name == "logistic_regression":
            # initialize self.lr and fit
            pass
        elif self.algo_name == "svm":
            # initialize self.svm and fit
            pass
        elif self.algo_name == "deep_learning":
            # initialize self.dl and fit
            pass

    def predict(self, X):
        if self.algo_name == "logistic_regression":
            # use self.lr and predict
            pass
        elif self.algo_name == "svm":
            # use self.svm and predict
            pass
        elif self.algo_name == "deep_learning":
            # use self.dl and predict
            pass

    def calc_error(self, X, Y):
        if self.algo_name == "logistic_regression":
            # use self.lr and calculate error
            pass
        elif self.algo_name == "svm":
            # use self.svm and calculate error
            pass
        elif self.algo_name == "deep_learning":
            # use self.dl and calculate error
            pass

    def learning_curve(self, X, Y):
        if self.algo_name == "logistic_regression":
            # use self.lr and compute learning curves
            pass
        elif self.algo_name == "svm":
            # use self.svm and compute learning curves
            pass
        elif self.algo_name == "deep_learning":
            # use self.dl and compute learning curves
            pass
