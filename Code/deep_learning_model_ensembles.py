import torch.nn as nn


class CrossingPredictor1(nn.Module):
    def __init__(self, time_window, hidden_sizes, num_features, input_nodes, dropout=None):
        super().__init__()
        self.num_features = num_features
        self.input_size = time_window * self.num_features
        self.input_nodes = input_nodes
        if dropout is None:
            self.layers = nn.Sequential(nn.Linear(self.input_size, self.input_nodes),
                                        nn.ReLU(),
                                        nn.Linear(self.input_nodes, hidden_sizes[0]),
                                        nn.ReLU(),
                                        nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                                        nn.ReLU(),
                                        nn.Linear(hidden_sizes[1], 1),
                                        nn.Sigmoid())
        else:
            self.layers = nn.Sequential(nn.Linear(self.input_size, self.input_nodes),
                                        nn.ReLU(),
                                        nn.Dropout(p=dropout),
                                        nn.Linear(self.input_nodes, hidden_sizes[0]),
                                        nn.ReLU(),
                                        nn.Dropout(p=dropout),
                                        nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                                        nn.ReLU(),
                                        nn.Dropout(p=dropout),
                                        nn.Linear(hidden_sizes[1], 1),
                                        nn.Sigmoid())

    def forward(self, X):
        X = self.layers(X)
        return X


class CrossingPredictor2(nn.Module):
    def __init__(self, time_window, hidden_sizes, num_features, input_nodes, dropout=None):
        super().__init__()
        self.num_features = num_features
        self.input_size = time_window * self.num_features
        self.input_nodes = input_nodes
        if dropout is None:
            self.layers = nn.Sequential(nn.Linear(self.input_size, self.input_nodes),
                                        nn.ReLU(),
                                        nn.Linear(self.input_nodes, hidden_sizes[0]),
                                        nn.ReLU(),
                                        nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                                        nn.ReLU(),
                                        nn.Linear(hidden_sizes[1], hidden_sizes[2]),
                                        nn.ReLU(),
                                        nn.Linear(hidden_sizes[2], 1),
                                        nn.Sigmoid())
        else:
            self.layers = nn.Sequential(nn.Linear(self.input_size, self.input_nodes),
                                        nn.ReLU(),
                                        nn.Dropout(p=dropout),
                                        nn.Linear(self.input_nodes, hidden_sizes[0]),
                                        nn.ReLU(),
                                        nn.Dropout(p=dropout),
                                        nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                                        nn.ReLU(),
                                        nn.Dropout(p=dropout),
                                        nn.Linear(hidden_sizes[1], hidden_sizes[2]),
                                        nn.ReLU(),
                                        nn.Dropout(p=dropout),
                                        nn.Linear(hidden_sizes[2], 1),
                                        nn.Sigmoid())

    def forward(self, X):
        X = self.layers(X)
        return X


class CrossingPredictor3(nn.Module):
    def __init__(self, time_window, hidden_sizes, num_features, input_nodes, dropout=None):
        super().__init__()
        self.num_features = num_features
        self.input_size = time_window * self.num_features
        self.input_nodes = input_nodes
        if dropout is None:
            self.layers = nn.Sequential(nn.Linear(self.input_size, self.input_nodes),
                                        nn.ReLU(),
                                        nn.Linear(self.input_nodes, hidden_sizes[0]),
                                        nn.ReLU(),
                                        nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                                        nn.ReLU(),
                                        nn.Linear(hidden_sizes[1], hidden_sizes[2]),
                                        nn.ReLU(),
                                        nn.Linear(hidden_sizes[2], hidden_sizes[3]),
                                        nn.ReLU(),
                                        nn.Linear(hidden_sizes[3], 1),
                                        nn.Sigmoid())
        else:
            self.layers = nn.Sequential(nn.Linear(self.input_size, self.input_nodes),
                                        nn.ReLU(),
                                        nn.Dropout(p=dropout),
                                        nn.Linear(self.input_nodes, hidden_sizes[0]),
                                        nn.ReLU(),
                                        nn.Dropout(p=dropout),
                                        nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                                        nn.ReLU(),
                                        nn.Dropout(p=dropout),
                                        nn.Linear(hidden_sizes[1], hidden_sizes[2]),
                                        nn.ReLU(),
                                        nn.Dropout(p=dropout),
                                        nn.Linear(hidden_sizes[2], hidden_sizes[3]),
                                        nn.ReLU(),
                                        nn.Dropout(p=dropout),
                                        nn.Linear(hidden_sizes[3], 1),
                                        nn.Sigmoid())

    def forward(self, X):
        X = self.layers(X)
        return X
