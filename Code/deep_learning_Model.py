from tqdm import tqdm

import torch
import torch.nn as nn
from torch.autograd import Variable


class CrossingPredictor(nn.Module):
    def __init__(self, time_window, hidden_sizes):
        super().__init__()
        self.num_features = 12
        self.input_size = time_window * self.num_features
        self.input_nodes = 64
        self.layers = nn.Sequential(nn.Linear(self.input_size, self.input_nodes),
                                   nn.ReLU(),
                                   nn.Dropout(p=0.5),
                                   nn.Linear(self.input_nodes, hidden_sizes[0]),
                                   nn.ReLU(),
                                   nn.Dropout(p=0.5),
                                   nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                                   nn.ReLU(),
                                   nn.Dropout(p=0.5),
                                   nn.Linear(hidden_sizes[1], hidden_sizes[2]),
                                   nn.ReLU(),
                                   nn.Dropout(p=0.5),
                                   nn.Linear(hidden_sizes[2], 1),
                                   nn.Sigmoid())

    def forward(self, X):
        X = self.layers(X)
        return X


def train(model, loss_crit, opt, dataloader, device):
    '''
    Method to define the training pipeline

    This pipeline uses data loaders to generate random data through entire training dataset once
    '''
    # initialize training mode
    model.train()
    pbar = tqdm(dataloader, position=0, leave=True)
    total_loss = 0
    count = 0
    # loop through entire dataset
    for sample in pbar:
        # initialize loss
        loss = 0.0
        count += 1

        # extract input and output from the batch loaded by the data loader
        feats = Variable(sample["features"].to(device))
        target = Variable(sample["output"].to(device))

        # forward pass and loss calculation
        pred = model(feats).squeeze(dim=1)
        loss = loss_crit(pred, target)

        # backpropagation
        opt.zero_grad()
        loss.backward()
        opt.step()

        # book-keeping of loss values
        total_loss += loss.item()
        pbar.set_description(
            "Loss {:.3f} | Avg Loss {:.3f}".format(loss.item(), total_loss / (4 * (count)))
        )


def validate(model, loss_crit, dataloader, device):
    '''
    Method to define the validation pipeline to predict class probaility on untrained data

    This pipeline uses data loaders to generate random data through entire testing dataset once
    '''
    # set model in validation mode
    model.eval()
    # reduce any accumulated gradients to zero
    with torch.no_grad():
        total_loss = 0
        count = 0
        pbar = tqdm(dataloader, position=0, leave=True)
        # loop through entire dataset
        for sample in pbar:
            # extract input and output from the batch loaded by the data loader
            feats = Variable(sample["features"].to(device))
            target = Variable(sample["output"].to(device))

            # forward pass and loss calculation
            pred = model(feats).squeeze(dim=1)
            loss = loss_crit(pred, target)

            # book-keeping of loss values
            total_loss += loss.item()
            count += 1
            pbar.set_description(
                "Loss {:.3f} | Avg Loss {:.3f}".format(loss.item(), total_loss / ((count)))
            )


def test(model, dataloader, device):
    # set model in validation mode
    model.eval()

    # initialize containers to hold ground truth and predictions
    ground_truth = []
    predicted = []

    # reduce any accumulated gradients to zero
    with torch.no_grad():
        #         pbar = tqdm(dataloader)
        pbar = dataloader
        count = 0
        # loop through entire dataset
        for sample in pbar:
            # extract input and output from the batch loaded by the data loader
            feats = Variable(sample["features"].to(device))
            target = Variable(sample["output"].to(device))

            # forward pass
            pred = model(feats).squeeze(dim=1)

            # ground truth and prediction book-keeping
            # predictions are saved in binary format
            ground_truth.append(target.item())
            predicted.append(float(pred.item() >= 0.5))
            count += 1
    #             pbar.set_description("Processing ... {}/{}".format(count, len(dataloader)))
    return ground_truth, predicted