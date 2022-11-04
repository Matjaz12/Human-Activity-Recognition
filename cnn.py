import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from data_loader import HarDataset, Group


def transform_tensor(t):
    """
    Function applies the following transformations to the tensor `t`.
    1. Loose the channel dimension
        Given an input tensor `t` of shape [batch_size, 1, signal_length, num_features],
        Returns tensor `t` of shape [batch_size, signal_length, num_features]

    2. Transpose along axes: 1 and 2.
        `PyTorch` expects the features as rows, and values as columns.
        rows ... features
        cols ... values

        t[i] ... signal `s_i`
        t[i][j] ... j-th value of signal `s_i`

        After transformation `t` is of shape: [batch_size, num_features, signal_length]

    :param t: Input tensor of shape [batch_size, 1, signal_length, num_features]
    :return: Transformed tensor [batch_size, num_features, signal_length]
    """

    # 1. Loose the channel dimension
    t = torch.squeeze(t, dim=1)

    # 2. Transpose along axes: 1 and 2.
    t = torch.transpose(t, 1, 2).contiguous()

    return t


class CNN(nn.Module):
    def __init__(self, kernel_size=5):
        super(CNN, self).__init__()

        # Convolutional layers
        # We convolve over the length of our sequence with a kernel of size `kernel_size`
        self.conv1 = nn.Conv1d(in_channels=9, out_channels=18, kernel_size=kernel_size)
        self.conv2 = nn.Conv1d(in_channels=18, out_channels=36, kernel_size=kernel_size)

        # Fully connected layers
        self.fc1 = nn.Linear(in_features=4320, out_features=2160)
        self.fc2 = nn.Linear(in_features=2160, out_features=1080)

        # Output layers
        self.out = nn.Linear(in_features=1080, out_features=6)

    def forward(self, t):
        # Transform input tensor
        t = transform_tensor(t)

        # (1) Input layer
        t = t

        # (2) convolutional layer
        t = t.float()
        t = self.conv1(t)
        t = F.relu(t)

        # (3) convolutional layer
        t = self.conv2(t)
        t = F.relu(t)

        # (4) fully connected layer
        t = t.reshape(-1, t.shape[1] * t.shape[2])
        t = self.fc1(t)
        t = F.relu(t)

        # (5) fully connected layer
        t = self.fc2(t)
        t = F.relu(t)

        # (5) output layer
        t = self.out(t)

        return t


if __name__ == "__main__":
    # Load dataset
    trans = transforms.Compose([
        transforms.ToTensor()
    ])
    train_set = HarDataset(Group.TRAIN, transform=trans)

    # [1, 128, 9]
    X_train, y_train = next(iter(train_set))

    # Init network
    network = CNN()

    # [1, 1, 128, 9]
    X_train = X_train.unsqueeze(dim=0)
    y_hat = network(X_train.float())
    print(y_hat)
