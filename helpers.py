from typing import List

import torch
import torchvision
import torchvision.transforms as transforms

from data_loader import HarDataset, Group


def load_data(batch_size):
    """
    Function loads the HAR dataset
    :param batch_size: Number of datapoints in a single batch of data
    :return: Tuple of `train_set`, `train_loader`, `test_set`, `test_loader`.
    """

    # Load dataset
    trans = transforms.Compose([
        transforms.ToTensor()
    ])
    train_set = HarDataset(Group.TRAIN, transform=trans)
    # [batch_size, 1, signal_length, num_features]

    # Make a dataloader
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size)

    # Load dataset
    test_set = HarDataset(Group.TEST, transform=trans)
    # [batch_size, 1, signal_length, num_features]

    # Make a dataloader
    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_size)

    return train_set, train_loader, test_set, test_loader


def my_confusion_matrix(y: List[float], y_hat: List[float], num_classes: int):
    cm = torch.zeros((num_classes, num_classes), dtype=torch.int64)

    for y_i, y_hat_i in zip(y, y_hat):
        cm[y_i][y_hat_i] += 1

    return cm


def get_labels_and_predictions(model, data_loader):
    labels, predictions = [], []

    for batch in data_loader:

        # Check if model is on GPU
        if next(model.parameters()).is_cuda:
            X, y = batch[0].to("cuda"), batch[1].to("cuda")
        else:
            X, y = batch

        y_hat = model(X)
        y_hat = y_hat.argmax(dim=1)

        labels.extend(y.tolist())
        predictions.extend(y_hat.tolist())

    return labels, predictions


def compute_acc(y: List[float], y_hat: List[float]) -> float:
    t = torch.tensor(y)
    t_hat = torch.tensor(y_hat)

    return (t_hat.eq(t).sum().item()) / len(t)
