from collections import OrderedDict
from collections import namedtuple
from itertools import product

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from cnn import CNN
from helpers import get_labels_and_predictions, compute_acc


class RunBuilder:
    @staticmethod
    def get_runs(parameters: OrderedDict):
        """
        Function returns all permutations of parameters.
        :param parameters: Dictionary of parameters
        :return: List of all parameter permutations.
        """

        Run = namedtuple("Run", parameters.keys())
        runs = []

        for val in product(*parameters.values()):
            runs.append(Run(*val))

        return runs


class NetworkFactory:
    @staticmethod
    def get_network(network_name: str):
        """
        Function returns an instance of a network using provided `network_name`
        :param network_name: Name of the network to instantiate
        :return: Network instance
        """
        torch.manual_seed(100)

        net = None
        if network_name == "cnn":
            net = CNN()

        return net


class RunManager:
    def __init__(self, network_name, model, run):
        self.network_name = network_name
        self.model = model
        self.run = run
        self.epoch_list = []
        self.loss_list = []
        self.acc_list = []

        # Init run in `TensorBoard`
        self.tb = SummaryWriter(comment=f"-{run}")

    def add(self, epoch: int, loss: float, acc: float) -> None:
        """
        Function simply adds data to lists
        :param epoch: current epoch number
        :param loss: total loss during this epoch
        :param acc: accuracy during this epoch
        """
        print(f"epoch: {epoch} \t loss: {loss} \t acc: {acc}")

        epoch_count = len(self.epoch_list)
        self.tb.add_scalar("loss", loss, epoch_count)
        self.tb.add_scalar("accuracy", acc, epoch_count)

        for name, param in self.model.named_parameters():
            self.tb.add_histogram(name, param, epoch_count)
            self.tb.add_histogram(f"{name}.grad", param.grad, epoch_count)

        self.epoch_list.append(epoch)
        self.loss_list.append(loss)
        self.acc_list.append(acc)

    def plot(self):
        """
        Function plots accuracy and loss as a function of epochs and
        saves the graphs.
        """

        # Use backend that doesn't display plots.
        import matplotlib
        matplotlib.use('Agg')

        # Save loss and accuracy plots.
        fig = plt.figure()
        plt.plot(self.epoch_list, self.loss_list)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"{self.run}")
        fig.savefig(f"./results/{self.run}_loss.jpg")

        fig = plt.figure()
        plt.plot(self.epoch_list, self.acc_list)
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title(f"{self.run}")
        fig.savefig(f"./results/{self.run}_acc.jpg")

    def save(self, test_acc=None):
        with open(f"./logs/{self.run}", "w", encoding="utf-8") as file:
            for epoch, loss, acc in zip(self.epoch_list, self.loss_list, self.acc_list):
                file.write(f"epoch: {epoch} \t loss: {loss} \t acc: {acc}\n")
            if test_acc:
                file.write(f"(test set) acc: {test_acc}")

        torch.save(self.model, f"./models/{self.run}")


def train(parameters, train_set, test_set=None, num_epochs=20, device="cpu"):
    """
    Function trains a model / a set of models using provided parameters.

    :param parameters: OrderedDict of hyperparameters or models to try out.
    :param train_set: Train dataset
    :param test_set: Test dataset
    :param num_epochs: Number of epoch to perform
    :param device: Device on which training will take place
    """
    for run in RunBuilder.get_runs(parameters):
        comment = f"-{run}"
        print(comment)

        # Create network and optimizer
        net = NetworkFactory.get_network(run.network).to(device)
        optimizer = optim.Adam(net.parameters(), lr=run.lr)
        if net is None:
            continue

        # Fetch data
        train_loader = torch.utils.data.DataLoader(train_set,
                                                   batch_size=run.batch_size)

        run_manager = RunManager(network_name=run.network, model=net, run=run)
        for epoch in range(num_epochs):
            total_correct = 0
            total_loss = 0

            for batch in train_loader:
                X_train, y_train = batch[0].to(device), batch[1].to(device)

                y_hat = net(X_train)                        # Forward pass
                loss = F.cross_entropy(y_hat, y_train)      # Compute loss
                optimizer.zero_grad()                       # Clear previous gradients
                loss.backward()                             # Compute new gradients
                optimizer.step()                            # Update network weights

                total_loss += loss.item() / run.batch_size
                total_correct += y_hat.argmax(dim=1).eq(y_train).sum().item()

            run_manager.add(epoch, total_loss, total_correct / len(train_set))

        run_manager.plot()

        if test_set:
            test_loader = torch.utils.data.DataLoader(test_set, batch_size=1000)
            y, y_hat = get_labels_and_predictions(model=net, data_loader=test_loader)
            acc = compute_acc(y, y_hat)
            print(f"acc (on test data): {acc}")
            run_manager.save(acc)
        else:
            run_manager.save()
