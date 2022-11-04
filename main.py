from collections import OrderedDict

import torch
from matplotlib import pyplot as plt
import seaborn as sn
from torchvision import transforms

from data_loader import HarDataset, Group
from helpers import load_data, get_labels_and_predictions, my_confusion_matrix
from training import train


class TaskRunner:

    @staticmethod
    def run(task_num=1):
        train_set, train_loader, test_set, test_loader = load_data(1000)

        if task_num == 1:
            # Visualize
            pass
            X_train, y_train = next(iter(train_loader))
            print(X_train.shape)
            print(y_train.shape)

            labels = {
                1: "WALKING",
                2: "WALKING_UPSTAIRS",
                3: "WALKING_DOWNSTAIRS",
                4: "SITTING",
                5: "STANDING",
                6: "LAYING"
            }
            X_train = torch.transpose(X_train, 2, 3).contiguous()
            for feature_index in range(9):
                for y in range(6):
                    y_idx = torch.where(y_train == y)[0][0]  # take a single sample of class `y`
                    plt.plot(X_train[y_idx].squeeze()[feature_index], label=labels[y + 1])
                    plt.legend()

                plt.title(f"Signal_{feature_index}")
                plt.savefig(f"./results/Signal_{feature_index}")
                plt.show()

        elif task_num == 2:
            parameters = OrderedDict(
                lr=[0.01, 0.001],
                batch_size=[100, 1000],
                kernel_size=[9],
                network=["cnn"]
            )

            train(parameters, train_set, test_set, num_epochs=30, device="cuda")

        elif task_num == 3:
            net = torch.load("./models/Run(lr=0.001, batch_size=100, kernel_size=5, network='cnn')")
            y, y_hat = get_labels_and_predictions(model=net, data_loader=test_loader)
            cm = my_confusion_matrix(y, y_hat, num_classes=6)

            plt.figure(figsize=(10, 8))
            plt.title(f"Confusion matrix, {'Run(lr=0.001, batch_size=100, kernel_size=5, network=cnn)'}")
            sn.heatmap(cm, annot=True, cmap="Blues")
            plt.savefig("./results/confusion_matrix")


if __name__ == "__main__":
    TaskRunner.run(3)