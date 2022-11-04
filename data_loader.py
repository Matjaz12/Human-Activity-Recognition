from enum import Enum

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from torchvision import transforms


class Group(Enum):
    TRAIN = "train"
    TEST = "test"

    def __str__(self):
        return str(self.value)


def load_dataset(group: Group):
    """
    Function loads data in numpy format using the provided data group.
    :param group: Data group enum
    :return: numpy array `X` of shape [num_samples, signal_length, num_features],
    numpy array `y` of shape [num_samples, 1].
    """

    group_str = str(group)

    if group_str != "train" and group_str != "test":
        return None, None

    filepath = "data/" + group_str + "/Inertial Signals/"

    # Load all 9 files as a single array
    filenames = []

    # total acceleration
    filenames += ['total_acc_x_' + group_str + '.txt', 'total_acc_y_' + group_str + '.txt', 'total_acc_z_' + group_str + '.txt']
    # body acceleration
    filenames += ['body_acc_x_' + group_str + '.txt', 'body_acc_y_' + group_str + '.txt', 'body_acc_z_' + group_str + '.txt']
    # body gyroscope
    filenames += ['body_gyro_x_' + group_str + '.txt', 'body_gyro_y_' + group_str + '.txt', 'body_gyro_z_' + group_str + '.txt']

    # Load input data
    loaded_data = []
    for filename in filenames:
        df = pd.read_csv(filepath + filename, header=None, delim_whitespace=True)
        loaded_data.append(df)

    X = np.dstack(loaded_data)
    y = pd.read_csv("data/" + group_str + "/y_" + group_str + ".txt", header=None, delim_whitespace=True).to_numpy()
    y = y - 1  # zero offset labels
    y = y.flatten()

    return X, y


class HarDataset(torch.utils.data.Dataset):
    def __init__(self, group, transform=None):
        self.X, self.y = load_dataset(group)
        self.transform = transform

    def __getitem__(self, idx):
        X = self.X[idx]
        y = self.y[idx]

        if self.transform is not None:
            X = self.transform(X)

        return X, y

    def __len__(self):
        return self.y.shape[0]


if __name__ == "__main__":
    # Load dataset
    trans = transforms.Compose([
        transforms.ToTensor()
    ])
    train_set = HarDataset(Group.TRAIN, transform=trans)
    X_train, y_train = next(iter(train_set))
    X_train = torch.transpose(X_train, 1, 2).contiguous()

    from scipy.fftpack import fft, ifft
    import numpy as np
    signal = X_train[0][7].squeeze().tolist()
    plt.plot(signal)
    plt.show()
    plt.plot(abs(fft(signal)))
    plt.show()

    # Make a dataloader
    train_loader = torch.utils.data.DataLoader(dataset=train_set,
                                               batch_size=1000)
    X_train, y_train = next(iter(train_loader))
    print(X_train.shape)
    print(y_train.shape)

    y1_idx = torch.where(y_train == 0)[0] # walking
    y2_idx = torch.where(y_train == 4)[0] # standing

    X_train = torch.transpose(X_train, 2, 3).contiguous()

    labels = {
        1: "WALKING",
        2: "WALKING_UPSTAIRS",
        3: "WALKING_DOWNSTAIRS",
        4: "SITTING",
        5: "STANDING",
        6: "LAYING"
    }

    feature_index = 0
    for y in range(6):
        y_idx = torch.where(y_train == y)[0][0]  # take a single sample of class `y`
        plt.plot(X_train[y_idx].squeeze()[feature_index], label=labels[y + 1])
        plt.legend()

    plt.title(f"Signal {feature_index}")
    plt.show()
