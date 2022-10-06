import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, filter1=24, filter2=48, filter3=96, kernel1=(5, 5), kernel2=(3, 3), kernel3=(3, 3), **kwargs):
        super().__init__()
        self.DefaultModel = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=filter1, kernel_size=kernel1, stride=1, padding='valid'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),

            nn.Conv2d(in_channels=filter1, out_channels=filter2, kernel_size=kernel2, stride=1, padding='valid'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),

            nn.Conv2d(in_channels=filter2, out_channels=filter3, kernel_size=kernel3, stride=1, padding='valid'),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),

            nn.Linear(in_features=filter3, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=10),

            nn.LogSoftmax()
        )

        self.ConnectedLayerModel = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=filter1, kernel_size=kernel1, stride=1, padding='valid'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),

            nn.Conv2d(in_channels=filter1, out_channels=filter2, kernel_size=kernel2, stride=1, padding='valid'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),

            nn.Conv2d(in_channels=filter2, out_channels=filter3, kernel_size=kernel3, stride=1, padding='valid'),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),

            nn.Linear(in_features=filter3, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=10),

            nn.LogSoftmax()
        )

        self.LeakyReLUModel = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=filter1, kernel_size=kernel1, stride=1, padding='valid'),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),

            nn.Conv2d(in_channels=filter1, out_channels=filter2, kernel_size=kernel2, stride=1, padding='valid'),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),

            nn.Conv2d(in_channels=filter2, out_channels=filter3, kernel_size=kernel3, stride=1, padding='valid'),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),

            nn.Linear(in_features=filter3, out_features=512),
            nn.LeakyReLU(),
            nn.Linear(in_features=512, out_features=10),

            nn.LogSoftmax()

        )

        self.DropoutModel = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=filter1, kernel_size=kernel1, stride=1, padding='valid'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),

            nn.Conv2d(in_channels=filter1, out_channels=filter2, kernel_size=kernel2, stride=1, padding='valid'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),

            nn.Conv2d(in_channels=filter2, out_channels=filter3, kernel_size=kernel3, stride=1, padding='valid'),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),

            nn.Linear(in_features=filter3, out_features=512),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(in_features=512, out_features=10),

            nn.LogSoftmax()

        )

        self.batchNormModel = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=filter1, kernel_size=kernel1, stride=1, padding='valid'),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=filter1),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),

            nn.Conv2d(in_channels=filter1, out_channels=filter2, kernel_size=kernel2, stride=1, padding='valid'),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=filter2),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),

            nn.Conv2d(in_channels=filter2, out_channels=filter3, kernel_size=kernel3, stride=1, padding='valid'),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),

            nn.Linear(in_features=filter3, out_features=512),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=512),
            nn.Linear(in_features=512, out_features=10),

            nn.LogSoftmax()

        )


