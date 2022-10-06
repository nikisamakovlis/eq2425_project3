import ast
import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, model_params):
        super().__init__()
        variant = model_params['variant']
        filter1, filter2, filter3 = ast.literal_eval(str(model_params['filter_num']))
        kernel1, kernel2 = ast.literal_eval(str(model_params['filter_size12']))

        self.variant = variant
        self.relu = nn.LeakyReLU() if variant == 'LeakyReLUModel' else nn.ReLu()
        self.maxpool = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=filter1, kernel_size=(kernel1,kernel1), stride=1, padding='valid')
        self.bn1 = nn.BatchNorm2d(num_features=filter1) if variant == 'batchNormModel' else nn.Identity()

        self.conv2 = nn.Conv2d(in_channels=filter1, out_channels=filter2, kernel_size=(kernel2,kernel2), stride=1, padding='valid')
        self.bn2 = nn.BatchNorm2d(num_features=filter2) if variant == 'batchNormModel' else nn.Identity()

        self.conv3 = nn.Conv2d(in_channels=filter2, out_channels=filter3, kernel_size=(3,3), stride=1, padding='valid')

        self.fc1 = nn.Linear(in_features=filter3, out_features=512)
        self.dropout = nn.Dropout(p=0.3) if variant == 'DropoutModel' else nn.Identity()
        self.bn = nn.BatchNorm2d(num_features=512) if variant == 'batchNormModel' else nn.Identity()

        self.fc_optional = nn.Linear(in_features=512, out_features=128)
        self.relu_optional = nn.ReLU()

        self.fc2 = nn.Linear(in_features=512, out_features=10)
        self.logsoftmax = nn.LogSoftmax()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)
        x = self.maxpool(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.bn2(x)
        x = self.maxpool(x)

        x = self.conv3(x)
        x = self.maxpool(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.bn(x)
        if self.variant == 'ConnectedLayerModel':
            x = self.fc_optional(x)
            x = self.relu_optional(x)
        x = self.fc2(x)
        x = self.logsoftmax(x)

        return x



