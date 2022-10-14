import ast
import torch
import torch.nn as nn


# TODO: double check everything here in this class
class CNN(nn.Module):
    def __init__(self, model_params):
        super().__init__()

        variant = model_params['variant']
        filter1, filter2, filter3 = ast.literal_eval(str(model_params['filter_num']))
        kernel1, kernel2 = ast.literal_eval(str(model_params['filter_size12']))

        self.variant = variant
        self.relu = nn.LeakyReLU() if 'LeakyReLUModel' in variant else nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=filter1, kernel_size=(kernel1,kernel1), stride=1, padding='valid')
        self.bn1 = nn.BatchNorm2d(num_features=filter1) if 'BatchNormModel' in variant else nn.Identity()

        self.conv2 = nn.Conv2d(in_channels=filter1, out_channels=filter2, kernel_size=(kernel2,kernel2), stride=1, padding='valid')
        self.bn2 = nn.BatchNorm2d(num_features=filter2) if 'BatchNormModel' in variant else nn.Identity()

        self.conv3 = nn.Conv2d(in_channels=filter2, out_channels=filter3, kernel_size=(3,3), stride=1, padding='valid')

        if model_params['filter_size12'] == '(5,3)':
            fc_in_features = filter3*2*2
        elif model_params['filter_size12'] == '(7,5)':
            fc_in_features = filter3*1*1
        self.fc1 = nn.Linear(in_features=fc_in_features, out_features=512)
        self.dropout = nn.Dropout(p=0.3) if 'DropoutModel' in variant else nn.Identity()
        # self.bn = nn.BatchNorm2d(num_features=512) if 'BatchNormModel' in variant else nn.Identity()

        self.fc_optional = nn.Linear(in_features=512, out_features=128) if 'ConnectedLayerModel' in variant else nn.Identity()
        self.relu_optional = nn.ReLU() if 'ConnectedLayerModel' in variant else nn.Identity()
        # self.bn_optional = nn.BatchNorm2d(num_features=128) if 'BatchNormModel' in variant else nn.Identity()

        if 'ConnectedLayerModel' in variant:
            self.fc2 = nn.Linear(in_features=128, out_features=10)
        else:
            self.fc2 = nn.Linear(in_features=512, out_features=10)
        self.logsoftmax = nn.LogSoftmax()

    def forward(self, x):
        # x size torch.Size([64, 3, 32, 32])
        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)
        x = self.maxpool(x)
        # x size (32-5+1)/2 = 14 torch.Size([64, 24, 14, 14])
        # x size (32-7+1)/2 = 13 torch.Size([64, 24, 13, 13])

        x = self.conv2(x)
        x = self.relu(x)
        x = self.bn2(x)
        x = self.maxpool(x)
        # x size (14-3+1)/2 = 6 torch.Size([64, 48, 6, 6])
        # x size (13-5+1)/2 = 4 torch.Size([64, 48, 4, 4])

        x = self.conv3(x)
        x = self.maxpool(x)
        # x size = (6-3+1)/2 = 2 torch.Size([64, 96, 2, 2])
        # x size = (4-3+1)/2 = 1 torch.Size([64, 96, 1, 1])

        # x = self.avgpool(x)
        x = torch.flatten(x, 1)  # torch.Size([64, 384])
        
        x = self.fc1(x)
        x = self.relu(x)
        # x = self.bn(x)
        x = self.dropout(x)
        if 'ConnectedLayerModel' in self.variant:
            x = self.fc_optional(x)
            x = self.relu_optional(x)
            # x = self.bn_optional(x)

        x = self.fc2(x)
        x = self.logsoftmax(x)

        return x



