import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(),
            nn.ReLU(),
        )
