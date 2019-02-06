## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Net(nn.Module):
    def __init__(self, input_dim):
        super(Net, self).__init__()

        self.input_dim = input_dim

        self.output_size = 136  # 136 as suggested

        self.model = nn.Sequential(
            nn.Conv2d(self.input_dim, 32, 5, 1),
            nn.Conv2d(32, 32, 2, 2),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 5, 1),
            nn.Conv2d(64, 64, 2, 2),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 5, 1),
            nn.Conv2d(128, 128, 2, 2),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 5, 1),
            nn.Conv2d(256, 256, 2, 2),
            nn.ReLU(True),
            nn.Conv2d(256, 512, 5, 1),
            nn.Conv2d(512, 512, 2, 2),
            nn.ReLU(True),
            nn.Conv2d(512, self.output_size, 1, 1),
            nn.AdaptiveAvgPool2d((1, 1))
        )

    def forward(self, x):
        x = self.model(x)
        return x.view(x.size(0), -1)
