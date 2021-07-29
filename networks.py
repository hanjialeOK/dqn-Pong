import torch
import torch.nn as nn
import torch.nn.functional as F

class Q_Network(nn.Module):

    def __init__(self, num_frame, num_action):
        super(Q_Network, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=num_frame, out_channels=16, kernel_size=8, stride=4, padding=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2)
        self.fc1 = nn.Linear(32*9*9, 256)
        self.fc2 = nn.Linear(256, num_action)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 9*9*32)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# if __name__ == '__main__':
#     net = Q_Network(4, 2)
#     x = torch.ones(10, 4, 80, 80)
#     out = net.forward(x)