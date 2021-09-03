import torch
import torch.nn as nn
import torch.nn.functional as F

class Q_Network(nn.Module):

    def __init__(self, num_frame, num_action):
        super(Q_Network, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=num_frame, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(64*7*7, 512)
        self.fc2 = nn.Linear(512, num_action)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 64*7*7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

if __name__ == '__main__':
    net = Q_Network(4, 3)
    x = torch.ones(10, 4, 84, 84)
    out = net.forward(x)