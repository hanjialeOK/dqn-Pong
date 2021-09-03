import torch
import torch.nn as nn
import torch.nn.functional as F

class Q_Network(nn.Module):

    def __init__(self, num_frame, num_action):
        super(Q_Network, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=num_frame, out_channels=32, kernel_size=8, stride=4, padding=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        self.fc1_online = nn.Linear(64*7*7, 512)
        self.fc2_online = nn.Linear(512, num_action)
        self.fc1_target = nn.Linear(64*7*7, 512)
        self.fc2_target = nn.Linear(512, num_action)
        # self.conv1 = nn.Conv2d(in_channels=num_frame, out_channels=16, kernel_size=8, stride=4, padding=2)
        # self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2)
        # self.fc1_online = nn.Linear(32*9*9, 256)
        # self.fc2_online = nn.Linear(256, num_action)
        # self.fc1_target = nn.Linear(32*9*9, 256)
        # self.fc2_target = nn.Linear(256, num_action)
        for param in self.fc1_target.parameters():
            param.requires_grad_(False)
        for param in self.fc2_target.parameters():
            param.requires_grad_(False)

    def extract_feature(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 64*7*7)
        # x = x.view(-1, 32*9*9)
        return x

    def online_forward(self, x):
        x = self.extract_feature(x)
        x = F.relu(self.fc1_online(x))
        x = self.fc2_online(x)
        return x

    def target_forward(self, x):
        x = self.extract_feature(x)
        x = F.relu(self.fc1_target(x))
        x = self.fc2_target(x)
        return x

    def synchronize(self):
        self.fc1_target.weight.data.copy_(self.fc1_online.weight.data)
        self.fc1_target.bias.data.copy_(self.fc1_online.bias.data)
        self.fc2_target.weight.data.copy_(self.fc2_online.weight.data)
        self.fc2_target.bias.data.copy_(self.fc2_online.bias.data)

if __name__ == '__main__':
    net = Q_Network(4, 3)
    x = torch.ones(10, 4, 80, 80)
    out = net.online_forward(x)