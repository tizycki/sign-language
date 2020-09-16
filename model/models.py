from torch import nn
import torch.nn.functional as F


class StopPoseNet(nn.Module):

    def __init__(self):
        super(StopPoseNet, self).__init__()

        self.fc1 = nn.Linear(114, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 128)
        self.fc4 = nn.Linear(128, 32)
        self.fc5 = nn.Linear(32, 2)

    def forward(self, x):
        output = x.view(x.size(0), -1)

        output = self.fc1(output)
        output = F.relu(output)
        F.dropout(output, 0.1)

        output = self.fc2(output)
        output = F.relu(output)
        F.dropout(output, 0.1)

        output = self.fc3(output)
        output = F.relu(output)
        F.dropout(output, 0.1)

        output = self.fc4(output)
        output = F.relu(output)
        F.dropout(output, 0.1)

        output = self.fc5(output)

        return output
