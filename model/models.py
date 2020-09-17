import torch
from torch import nn


class StopPoseNet(nn.Module):
    """
    Model for detecting stop poses from single frame
    """
    def __init__(self):
        """
        Architecture initialization
        """
        super(StopPoseNet, self).__init__()

        self.fc1 = nn.Linear(114, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 128)
        self.fc4 = nn.Linear(128, 32)
        self.fc5 = nn.Linear(32, 2)
        self.dropout1 = nn.Dropout(0.1)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass function
        :param x: input X tensor
        :return: tensor with prediction logits
        """
        output = x.view(x.size(0), -1)

        output = self.fc1(output)
        output = self.relu(output)
        self.dropout1(output)

        output = self.fc2(output)
        output = self.relu(output)
        self.dropout1(output)

        output = self.fc3(output)
        output = self.relu(output)
        self.dropout1(output)

        output = self.fc4(output)
        output = self.relu(output)
        self.dropout1(output)

        output = self.fc5(output)

        return output


class SequenceRecognitionNet(nn.Module):
    """
    Model for keypoints sequence classification
    """
    def __init__(self, out_class_num):
        """
        Architecture initialization
        :param out_class_num: number of output classes
        """
        super(SequenceRecognitionNet, self).__init__()

        self.fc1 = nn.Linear(5700, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, 128)
        self.fc5 = nn.Linear(128, out_class_num)
        self.dropout1 = nn.Dropout(0.1)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass function
        :param x: input X tensor
        :return: tensor with prediction logits
        """
        output = x.view(x.size(0), -1)

        output = self.fc1(output)
        output = self.relu(output)
        self.dropout1(output)

        output = self.fc2(output)
        output = self.relu(output)
        self.dropout1(output)

        output = self.fc3(output)
        output = self.relu(output)
        self.dropout1(output)

        output = self.fc4(output)
        output = self.relu(output)
        self.dropout1(output)

        output = self.fc5(output)

        return output
