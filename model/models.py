from torch import nn


class StopPoseNet(nn.Module):

    def __init__(self):
        super(StopPoseNet, self).__init__()

        self.fc1 = nn.Linear(114, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 128)
        self.fc4 = nn.Linear(128, 32)
        self.fc5 = nn.Linear(32, 2)
        self.dropout1 = nn.Dropout(0.1)
        self.relu = nn.ReLU()

    def forward(self, x):
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

    def __init__(self, out_feature_num):
        super(SequenceRecognitionNet, self).__init__()

        self.fc1 = nn.Linear(5700, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, 128)
        self.fc5 = nn.Linear(128, out_feature_num)
        self.dropout1 = nn.Dropout(0.1)
        self.relu = nn.ReLU()

    def forward(self, x):
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
