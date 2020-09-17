from utils.keypoints import rescale_keypoints, read_keypoints
from torch.utils.data import Dataset
import numpy as np


class OneClassKeypointClassificationDataset(Dataset):

    def __init__(self, annotations, keypoints_transform=None):
        super().__init__()
        self.annotations = annotations
        self.keypoints_transform = keypoints_transform

    def __len__(self):
        return self.annotations.shape[0]

    def __getitem__(self, index):
        if self.keypoints_transform:
            keypoints = self.keypoints_transform(self.load_from_disk(index))
        else:
            keypoints = self.load_from_disk(index)

        target = self.load_target(index)

        return keypoints, target

    def load_from_disk(self, index):
        keypoints = rescale_keypoints(read_keypoints(self.annotations.iloc[index].keypoints_path))

        return keypoints

    def load_target(self, index):

        return self.annotations.iloc[index]['stop_pose']


class SequenceKeypointsDataset(Dataset):

    def __init__(self, annotations, keypoints_transform=None):
        super().__init__()
        self.annotations = annotations
        self.keypoints_transform = keypoints_transform

    def __len__(self):
        return self.annotations.shape[0]

    def __getitem__(self, index):
        if self.keypoints_transform:
            keypoints_sequence = self.keypoints_transform(self.load_from_disk(index))
        else:
            keypoints_sequence = self.load_from_disk(index)

        target = self.load_target(index)

        return keypoints_sequence, target

    def load_from_disk(self, index):
        keypoints_sequence = []
        for keypoints_path in self.annotations.iloc[index].keypoints_path:
            keypoints = rescale_keypoints(read_keypoints(keypoints_path))
            keypoints_sequence.append(keypoints)
        return np.array(keypoints_sequence).reshape([len(keypoints_sequence), -1, 2])

    def load_target(self, index):

        return self.annotations.iloc[index].label_id
