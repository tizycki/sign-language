from utils.keypoints import rescale_keypoints, read_keypoints
from torch.utils.data import Dataset
import numpy as np
import pandas as pd


class OneClassKeypointClassificationDataset(Dataset):
    """
    Dataset for one class classification of keypoints (used to train StopPoseNet)
    """
    def __init__(self, annotations: pd.DataFrame, keypoints_transform=None):
        """
        Initialize dataset with annotations and transformer
        :param annotations: DataFrame with X and y
        :param keypoints_transform: Transformer for X
        """
        super().__init__()
        self.annotations = annotations
        self.keypoints_transform = keypoints_transform

    def __len__(self) -> int:
        """
        Get number of keypoints in DataFrame
        :return: DataFrame.shape[0]
        """
        return self.annotations.shape[0]

    def __getitem__(self, index: int) -> (np.array, int):
        """
        Get single pair of (X, y) from dataframe
        :param index: index of element in df
        :return: transformed X and target value
        """
        # Load from disk and transform (if needed)
        if self.keypoints_transform:
            keypoints = self.keypoints_transform(self.load_from_disk(index))
        else:
            keypoints = self.load_from_disk(index)

        # Load target
        target = self.load_target(index)

        return keypoints, target

    def load_from_disk(self, index: int) -> np.array:
        """
        Function to load keypoints from JSON specified in row, and rescale those keypoints
        :param index: index of element in df
        :return: numpy array of keypoints (from single frame)
        """
        # Load from disk and rescale keypoints
        keypoints = rescale_keypoints(read_keypoints(self.annotations.iloc[index].keypoints_path))

        return keypoints

    def load_target(self, index: int) -> float:
        """
        Load target value from specified row
        :param index: index of element in df
        :return: stop pose label
        """
        return self.annotations.iloc[index]['stop_pose']


class SequenceKeypointsDataset(Dataset):
    """
    Dataset for one class classification of keypoints sequence (used to train SequenceRecognitionNet)
    """
    def __init__(self, annotations: pd.DataFrame, keypoints_transform=None):
        """
        Initialize dataset with annotations and transformer
        :param annotations: DataFrame with X and y
        :param keypoints_transform: Transformer for X
        """
        super().__init__()
        self.annotations = annotations
        self.keypoints_transform = keypoints_transform

    def __len__(self) -> int:
        """
        Get number of sequences in DataFrame
        :return: DataFrame.shape[0]
        """
        return self.annotations.shape[0]

    def __getitem__(self, index: int):
        """
        Get single pair of (X, y) from dataframe
        :param index: index of element in df
        :return: transformed X and target value
        """
        # Load from disk and transform (if needed)
        if self.keypoints_transform:
            keypoints_sequence = self.keypoints_transform(self.load_from_disk(index))
        else:
            keypoints_sequence = self.load_from_disk(index)

        target = self.load_target(index)

        return keypoints_sequence, target

    def load_from_disk(self, index: int) -> np.array:
        """
        Function to load keypoints sequence from list of JSON files specified in row, and rescale those keypoints
        :param index: index of element in df
        :return: numpy array of keypoints (from single frame)
        """
        # Load from disk, rescale keypoints and append results
        keypoints_sequence = []
        for keypoints_path in self.annotations.iloc[index].keypoints_path:
            keypoints = rescale_keypoints(read_keypoints(keypoints_path))
            keypoints_sequence.append(keypoints)

        return np.array(keypoints_sequence).reshape([len(keypoints_sequence), -1, 2])

    def load_target(self, index: int) -> int:
        """
        Load label from specified row
        :param index: index of element in df
        :return: sequence label
        """
        return self.annotations.iloc[index].label_id
