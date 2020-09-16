from utils.keypoints import rescale_keypoints, read_keypoints
from torch.utils.data import Dataset


class OneClassKeypointClassificationDataset(Dataset):

    def __init__(self, annotations, keypoints_transform=None):
        super().__init__()
        self.annotations = annotations
        self.keypoints_transform = keypoints_transform

    def __len__(self):
        return self.annotations.shape[0]

    def __getitem__(self, index):
        if self.keypoints_transform:
            keypoints = self.load_from_disk(index)
        else:
            keypoints = self.keypoints_transform(self.load_from_disk(index))

        target = self.load_target(index)

        return keypoints, target

    def load_from_disk(self, index):
        keypoints = rescale_keypoints(read_keypoints(self.annotations.iloc[index].keypoints_path))

        return keypoints

    def load_target(self, index):

        return self.annotations.iloc[index]['stop_pose']
