from utils.data_augmentation import rotate_keypoints, rotate_keypoints_sequence
from utils.keypoints import keypoints_sequence_padding


class RotateKeypoints(object):
    """
    Transformer for keypoints rotation
    """
    def __init__(self, min_angle, max_angle):
        self.min_angle = min_angle
        self.max_angle = max_angle

    def __call__(self, keypoints):
        return rotate_keypoints(keypoints, self.min_angle, self.max_angle)


class RotateKeypointsSequence(object):
    """
    Transformer for rotation of keypoints sequence
    """
    def __init__(self, min_angle, max_angle):
        self.min_angle = min_angle
        self.max_angle = max_angle

    def __call__(self, keypoints_sequence):
        return rotate_keypoints_sequence(keypoints_sequence, self.min_angle, self.max_angle)


class KeypointsSequencePadding(object):
    """
    Transformer for padding of keypoints sequence
    """
    def __init__(self, output_length):
        self.output_length = output_length

    def __call__(self, keypoints_sequence):
        return keypoints_sequence_padding(keypoints_sequence, self.output_length)
