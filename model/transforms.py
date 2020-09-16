from utils.data_augmentation import rotate_keypoints


class RotateKeypoints(object):

    def __init__(self, min_angle, max_angle):
        self.min_angle = min_angle
        self.max_angle = max_angle

    def __call__(self, keypoints):
        return rotate_keypoints(keypoints, self.min_angle, self.max_angle)
