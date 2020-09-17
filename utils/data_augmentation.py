import numpy as np
import math
from utils.keypoints import rescale_keypoints


def rotate_keypoints(keypoints: np.array, min_angle: float, max_angle: float) -> np.array:
    """
    Rotate keypoints by random degree in range specified by user. Each keypoint rotated by the same degree.
    :param keypoints: numpy array of keypoints (single frame)
    :param min_angle: minimum degree for rotation
    :param max_angle: maximum degree for rotation
    :return: numpy array of keypoints rotated by random degree
    """
    # Copy input array
    rotated_keypoints = np.copy(keypoints)

    # Get random degree
    angle = math.radians(np.random.uniform(min_angle, max_angle))

    # Rotate each keypoint
    rotated_keypoints[:, 0] = math.cos(angle) * keypoints[:, 0] - math.sin(angle) * keypoints[:, 1]
    rotated_keypoints[:, 1] = math.sin(angle) * keypoints[:, 0] + math.cos(angle) * keypoints[:, 1]

    return rescale_keypoints(rotated_keypoints)


def rotate_keypoints_sequence(keypoints_sequence: np.array, min_angle: float, max_angle: float) -> np.array:
    """
    Rotate sequence of keypoints by random degree in range specified by user. Each frame rotated by the same degree.
    :param keypoints_sequence: numpy array of keypoints sequence
    :param min_angle: minimum degree for rotation
    :param max_angle: maximum degree for rotation
    :return: numpy array of keypoints sequence rotated by random degree
    """
    # Copy input array
    rotated_keypoints_sequence = np.copy(keypoints_sequence)

    # Get random degree
    angle = math.radians(np.random.uniform(min_angle, max_angle))

    # Rotate each frame
    rotated_keypoints_sequence[:, :, 0] = math.cos(angle) * keypoints_sequence[:, :, 0] - math.sin(angle) * keypoints_sequence[:, :, 1]
    rotated_keypoints_sequence[:, :, 1] = math.sin(angle) * keypoints_sequence[:, :, 0] + math.cos(angle) * keypoints_sequence[:, :, 1]
    rotated_keypoints_sequence[:] = rescale_keypoints(rotated_keypoints_sequence[:])

    return np.array(rotated_keypoints_sequence).reshape(len(keypoints_sequence), -1, 2)
