import numpy as np
import math
from utils.keypoints import rescale_keypoints


def rotate_keypoints(keypoints, min_angle, max_angle):
    angle = math.radians(np.random.uniform(min_angle, max_angle))
    rotated_keypoints = np.copy(keypoints)

    rotated_keypoints[:, 0] = math.cos(angle) * keypoints[:, 0] - math.sin(angle) * keypoints[:, 1]
    rotated_keypoints[:, 1] = math.sin(angle) * keypoints[:, 0] + math.cos(angle) * keypoints[:, 1]

    return rescale_keypoints(rotated_keypoints)
