import numpy as np
import math
from utils.keypoints import rescale_keypoints


def rotate_keypoints(keypoints):
    angle = math.radians(np.random.uniform(-30, 30))
    rotated_keypoints = np.copy(keypoints)

    rotated_keypoints[:, 0] = math.cos(angle) * keypoints[:, 0] - math.sin(angle) * keypoints[:, 1]
    rotated_keypoints[:, 1] = math.sin(angle) * keypoints[:, 0] + math.cos(angle) * keypoints[:, 1]

    return rescale_keypoints(rotated_keypoints)