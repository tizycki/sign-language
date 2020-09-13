import json
import numpy as np


def read_keypoints(file_path):
    # Select keypoints to extract
    keypoints_map = {
        'nose': 0,
        'neck': 1,
        'right_shoulder': 2,
        'right_elbow': 3,
        'right_wrist': 4,
        'left_shoulder': 5,
        'left_elbow': 6,
        'left_wrist': 7,
        'middle_hip': 8,
        'right_hip': 9,
        'left_hip': 12,
        'right_eye': 15,
        'left_eye': 16,
        'right_ear': 17,
        'left_ear': 18,
    }
    keypoints_filter = list(keypoints_map.values())

    # Read json file with keypoints
    keypoints = None
    with open(file_path, 'r') as f:
        for row in f.readlines():
            keypoints = json.loads(row)

    # Extract and concatenate selected keypoints
    results = np.append([
        np.array(keypoints['people'][0]['pose_keypoints_2d']).reshape(-1, 3)[:, :2],
        np.array(keypoints['people'][0]['hand_left_keypoints_2d']).reshape(-1, 3)[:, :2],
        np.array(keypoints['people'][0]['hand_right_keypoints_2d']).reshape(-1, 3)[:, :2]],
        axis=1
    )
    return results.reshape([1, len(results), 2])
