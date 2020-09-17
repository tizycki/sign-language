import json
import numpy as np
from tqdm import tqdm

# Body keypoints to extract
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
# Set number of output keypoints
OUTPUT_KEYPOINTS_NUM = len(keypoints_map) + 21 + 21


def read_keypoints(file_path: str) -> np.array:
    """
    Read body, left-hand, right-hand keypoints from JSON file
    :param file_path: path to json file with skeleton keypoints
    :return: numpy array of keypoints
    """
    # Set filter for body keypoints
    keypoints_filter = list(keypoints_map.values())

    # Read json file with keypoints
    keypoints = None
    with open(file_path, 'r') as f:
        for row in f.readlines():
            keypoints = json.loads(row)

    # Extract and concatenate selected keypoints
    if keypoints['people']:
        results = np.concatenate([
            np.array(keypoints['people'][0]['pose_keypoints_2d']).reshape(-1, 3)[keypoints_filter, :2],
            np.array(keypoints['people'][0]['hand_left_keypoints_2d']).reshape(-1, 3)[:, :2],
            np.array(keypoints['people'][0]['hand_right_keypoints_2d']).reshape(-1, 3)[:, :2]],
            axis=0
        )
    else:
        results = np.zeros((1, 0))

    return results


def rescale_keypoints(keypoints: np.array) -> np.array:
    """
    Rescale coordinates of keypoints array to range [0,1]. Output is a bounding-box with keypoints
    :param keypoints: numpy array of keypoints for single video frame
    :return: numpy array of rescaled keypoints
    """
    # Get params for min-max scaling
    max_coordinates = np.max(keypoints, axis=0)
    min_coordinates = np.min(keypoints, axis=0)

    # Filter
    denominator = max_coordinates - min_coordinates
    denominator[(denominator == 0.0)] = 1.0
    keypoints = (keypoints - min_coordinates) / denominator

    return keypoints


def read_rescale_keypoints_list(keypoints_json_paths: list) -> np.array:
    """
    Read and rescale list of keypoints (as list of JSON files)
    :param keypoints_json_paths: list of paths to JSON files with keypoints
    :return: numpy array of all rescaled keypoints
    """
    results = []
    for keypoints_json in tqdm(keypoints_json_paths):
        keypoints = rescale_keypoints(read_keypoints(keypoints_json))
        results.append(keypoints)

    return np.array(results).reshape([len(keypoints_json_paths), OUTPUT_KEYPOINTS_NUM, 2])


def keypoints_sequence_padding(keypoints_sequence: np.array, output_length: int) -> np.array:
    """
    Performs sequence padding with last frame. If sequence is too long, it returns first N frames
    :param keypoints_sequence: numpy array of keypoints sequence
    :param output_length: length of output sequence
    :return: fixed shape (output_length param) numpy array of sequence after padding
    """
    output_sequence = np.copy(keypoints_sequence)
    input_seq_length = keypoints_sequence.shape[0]

    if input_seq_length < output_length:
        pad_sequence = np.zeros([output_length - input_seq_length, keypoints_sequence.shape[1], keypoints_sequence.shape[2]])
        pad_sequence[:] = keypoints_sequence[input_seq_length - 1]
        output_sequence = np.append(output_sequence, pad_sequence, axis=0)

    return output_sequence[:output_length]
