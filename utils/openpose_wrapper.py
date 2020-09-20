import sys
import cv2
import os
import argparse
import time
from tqdm import tqdm

OPENPOSE_PATH = '/sign-language/openpose/'


def keypoints_from_images(image_dir, write_json, write_images):
    """
    Use openpose tool to convert images into JSON files with keypoints
    :param image_dir: directory of input images
    :param write_json: directory of output JSON files with keypoints
    :param write_images: directory of output images - with rendered skeleton
    :return:
    """
    # Set timer
    start = time.time()

    # Load openpose module
    try:
        sys.path.append(os.path.join(OPENPOSE_PATH, 'build/python'))
        from openpose import pyopenpose as op
    except ImportError as e:
        print('Error: OpenPose library could not be found.')
        raise e

    # Set openpose params
    params = dict()
    params['write_json'] = write_json
    params['write_images'] = write_images
    params['model_folder'] = os.path.join(OPENPOSE_PATH, 'models/')
    params['model_pose'] = 'BODY_25'
    params["face"] = False
    params['hand'] = True
    params['keypoint_scale'] = 0  # 3 -> scales to range [0,1] where (0,0) is top-left corner
    params['display'] = 0
    params['render_pose'] = 2
    params['alpha_pose'] = 0.5

    # Create openpose python wrapper
    op_wrapper = op.WrapperPython()
    op_wrapper.configure(params)
    op_wrapper.start()

    # Read frames from input directory
    image_paths = op.get_images_on_directory(image_dir)

    # Process and display images
    for image_path in tqdm(image_paths):
        datum = op.Datum()
        image_to_process = cv2.imread(image_path)
        datum.cvInputData = image_to_process
        datum.name = os.path.basename(image_path).split('.')[0]
        op_wrapper.emplaceAndPop([datum])

    # Print execution time
    end = time.time()
    print("Keypoints extraction successfully finished. Total time: " + str(end - start) + " seconds")


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", default='', help='Directory to input images')
    parser.add_argument('--write_json', default='', help='Directory to save keypoints')
    parser.add_argument('--write_images', default='', help='Directory to save rendered images')

    args = parser.parse_known_args()

    # Get keypoints
    keypoints_from_images(
        args[0].image_dir,
        args[0].write_json,
        args[0].write_images
    )
