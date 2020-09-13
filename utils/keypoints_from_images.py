import sys
import cv2
import os
import argparse
import time
from tqdm import tqdm

OPENPOSE_PATH = '/sign-language/openpose/'


def keypoints_from_images(image_dir, write_json, render_pose=-1, alpha_pose=0.5, hand_render=-1, hand_alpha_pose=0.5):
    try:
        sys.path.append(os.path.join(OPENPOSE_PATH, 'build/python'))
        from openpose import pyopenpose as op
    except ImportError as e:
        print('Error: OpenPose library could not be found.')
        raise e

    # Params
    params = dict()
    params['write_json'] = write_json
    params['render_pose'] = render_pose
    params['alpha_pose'] = alpha_pose
    params['hand_render'] = hand_render
    params['hand_alpha_pose'] = hand_alpha_pose
    params['model_folder'] = os.path.join(OPENPOSE_PATH, 'models/')
    params["face"] = False
    params['hand'] = True
    params['model_pose'] = 'BODY_25'
    params['keypoint_scale'] = 3  # scales to range [0,1] where (0,0) is top-left corner

    # Starting OpenPose
    op_wrapper = op.WrapperPython()
    op_wrapper.configure(params)
    op_wrapper.start()

    # Read frames on directory
    image_paths = op.get_images_on_directory(image_dir)
    start = time.time()

    # Process and display images
    for image_path in tqdm(image_paths):
        datum = op.Datum()
        image_to_process = cv2.imread(image_path)
        datum.cvInputData = image_to_process
        datum.name = os.path.basename(image_path).split('.')[0]
        op_wrapper.emplaceAndPop([datum])

    end = time.time()
    print("Keypoints extraction successfully finished. Total time: " + str(end - start) + " seconds")


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", default='', help='Directory of input images')
    parser.add_argument('--write_json', default='', help='Directory to save keypoints')
    parser.add_argument('--render_pose', default=-1, help='Flag for rendering pose in output image')
    parser.add_argument('--alpha_pose', default=0.5, help='Alpha for rendering pose in output image')
    parser.add_argument('--hand_render', default=-1, help='Flag for rendering hand in output image')
    parser.add_argument('--hand_alpha_pose', default=0.5, help='Alpha for rendering hand in output image')

    args = parser.parse_known_args()

    # Get keypoints
    keypoints_from_images(
        args[0].image_dir,
        args[0].write_json,
        args[0].render_pose,
        args[0].alpha_pose,
        args[0].hand_render,
        args[0].hand_alpha_pose
    )
