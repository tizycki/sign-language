import sys
import cv2
import os
import argparse

OPENPOSE_PATH = '/sign-language/openpose/'


def keypoints_from_image(image_path, write_json, render_pose):
    try:
        sys.path.append(os.path.join(OPENPOSE_PATH, 'build/python'))
        from openpose import pyopenpose as op
    except ImportError as e:
        print('Error: OpenPose library could not be found.')
        raise e

    # Params
    params = dict()
    params['write_json'] = write_json
    params['model_folder'] = os.path.join(OPENPOSE_PATH, 'models/')
    params["face"] = False
    params['hand'] = True
    params['model_pose'] = 'BODY_25'
    params['render_pose'] = render_pose
    params['keypoint_scale'] = 3  # scales to range [0,1] where (0,0) is top-left corner

    # Starting OpenPose
    op_wrapper = op.WrapperPython()
    op_wrapper.configure(params)
    op_wrapper.start()

    # Read and process Image
    datum = op.Datum()
    image_to_process = cv2.imread(image_path)
    datum.cvInputData = image_to_process
    datum.name = os.path.basename(image_path).split('.')[0]
    op_wrapper.emplaceAndPop([datum])


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--image_path',
        default=os.path.join(OPENPOSE_PATH, 'examples/media/COCO_val2014_000000000241.jpg'),
        help='Path to input image')
    parser.add_argument('--write_json', default='', help='Directory to save keypoints')
    parser.add_argument('--render_pose', default=0, help='Flag for rendering pose in output image')
    args = parser.parse_known_args()

    # Get keypoints
    keypoints_from_image(
        args[0].image_path,
        args[0].write_json,
        args[0].render_pose
    )
