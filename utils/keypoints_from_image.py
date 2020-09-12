import sys
import cv2
import os
import argparse

OPENPOSE_PATH = '/sign-language/openpose/'

try:
    try:
        sys.path.append(os.path.join(OPENPOSE_PATH, 'build/python'))
        from openpose import pyopenpose as op
        print('Done importing')
    except ImportError as e:
        print('Error: OpenPose library could not be found.')
        raise e

    # Flags
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--image_path',
        default=os.path.join(OPENPOSE_PATH, 'examples/media/COCO_val2014_000000000241.jpg'),
        help='Path to input image')
    parser.add_argument('--write_json', default='', help='Directory to save keypoints')
    parser.add_argument('--render_pose', default=0, help='Flag for rendering pose in output image')
    args = parser.parse_known_args()

    # Params
    params = dict()
    params["model_folder"] = os.path.join(OPENPOSE_PATH, 'models/')
    params["face"] = False
    params['hand'] = True
    params['model_pose'] = 'BODY_25'
    params['write_json'] = args[0].write_json
    params['render_pose'] = args[0].render_pose

    # Starting OpenPose
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    # Process Image
    datum = op.Datum()
    imageToProcess = cv2.imread(args[0].image_path)
    datum.cvInputData = imageToProcess
    opWrapper.emplaceAndPop([datum])

except Exception as e:
    print(e)
    sys.exit(-1)
