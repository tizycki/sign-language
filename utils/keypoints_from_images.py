import sys
import cv2
import os
import argparse
import time

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
    parser.add_argument("--image_dir", default='', help='Directory of input images')
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
    params['keypoint_scale'] = 3  # scales to range [0,1] where (0,0) is top-left corner

    # Starting OpenPose
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    # Read frames on directory
    imagePaths = op.get_images_on_directory(args[0].image_dir);
    start = time.time()

    # Process and display images
    for imagePath in imagePaths:
        datum = op.Datum()
        imageToProcess = cv2.imread(imagePath)
        datum.cvInputData = imageToProcess
        opWrapper.emplaceAndPop([datum])

    end = time.time()
    print("Keypoints extraction successfully finished. Total time: " + str(end - start) + " seconds")

except Exception as e:
    print(e)
    sys.exit(-1)
