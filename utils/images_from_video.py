import argparse
import cv2
import os
import glob


def video_to_images(video_path, output_dir, file_name_prefix, output_fps, rotate, start_frame=0, end_frame=-1):

    # Video capture
    frame_count = 0
    video_capture = cv2.VideoCapture(video_path)

    # Video info
    input_fps = video_capture.get(cv2.CAP_PROP_FPS)
    frames_total = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = int(frames_total / input_fps)
    output_frames_total = int(frames_total * int(output_fps) / input_fps)
    print(f"Input video info:\nInput FPS: {input_fps}\nDuration: {video_duration}")

    if end_frame == -1:
        end_frame = output_frames_total

    # Create subdirectory for results
    video_name = f"{os.path.basename(video_path).split('.')[0].replace(' ', '_')}_fps{int(output_fps):02d}"
    output_path = os.path.join(output_dir, video_name)

    # Create or erase sub-folder for results
    if not os.path.exists(output_path):
        os.mkdir(output_path)
        print(f"Create output sub-folder: {video_name} in output_directory: {output_dir}")
    else:
        files_to_remove = glob.glob(f"{output_path}/*")
        for file in files_to_remove:
            os.remove(file)
        print(f"Removed content of directory: {output_path}")

    # Iterate through video and save images
    while video_capture.isOpened():
        # Set output fps
        video_capture.set(cv2.CAP_PROP_POS_MSEC, frame_count * 1000 / float(output_fps))

        # Read frame
        has_frame, frame = video_capture.read()
        print(f"Processing frame of index: {frame_count:06d} / ~{output_frames_total:06d}. \
            Frame read status: {has_frame}")

        # Save frame
        if has_frame:
            # Skip if not in frame range
            if frame_count < start_frame:
                print('Frame skipped.')
                frame_count += 1
                continue
            if frame_count > end_frame:
                print('Reached end_frame defined by user.')
                break

            # Rotate if specified
            if rotate:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

            # Save file
            cv2.imwrite(os.path.join(output_path, f"{file_name_prefix}_{frame_count:06d}.jpg"), frame)
            frame_count += 1
        else:
            break

    # Release video capture
    video_capture.release()
    print("Done.")


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--video_path", help="Path to video")
    a.add_argument("--output_dir_path", help="Path to output directory")
    a.add_argument("--image_prefix", help="Image files name prefix")
    a.add_argument("--output_fps", help="Frames per second of output")
    a.add_argument("--start_frame", help="Index of frame to start from")
    a.add_argument("--end_frame", help="Index of frame to stop at")

    args = a.parse_args()
    print("Initializing process with parameters:")
    print(args)

    video_to_images(
        args.video_path,
        args.output_dir_path,
        args.image_prefix,
        args.output_fps
    )
