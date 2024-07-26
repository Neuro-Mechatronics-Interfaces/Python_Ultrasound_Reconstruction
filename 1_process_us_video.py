import os
import cv2
import utils
import numpy as np


def process_frame(frame):
    """Helper function to process a single frame to create a mask.
    There are a few processing options commented out. Uncomment to see how
    they affect the mask.

    Parameters
    ----------
    frame : np.ndarray
        Input frame.

    Returns
    -------
    np.ndarray
        Mask of the frame.
    """
    # Convert to grayscale
    frame = utils.convertToGrayscale(frame)

    # Sharpen the image
    frame = utils.sharpenImage(frame)

    # Could also apply Gaussian Blur
    # frame = cv2.GaussianBlur(frame, (5, 5), 0)

    # Apply Otsu's thresholding to create a mask
    mask = utils.otsuThresholding(frame)

    # Could also apply kmeans segmentation
    # mask = kmeansSegmentation(frame)

    return mask


def save_frames(video_path, root_dir, crop_rect=None, frame_region=None, save_video=False):
    """Save raw frames and processed masks from a video."""

    # Specify the output directories for the raw frames and masks
    us_dir = os.path.join(root_dir, 'USdata', 'US')
    mask_dir = os.path.join(root_dir, 'USdata', 'USmasks')

    # Create output directories if they don't exist
    os.makedirs(us_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    # Display video length and size
    print("Video length: ", int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    print("Original video resolution: ", cap.get(cv2.CAP_PROP_FRAME_WIDTH), "x", cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if crop_rect:
        print("Modified video resolution: ", crop_rect[3], "x", crop_rect[2])

    start_frame = 1
    end_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_region:
        print("Choosing frame subsection: ", frame_region[0], "to", frame_region[1])
        start_frame = frame_region[0]
        end_frame = frame_region[1]

    frame_number = 1
    filename = os.path.splitext(os.path.basename(video_path))[0]
    while True:

        ret, frame = cap.read()
        if not ret:
            break

        if frame_number < start_frame:
            frame_number += 1
            continue
        elif frame_number > end_frame:
            break

        # Crop the frame if a crop rectangle is provided
        if crop_rect:
            frame = utils.cropFrame(frame, crop_rect)

        # Save the frame
        if save_video:
            raw_frame_path = os.path.join(us_dir, f"{filename}_US{frame_number}.png")
            cv2.imwrite(raw_frame_path, frame)

        # Process the frame to create a mask
        mask = process_frame(frame)

        # Save the mask
        if save_video:
            mask_frame_path = os.path.join(mask_dir, f"{filename}-mask{frame_number}.png")
            cv2.imwrite(mask_frame_path, mask)

        # Visualize the original frame and the mask side by side
        mask_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)  # Convert mask to BGR for consistent display
        combined = np.hstack([frame, mask_color])
        cv2.imshow('Ultrasound Masking', combined)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

        # Press Esc or q to exit
        if cv2.waitKey(255) == 27:
            break

        frame_number += 1

    cap.release()
    cv2.destroyAllWindows()  # Close all OpenCV windows
    print(f"Finished processing {video_path}")

    # Save the information about the processed video for later
    new_size = (crop_rect[3], crop_rect[2]) if crop_rect else (
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    us_params = {
        'n_frames': end_frame - start_frame + 1,
        'frame_rate': cap.get(cv2.CAP_PROP_FPS),
        'original_size': (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))),
        'new_size': new_size,
        'start_frame': start_frame,
        'end_frame': end_frame
    }

    return us_params


if __name__ == '__main__':
    print("This example script demonstrates how to process an ultrasound video")

    # The video path is defined. The root directory needs to be the dataset folder, where the processed frames, masks,
    # and ultrasound parameters will be saved.
    video_path = r'G:\Shared drives\NML_shared\DataShare\Ultrasound_Human_Healthy\062724\video\raw002.mp4'
    root_dir = r'/Python_Ultrasound_Reconstruction/dataset'

    # We can select a subregion of the frames to process by specifying the start and end frames [start, end]. If we want
    # to process all frames, we can set frame_region = None
    frame_region = None  # [100, 105]

    # We can also crop the frames to a specific region by specifying the crop rectangle (x, y, width, height). If we
    # want to process the entire frame, we can set crop_rect = None.
    # Note that for the Butterfly iQ+ system, the crop rectangle selected was (490, 360, 870, 1064)
    crop_rect = (490, 360, 870, 1064)  # Example crop rectangle (x, y, width, height)

    # Finally, we can process frames and masks as images by setting save_video = True. If we only
    # want to visualize the frames and masks without saving them, we can set save_video = False.
    us_params = save_frames(video_path, root_dir, crop_rect, frame_region, save_video=True)

    # Save the ultrasound parameters to a .npz file
    np.savez(os.path.join(root_dir, 'us_params.npz'), **us_params)
