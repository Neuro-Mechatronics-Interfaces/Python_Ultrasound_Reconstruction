import os
import cv2
import numpy as np

def crop_frame(frame, crop_rect):
    """Crop the frame to the specified rectangle."""
    x, y, w, h = crop_rect
    return frame[y:y+h, x:x+w]

def otsuThresholding(image):
    """Perform Otsu's thresholding to segment an image.

    Parameters
    ----------
    image : np.ndarray
        Grayscale image to be segmented.

    Returns
    -------
    np.ndarray
        Binary segmented image.

    """
    blur = cv2.GaussianBlur(image, (5, 5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th

def kmeansSegmentation(image, K=2):
    """Perform K-means clustering to segment an image.

    Parameters
    ----------
    image : np.ndarray
        Grayscale or color image to be segmented.

    K : int
        Number of clusters.

    Returns
    -------
    np.ndarray
        Segmented image.

    """
    Z = image.reshape((-1, 3))
    Z = np.float32(Z)

    # Define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Now convert back into uint8, and make original image
    centers = np.uint8(centers)
    res = centers[labels.flatten()]
    segmented_image = res.reshape((image.shape))

    return segmented_image

def sharpenImage(image):
    """Sharpen an image using a kernel.

    Parameters
    ----------
    image : np.ndarray
        Input image.

    Returns
    -------
    np.ndarray
        Sharpened image.

    """
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened_image = cv2.filter2D(image, -1, kernel)
    return sharpened_image

def process_frame(frame):
    """Process a single frame to create a mask. There are a few processing options commented out. Uncomment to see how
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
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Sharpen the image
    frame = sharpenImage(frame)

    # Apply Gaussian Blur (optional, can be adjusted or removed)
    #frame = cv2.GaussianBlur(frame, (5, 5), 0)

    # Apply Otsu's thresholding to create a mask (or)
    mask = otsuThresholding(frame)

    # Apply kmeans segmentation to create a mask
    #mask = kmeansSegmentation(frame)

    # Perform edge detection
    #mask = cv2.Canny(frame, 0, 255)

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
        x, y, w, h = crop_rect
        print("Modified video resolution: ", y+h, "x", x+w)

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
            frame = crop_frame(frame, crop_rect)

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
    cv2.destroyAllWindows() # Close all OpenCV windows
    print(f"Finished processing {video_path}")


if __name__ == '__main__':

    print("This example script demonstrates how to process an ultrasound video")

    # The video path as well as the output path needs to be defined at the top. The video file path is pretty
    # straightforward, but the root directory needs to be the dataset folder, where the processed frames and masks will
    # be saved.
    video_path = r'G:\Shared drives\NML_shared\DataShare\Ultrasound_Human_Healthy\062724\video\raw001.mp4'
    root_dir = r'C:\Users\HP\Documents\Github\Ultrasound\Python_Ultrasound_Reconstruction\dataset'

    # We can select a subregion of the frames to process by specifying the start and end frames [start, end]. If we want
    # to process all frames, we can set frame_region = None
    frame_region = None  # [100, 105]
    frame_region = [100, 110]

    # We can also crop the frames to a specific region by specifying the crop rectangle (x, y, width, height). If we
    # want to process the entire frame, we can set crop_rect = None.
    # Note that for the Butterfly iQ+ system, the crop rectangle selected was (490, 360, 870, 1064)
    crop_rect = (490, 360, 870, 1064)  # Example crop rectangle (x, y, width, height)

    # Finally, we can choose to save the processed frames and masks as images by setting save_video = True. If we only
    # want to visualize the frames and masks without saving them, we can set save_video = False.
    save_frames(video_path, root_dir, crop_rect, frame_region, save_video=False)
