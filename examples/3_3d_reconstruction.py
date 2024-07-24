import os
import cv2
import re
import cv2 as cv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R


def load_video(video_path, show_video=False, crop_width=0):
    # Load the ultrasound image data
    cap = cv2.VideoCapture(video_path)
    img_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    if not cap.isOpened():
        print("Error opening video stream or file")
    frame_data = []
    print("Processing video...", end='')
    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break

        # Do some cropping of the video since there is a lot of black/unused pixels, remove 200 pixel width around the video frame
        frame = frame[crop_width:img_size[1]-crop_width, crop_width:img_size[0]-crop_width]
        frame_data.append(frame)

        if show_video:
            # Display the resulting frame
            cv2.imshow('Frame', frame)
            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

    frame_data = np.array(frame_data)
    cap.release()
    print('Done')



    return frame_data

def euler_to_rotation_matrix(roll, pitch, yaw):
    """
    Convert Euler angles (roll, pitch, yaw) to a rotation matrix.

    Parameters:
    roll (float): Rotation angle around the X-axis in radians
    pitch (float): Rotation angle around the Y-axis in radians
    yaw (float): Rotation angle around the Z-axis in radians

    Returns:
    np.ndarray: The 3x3 rotation matrix
    """
    # Create a rotation object using the 'xyz' convention
    r = R.from_euler('xyz', [roll, pitch, yaw])
    # Convert to a rotation matrix
    rotation_matrix = r.as_matrix()
    return rotation_matrix


def plotPose(ax, R, t, scale=np.array((1, 1, 1)), l_width=2, text=None):
    """
        plot an coordinate system to visualize Pose (R|t)

        ax      : matplotlib axes to plot on
        R       : Rotation as rotation matrix
        t       : translation as np.array (1, 3)
        scale   : Scale as np.array (1, 3)
        l_width : linewidth of axis
        text    : Text written at origin
    """
    x_axis = np.array(([0, 0, 0], [1, 0, 0])) * scale
    y_axis = np.array(([0, 0, 0], [0, 1, 0])) * scale
    z_axis = np.array(([0, 0, 0], [0, 0, 1])) * scale

    x_axis += t
    y_axis += t
    z_axis += t

    x_axis = x_axis @ R
    y_axis = y_axis @ R
    z_axis = z_axis @ R

    ax.plot3D(x_axis[:, 0], x_axis[:, 1], x_axis[:, 2], color='red', linewidth=l_width)
    ax.plot3D(y_axis[:, 0], y_axis[:, 1], y_axis[:, 2], color='green', linewidth=l_width)
    ax.plot3D(z_axis[:, 0], z_axis[:, 1], z_axis[:, 2], color='blue', linewidth=l_width)

    if (text is not None):
        ax.text(x_axis[0, 0], x_axis[0, 1], x_axis[0, 2], "red")

    return None


def interpolate(p_from, p_to, num):
    direction = (p_to - p_from) / np.linalg.norm(p_to - p_from)
    distance = np.linalg.norm(p_to - p_from) / (num - 1)
    ret_vec = []
    for i in range(0, num):
        ret_vec.append(p_from + direction * distance * i)
    return np.array(ret_vec)


def plotImage(ax, img, R, t, size=np.array((1, 1)), img_scale=8):
    img_size = (np.array((img.shape[0], img.shape[1])) / img_scale).astype('int32')
    img = cv.resize(img, ((img_size[1], img_size[0])))

    corners = np.array(([0., 0, 0], [0, size[0], 0],
                        [size[1], 0, 0], [size[1], size[0], 0]))

    center = np.array([img_size[1] / 2, img_size[0] / 2, 0])
    R = cv2.warpAffine(img, R, (img_size[1], img_size[0]))
    corners += t
    xx = np.zeros((img_size[0], img_size[1]))
    yy = np.zeros((img_size[0], img_size[1]))
    zz = np.zeros((img_size[0], img_size[1]))
    l1 = interpolate(corners[0], corners[2], img_size[0])
    xx[:, 0] = l1[:, 0]
    yy[:, 0] = l1[:, 1]
    zz[:, 0] = l1[:, 2]
    l1 = interpolate(corners[1], corners[3], img_size[0])
    xx[:, img_size[1] - 1] = l1[:, 0]
    yy[:, img_size[1] - 1] = l1[:, 1]
    zz[:, img_size[1] - 1] = l1[:, 2]

    for idx in range(0, img_size[0]):
        p_from = np.array((xx[idx, 0], yy[idx, 0], zz[idx, 0]))
        p_to = np.array((xx[idx, img_size[1] - 1], yy[idx, img_size[1] - 1], zz[idx, img_size[1] - 1]))
        l1 = interpolate(p_from, p_to, img_size[1])
        xx[idx, :] = l1[:, 0]
        yy[idx, :] = l1[:, 1]
        zz[idx, :] = l1[:, 2]

    ax.plot_surface(xx, yy, zz, rstride=1, cstride=1, facecolors=img / 255, shade=False)
    return None



def plotImageOld(ax, img, R, t, size=np.array((1, 1)), img_scale=8):
    """
        plot image (plane) in 3D with given Pose (R|t) of corner point

        ax      : matplotlib axes to plot on
        R       : Rotation as rotation matrix
        t       : translation as np.array (1, 3), left down corner of image in real world coord
        size    : Size as np.array (1, 2), size of image plane in real world
        img_scale: Scale to bring down image, since this solution needs 1 face for every pixel it will become very slow on big images
    """
    img_size = (np.array((img.shape[0], img.shape[1])) / img_scale).astype('int32')
    img = cv.resize(img, ((img_size[1], img_size[0])))

    corners = np.array(([0., 0, 0], [0, size[0], 0],
                        [size[1], 0, 0], [size[1], size[0], 0]))

    corners += t
    corners = corners @ R
    xx = np.zeros((img_size[0], img_size[1]))
    yy = np.zeros((img_size[0], img_size[1]))
    zz = np.zeros((img_size[0], img_size[1]))
    l1 = interpolate(corners[0], corners[2], img_size[0])
    xx[:, 0] = l1[:, 0]
    yy[:, 0] = l1[:, 1]
    zz[:, 0] = l1[:, 2]
    l1 = interpolate(corners[1], corners[3], img_size[0])
    xx[:, img_size[1] - 1] = l1[:, 0]
    yy[:, img_size[1] - 1] = l1[:, 1]
    zz[:, img_size[1] - 1] = l1[:, 2]

    for idx in range(0, img_size[0]):
        p_from = np.array((xx[idx, 0], yy[idx, 0], zz[idx, 0]))
        p_to = np.array((xx[idx, img_size[1] - 1], yy[idx, img_size[1] - 1], zz[idx, img_size[1] - 1]))
        l1 = interpolate(p_from, p_to, img_size[1])
        xx[idx, :] = l1[:, 0]
        yy[idx, :] = l1[:, 1]
        zz[idx, :] = l1[:, 2]

    ax.plot_surface(xx, yy, zz, rstride=1, cstride=1, facecolors=img / 255, shade=False)
    return None


def load_config_data(config_path, file_name):
    # Load the time shift text file
    time_shift_file = os.path.join(config_path, file_name)
    time_shift_dict = {}
    with open(time_shift_file, 'r') as f:
        for line in f:
            key, value = line.split(':')
            time_shift_dict[key.strip()] = float(value.strip())  # Clean up whitespacing

    return time_shift_dict

def get_time_shift_lag(time_shift_dict, word):
    """ Helper function to return key with one word"""
    def find_key(time_shift_dict, word):
        for key in time_shift_dict.keys():
            if word in key:
                return key

    return time_shift_dict[find_key(time_shift_dict, 'lag')]

def get_number_from_filename(file_path):
    # Create a new folder with the file number tag
    number_match = re.search(r'(\d+)(?=\.\w+$)', file_path)
    if number_match:
        number = number_match.group()
    else:
        raise ValueError("No number found in the file name")
    return number

def load_data(data_folder):
    data = []
    for file in os.listdir(data_folder):
        if file.endswith('.png'):
            img = cv2.imread(os.path.join(data_folder, file), cv2.IMREAD_GRAYSCALE)
            data.append(img)
    return data

def synchronize_data(frame_data, pose_data, alignment_file):

    print("Synchronizing data...", end='')
    # Get the time shift lag and video frame rate
    time_shift_lag = get_time_shift_lag(alignment_file, 'lag')
    video_rate = alignment_file['Ultrasound Video Frame Rate']
    mocap_rate = alignment_file['Mocap Data Rate']

    # Add the time shift lag to the pose data
    pose_data['Time'] = pose_data['Time'] + time_shift_lag

    # Downsample the pose data to match the video frame rate
    pose_data_ds = downsample_by_averaging(pose_data, len(frame_data))

    # Get the start and end frame from the alignment file
    start_frame = int(alignment_file['Start Frame'])
    end_frame = int(alignment_file['End Frame'])

    # For each video frame (between the specified start and end frame), get the corresponding pose data
    pose_data_sync = pose_data_ds[start_frame:end_frame]

    frame_data_sync = frame_data[start_frame:end_frame]

    print('Done')
    return frame_data_sync, pose_data_sync

def downsample_by_averaging(source_array, target_length):
    # Calculate the segment size
    segment_size = len(source_array) / target_length

    # Create the downsampled array
    downsampled_array = np.array([
        np.mean(source_array[int(i * segment_size):int((i + 1) * segment_size)])
        for i in range(target_length)
    ])
    return downsampled_array

def get_processed_data_folder(number):
    # Check the current and previous directory for the data folder and make "processed_data" folder
    processed_data_folder = ''
    if os.path.exists(os.path.join(os.getcwd(), 'data')):
        if not os.path.exists(os.path.join(os.getcwd(), 'data', 'processed_{}'.format(number))):
            os.makedirs(os.path.join(os.getcwd(), 'data', 'processed_{}'.format(number)))
        processed_data_folder = os.path.join(os.getcwd(), 'data', 'processed_{}'.format(number))
    elif os.path.exists(os.path.join(os.getcwd(), '..', 'data')):
        if not os.path.exists(os.path.join(os.getcwd(), '..', 'data', 'processed_{}'.format(number))):
            os.makedirs(os.path.join(os.getcwd(), '..', 'data', 'processed_{}'.format(number)))
        processed_data_folder = os.path.join(os.getcwd(), '..', 'data', 'processed_{}'.format(number))
    else:
        raise ValueError("No data folder found")
    return processed_data_folder

def transform_pixels_to_metric(frame_data):
    # We use the upper left corner as the origin, but the rest of the image needs to be converted to mm units with the conversion 26.5 pixels per mm
    frame_data_mm = frame_data / 26.5   # 26.5 pixels per mm
    return frame_data_mm

def reconstruction_example(pose_path, video_path,
                           show_3d_reconstruction=False):

    # Load ultrasound video data
    frame_data = load_video(video_path, show_video=False, crop_width=300)

    # Load pose data from a csv file
    pose_data = pd.read_csv(pose_path)

    # Get the config data
    processed_data_folder = get_processed_data_folder(get_number_from_filename(video_path))
    alignment_file = load_config_data(processed_data_folder, 'alignment_data.txt')

    # Synchronize the video and pose data
    frame_data_sync, pose_data_sync = synchronize_data(frame_data, pose_data, alignment_file)

    # Transform the frame data from pixels to metric
    #frame_data_sync_mm = transform_pixels_to_metric(frame_data_sync)
    frame_data_sync_mm = frame_data_sync

    # Get the transform and rotation data from the pose data
    # t should have shape (n, 3) and R with shape (n, 3, 3)
    t = pose_data_sync[:,:3]
    rot_np = pose_data_sync[:,3:]
    R = []
    for idx in range(0, len(rot_np)):
        R.append(euler_to_rotation_matrix(rot_np[idx][0], rot_np[idx][1], rot_np[idx][2]))

    #frame_data_sync = frame_data_sync[::10]
    #R = R[::10]
    #t = t[::10]

    print('Creating Figure')
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the pose and image data
    #for idx in range(0, len(frame_data_sync)):
    #    plotImage(ax, frame_data_sync[idx], R[idx], t[idx])

    img1_idx = 100
    img2_idx = 101
    print("X: ", t[img1_idx][0], "Y: ", t[img1_idx][1], "Z: ", t[img1_idx][2])
    print("Roll: ", rot_np[img1_idx][0], "Pitch: ", rot_np[img1_idx][1], "Yaw: ", rot_np[img1_idx][2])
    plotImage(ax, frame_data_sync_mm[img1_idx], R[img1_idx], t[img1_idx])
    #print("X: ", t[img2_idx][0], "Y: ", t[img2_idx][1], "Z: ", t[img2_idx][2])
    #print("Roll: ", rot_np[img2_idx][0], "Pitch: ", rot_np[img2_idx][1], "Yaw: ", rot_np[img2_idx][2])
    #plotImage(ax, frame_data_sync_mm[img2_idx], R[img2_idx], t[img2_idx])
    #plt.xlabel('X')
    #plt.ylabel('Y')
    #ax.set_zlabel('Z')
    plt.show()

if __name__ == '__main__':
    # This example demonstrates visualizing 2D images in a 3D space
    # The pose data is processed from '2_pose_extraction.py' file

    # Notes:
    # 1) From pixel to metric, 265 pixels is the equivalent of 1cm spacing in the images (or 26.5 pixels per mm).
    # 2) Image data is in pixels, pose data is in mm
    # 3) The offset from the origin point to the middle edge of the probe is (-65, -70, -115) mm

    # Path to the folder with the mocap data
    pose_path = os.path.join(os.getcwd(), '..', 'data', 'processed_002')
    pose_name = 'pose_data.csv'
    trc_path = os.path.join(pose_path, pose_name)

    # Path to the folder with the ultrasound data
    ultrasound_path = r'G:\Shared drives\NML_shared\DataShare\Ultrasound_Human_Healthy\062724\video'
    video_filename = 'raw002.mp4'
    video_path = os.path.join(ultrasound_path, video_filename)


    # perform 3d reconstruction example
    reconstruction_example(trc_path, video_path,
                           show_3d_reconstruction=True)

